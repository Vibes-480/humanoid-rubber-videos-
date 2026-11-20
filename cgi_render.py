"""Render exported creature geometry as 3D CGI with Blender (Cycles, CPU).

Run with the Blender Python environment (bpy needs Python 3.11):
    .venv-blender/bin/python cgi_render.py cgi_out/strand_f0040.npz [--samples 64]

Writes <stem>_render.png (RGBA, transparent background) next to the .npz. The camera is
orthographic and matched to the video frame, so the render lines up with the pose tracking.
Blender coordinates: X = x / H, Y = depth / H (away from camera), Z = -y / H.
"""
import math
import sys
import time

import bpy
import numpy as np


LIGHT = 0.3
FAST = False                     # preview quality: no subsurface, fewer bounces


def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class Scene:
    def __init__(self, d):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.d = d
        self.W, self.H = int(d["width"]), int(d["height"])
        sc = self.sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.render.resolution_x, sc.render.resolution_y = self.W, self.H
        sc.render.film_transparent = True
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.cycles.use_denoising = True
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
        # Camera: orthographic, looking along +Y, framing exactly the video frame.
        cam = bpy.data.cameras.new("cam")
        cam.type = "ORTHO"
        cam.ortho_scale = self.W / self.H
        cam.clip_start, cam.clip_end = 0.01, 50
        co = bpy.data.objects.new("cam", cam)
        co.location = (self.W / (2 * self.H), -10.0, -0.5)
        co.rotation_euler = (math.pi / 2, 0, 0)
        sc.collection.objects.link(co)
        sc.camera = co
        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        bgn = world.node_tree.nodes["Background"]
        bgn.inputs[0].default_value = (0.55, 0.55, 0.6, 1)
        bgn.inputs[1].default_value = 0.6
        sc.world = world
        self.center = np.array([self.W / (2 * self.H), 0.0, -0.5])

    def bl(self, p):
        """Video-pixel coordinates (x, y, depth) -> Blender coordinates."""
        p = np.asarray(p, np.float32)
        return np.stack([p[..., 0] / self.H, p[..., 2] / self.H, -p[..., 1] / self.H], -1)

    def sun(self, name, direction_deg, strength, angle_deg=8, color=(1, 1, 1)):
        l = bpy.data.lights.new(name, "SUN")
        l.energy, l.angle, l.color = strength, math.radians(angle_deg), color
        o = bpy.data.objects.new(name, l)
        o.rotation_euler = tuple(math.radians(a) for a in direction_deg)
        self.sc.collection.objects.link(o)

    def three_point(self, key=4.5, fill=1.5, rim=6.0):
        # XYZ Euler on a sun pointing down -Z. +X tilt sends light away from the camera
        # (lit from the front); -X tilt sends it toward the camera (back light).
        self.sun("key", (55, 0, -35), key, 6)                   # front, upper left
        self.sun("fill", (70, 0, 45), fill, 25, (0.85, 0.9, 1.0))   # front right, soft
        self.sun("rim", (-40, 0, 20), rim, 4)                   # from behind, above

    def area(self, name, loc, size, energy, color=(1, 1, 1), target=None):
        from mathutils import Vector
        l = bpy.data.lights.new(name, "AREA")
        l.shape, l.size, l.size_y = "RECTANGLE", size, size * 0.6
        l.energy, l.color = energy, color
        o = bpy.data.objects.new(name, l)
        o.location = loc
        tgt = Vector(self.center if target is None else target)
        o.rotation_euler = (tgt - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        self.sc.collection.objects.link(o)
        return o

    def studio_lights(self, k=1.0):
        """Soft lighting matched to the room: ceiling panels in front, white-wall bounce,
        and a rim light behind for edge highlights. Units: 1 = frame height."""
        cx, cz = self.center[0], self.center[2]
        self.area("key", (cx - 1.4, -2.6, cz + 2.0), 1.6, 900 * k, (1.0, 0.97, 0.92))
        self.area("fill", (cx + 1.8, -3.0, cz + 0.4), 2.5, 380 * k, (0.9, 0.94, 1.0))
        self.area("bounce", (cx, -3.0, cz - 1.6), 2.5, 150 * k, (1.0, 0.95, 0.95))
        self.area("rim_l", (cx - 1.3, 2.2, cz + 1.4), 1.0, 700 * k)
        self.area("rim_r", (cx + 1.3, 2.2, cz + 1.2), 1.0, 500 * k)

    def shadow_wall(self, depth):
        """Invisible wall behind the creature that only receives its shadow."""
        bpy.ops.mesh.primitive_plane_add(size=1)
        o = bpy.context.object
        o.scale = (self.W / self.H * 3, 3, 1)
        o.rotation_euler = (math.pi / 2, 0, 0)
        o.location = (self.center[0], depth, self.center[2])
        o.is_shadow_catcher = True
        return o

    # -- materials ------------------------------------------------------------------
    def material(self, name):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        return m, nt, nt.nodes["Principled BSDF"]

    def attr(self, nt, name, kind="GEOMETRY"):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name, a.attribute_type = name, kind
        return a

    # -- geometry helpers -----------------------------------------------------------
    def hair(self, name, pts, radius, attrs, mat):
        """pts [n,K,3] Blender coords, radius [n,K]; attrs: {name: [n,C] per-curve floats}."""
        n, K = pts.shape[:2]
        c = bpy.data.hair_curves.new(name)
        c.add_curves([K] * n)
        c.position_data.foreach_set("vector", pts.astype(np.float32).ravel())
        r = c.attributes.get("radius") or c.attributes.new("radius", "FLOAT", "POINT")
        r.data.foreach_set("value", radius.astype(np.float32).ravel())
        for an, av in attrs.items():
            av = np.asarray(av, np.float32)
            if av.ndim == 2 and av.shape[1] == 3:
                av = np.concatenate([av, np.ones((len(av), 1), np.float32)], 1)
            if av.ndim == 2:
                a = c.attributes.new(an, "FLOAT_COLOR", "CURVE")
                a.data.foreach_set("color", av.ravel())
            else:
                a = c.attributes.new(an, "FLOAT", "CURVE")
                a.data.foreach_set("value", av.ravel())
        c.materials.append(mat)
        o = bpy.data.objects.new(name, c)
        self.sc.collection.objects.link(o)
        return o

    def points(self, name, pos, radius, attrs, mat, instance=None, rot=None, letters=None):
        """Point cloud via geometry nodes: spheres (instance=None), instanced cubes, or letters."""
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in pos.astype(float)], [], [])
        def add(an, av, kind):
            av = np.asarray(av, np.float32)
            if kind == "col":
                if av.shape[1] == 3:
                    av = np.concatenate([av, np.ones((len(av), 1), np.float32)], 1)
                a = me.attributes.new(an, "FLOAT_COLOR", "POINT")
                a.data.foreach_set("color", av.ravel())
            elif kind == "vec":
                a = me.attributes.new(an, "FLOAT_VECTOR", "POINT")
                a.data.foreach_set("vector", av.ravel())
            elif kind == "int":
                a = me.attributes.new(an, "INT", "POINT")
                a.data.foreach_set("value", av.astype(np.int32).ravel())
            else:
                a = me.attributes.new(an, "FLOAT", "POINT")
                a.data.foreach_set("value", av.ravel())
        add("rad", radius, "f")
        for an, (av, kind) in attrs.items():
            add(an, av, kind)
        if rot is not None:
            add("rot", rot, "vec")
        o = bpy.data.objects.new(name, me)
        self.sc.collection.objects.link(o)
        ng = bpy.data.node_groups.new(name + "_gn", "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        N = ng.nodes
        gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
        rad = N.new("GeometryNodeInputNamedAttribute")
        rad.data_type, rad.inputs["Name"].default_value = "FLOAT", "rad"
        sm = N.new("GeometryNodeSetMaterial")
        sm.inputs["Material"].default_value = mat
        L = ng.links
        if instance is None:
            m2p = N.new("GeometryNodeMeshToPoints")
            L.new(gi.outputs[0], m2p.inputs["Mesh"])
            L.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
            L.new(m2p.outputs[0], sm.inputs["Geometry"])
            L.new(sm.outputs[0], go.inputs[0])
        else:
            iop = N.new("GeometryNodeInstanceOnPoints")
            L.new(gi.outputs[0], iop.inputs["Points"])
            if instance == "cube":
                cube = N.new("GeometryNodeMeshCube")
                cube.inputs["Size"].default_value = (2, 2, 2)
                L.new(cube.outputs[0], sm.inputs["Geometry"])
                L.new(sm.outputs[0], iop.inputs["Instance"])
            else:                                            # letters: pick by index
                ci = N.new("GeometryNodeCollectionInfo")
                ci.inputs["Collection"].default_value = letters
                ci.inputs["Separate Children"].default_value = True
                ci.inputs["Reset Children"].default_value = True
                iop.inputs["Pick Instance"].default_value = True
                li = N.new("GeometryNodeInputNamedAttribute")
                li.data_type, li.inputs["Name"].default_value = "INT", "letter"
                L.new(li.outputs["Attribute"], iop.inputs["Instance Index"])
                L.new(ci.outputs[0], sm.inputs["Geometry"])
                L.new(sm.outputs[0], iop.inputs["Instance"])
            L.new(rad.outputs["Attribute"], iop.inputs["Scale"])
            if rot is not None:
                ra = N.new("GeometryNodeInputNamedAttribute")
                ra.data_type, ra.inputs["Name"].default_value = "FLOAT_VECTOR", "rot"
                L.new(ra.outputs["Attribute"], iop.inputs["Rotation"])
            L.new(iop.outputs[0], go.inputs[0])
        mod = o.modifiers.new("gn", "NODES")
        mod.node_group = ng
        return o

    def capsule_body(self, name, A, B, R, sph, mat):
        import bmesh
        from mathutils import Matrix, Vector
        bm = bmesh.new()
        for a, b, r, s in zip(A, B, R, sph):
            a, b = Vector(a), Vector(b)
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r,
                                      matrix=Matrix.Translation(a))
            if s or (b - a).length < 1e-4:
                continue
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r,
                                      matrix=Matrix.Translation(b))
            axis = (b - a).normalized()
            rot = Vector((0, 0, 1)).rotation_difference(axis).to_matrix().to_4x4()
            bmesh.ops.create_cone(bm, cap_ends=False, segments=32, radius1=r, radius2=r,
                                  depth=(b - a).length, matrix=Matrix.Translation((a + b) / 2) @ rot)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        self.sc.collection.objects.link(o)
        return o


# =================================================================================
def build_strand(S: Scene, d, fiber=False, hq=False):
    P = S.bl(d["points"])
    n, K = P.shape[:2]
    t = np.linspace(0, 1, K, dtype=np.float32)[None]
    if fiber:
        r = np.full((n, K), float(d["fiber_w"]) / 2 / S.H, np.float32)
    else:
        r = (1.7 if hq else 1.4) * (float(d["root_w"]) + (float(d["tip_w"]) - float(d["root_w"])) * t) / 2 / S.H
        r = np.repeat(r, n, 0)
    if not fiber and hq:
        build_strand_hq(S, d, P, r)
        return
    if not fiber:
        base = srgb_to_linear(d["color"])
        col = np.clip(base[None] * d["var"], 0, 1)
        m, nt, bsdf = S.material("rubber")
        # Root -> tip gradient (darker roots, lighter tips) via Hair Info "Intercept".
        hi = nt.nodes.new("ShaderNodeHairInfo")
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 0.35, 1.25
        nt.links.new(hi.outputs["Intercept"], ramp.inputs["Value"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(S.attr(nt, "col").outputs["Color"], mix.inputs[6])
        nt.links.new(ramp.outputs[0], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.32
        bsdf.inputs["Coat Weight"].default_value = 0.6
        bsdf.inputs["Coat Roughness"].default_value = 0.12
        bsdf.inputs["Subsurface Weight"].default_value = 0.25
        bsdf.inputs["Subsurface Scale"].default_value = 0.01
        S.hair("strands", P, r, {"col": col}, m)
        S.three_point()
    else:
        S.sc.view_settings.view_transform = "Standard"
        rgb = srgb_to_linear(d["fiber_rgb"])
        m, nt, bsdf = S.material("fiber")
        bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.015, 1)
        bsdf.inputs["Roughness"].default_value = 0.15
        bsdf.inputs["Coat Weight"].default_value = 1.0
        hi = nt.nodes.new("ShaderNodeHairInfo")
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        pw.inputs[1].default_value = 2.5
        nt.links.new(hi.outputs["Intercept"], pw.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 1.6 + 1.0 * min(float(d["env"]), 1.5)
        nt.links.new(pw.outputs[0], mul.inputs[0])
        nt.links.new(S.attr(nt, "col").outputs["Color"], bsdf.inputs["Emission Color"])
        nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
        S.hair("fibers", P, r, {"col": rgb}, m)
        # Glowing tips: spheres at the fiber ends.
        mt, ntt, bt = S.material("tips")
        bt.inputs["Base Color"].default_value = (1, 1, 1, 1)
        ntt.links.new(S.attr(ntt, "col").outputs["Color"], bt.inputs["Emission Color"])
        bt.inputs["Emission Strength"].default_value = 4.0
        S.points("tips", P[:, -1], np.full(n, 0.7 * float(d["tip"]) / S.H), {"col": (rgb, "col")}, mt)
        S.three_point(key=1.0, fill=0.3, rim=2.5)


def build_strand_hq(S: Scene, d, P, r):
    """High-quality wet-rubber strands: saturated subsurface material with a glossy coat,
    soft area lighting matched to the room, and a shadow on the wall behind."""
    sc = S.sc
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.cycles_curves.shape = "THICK"
    sc.cycles_curves.subdivisions = 3 if FAST else 4
    sc.cycles.max_bounces = 4 if FAST else 8
    sc.cycles.transparent_max_bounces = 4 if FAST else 8
    base = srgb_to_linear(d["color"])
    rng = np.random.default_rng(11)
    var = np.clip(d["var"] * rng.uniform(0.9, 1.1, (len(P), 1)), 0.5, 1.6)
    col = np.clip(base[None] * var, 0, 1)
    m, nt, bsdf = S.material("rubber_hq")
    N, Lk = nt.nodes, nt.links
    hi = N.new("ShaderNodeHairInfo")
    # Darker, denser colour at the roots (fake occlusion), lighter at the tips.
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.28, 0.28, 0.28, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.55, (1, 1, 1, 1)
    e = ramp.color_ramp.elements.new(1.0)
    e.color = (1.15, 1.15, 1.15, 1)
    Lk.new(hi.outputs["Intercept"], ramp.inputs["Fac"])
    mix = N.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    Lk.new(S.attr(nt, "col").outputs["Color"], mix.inputs[6])
    Lk.new(ramp.outputs["Color"], mix.inputs[7])
    Lk.new(mix.outputs[2], bsdf.inputs["Base Color"])
    Lk.new(mix.outputs[2], bsdf.inputs["Subsurface Radius"])
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Subsurface Weight"].default_value = 0.0 if FAST else 0.45
    bsdf.inputs["Subsurface Scale"].default_value = float(np.mean(r)) * 1.5
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.06
    bsdf.inputs["Coat IOR"].default_value = 1.5
    bsdf.inputs["Specular IOR Level"].default_value = 0.6
    S.hair("strands", P, r, {"col": col}, m)
    world = sc.world.node_tree.nodes["Background"]
    world.inputs[0].default_value = (0.8, 0.8, 0.82, 1)
    world.inputs[1].default_value = 0.25
    S.studio_lights(LIGHT)
    depth = 3.2                       # behind the rim lights (they sit at y = 2.2)
    S.shadow_wall(depth)


def build_circuit(S: Scene, d):
    S.sc.view_settings.view_transform = "Standard"
    A, B = S.bl(d["bone_a"]), S.bl(d["bone_b"])
    R = d["bone_r"] / S.H
    body_rgb = srgb_to_linear(d["body_rgb"])
    mb, ntb, bb = S.material("body")
    bb.inputs["Base Color"].default_value = (*body_rgb, 1)
    bb.inputs["Metallic"].default_value = 0.4
    bb.inputs["Roughness"].default_value = 0.3
    bb.inputs["Coat Weight"].default_value = 0.8
    S.capsule_body("body", A, B, R, d["bone_sphere"], mb)
    trace_rgb = srgb_to_linear(d["color"])
    level = 0.45 + 0.55 * min(1.0, float(d["env"]) * 0.6)
    mt, ntt, bt = S.material("trace")
    bt.inputs["Base Color"].default_value = (*trace_rgb, 1)
    bt.inputs["Emission Color"].default_value = (*trace_rgb, 1)
    bt.inputs["Emission Strength"].default_value = 3.0 * level
    bt.inputs["Metallic"].default_value = 1.0
    tr = d["traces"]
    cnt = d["counts"]
    # Hair curves need a fixed point count; resample each trace to K points along its length.
    K = 16
    res = []
    for p, c in zip(tr, cnt):
        p = p[:c]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.linspace(0, s[-1], K)
        res.append(np.stack([np.interp(u, s, p[:, k]) for k in range(3)], 1))
    res = S.bl(np.array(res))
    r = np.full(res.shape[:2], float(d["trace_w"]) / 2 / S.H, np.float32)
    S.hair("traces", res, r, {}, mt)
    # Vias at both ends of each trace, and pulses.
    ends = np.concatenate([res[:, 0], res[:, -1]])
    S.points("vias", ends, np.full(len(ends), float(d["trace_w"]) * 1.6 / S.H), {}, mt)
    if len(d["pulses"]):
        mp, ntp, bp = S.material("pulse")
        bp.inputs["Emission Color"].default_value = (*np.clip(trace_rgb * 0.6 + 0.4, 0, 1), 1)
        bp.inputs["Emission Strength"].default_value = 25.0
        S.points("pulses", S.bl(d["pulses"][:, :3]),
                 np.full(len(d["pulses"]), float(d["trace_w"]) * 2.2 / S.H), {}, mp)
    # Chips: thin dark boxes on the skin.
    if len(d["chips"]):
        import bmesh
        from mathutils import Vector
        mc, ntc, bc = S.material("chip")
        bc.inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1)
        bc.inputs["Roughness"].default_value = 0.25
        bm = bmesh.new()
        for q in d["chips"]:
            vs = [bm.verts.new(Vector(S.bl(v[:3]))) for v in q]
            f = bm.faces.new(vs)
            nrm = S.bl(q[:, 3:].mean(0) * S.H)
            ext = bmesh.ops.extrude_face_region(bm, geom=[f])
            for v in [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]:
                v.co += Vector(nrm / (np.linalg.norm(nrm) + 1e-6) * float(d["trace_w"]) * 1.5 / S.H)
        me = bpy.data.meshes.new("chips")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mc)
        S.sc.collection.objects.link(bpy.data.objects.new("chips", me))
    S.three_point(key=2.0, fill=0.6, rim=4.0)


def build_swarm(S: Scene, d, vowels=False):
    P = S.bl(d["points"])
    n = len(P)
    rng = np.random.default_rng(3)
    rgb = srgb_to_linear(d["rgb"])
    m, nt, bsdf = S.material("particles")
    ca = S.attr(nt, "col", "INSTANCER")
    sa = S.attr(nt, "spd", "INSTANCER")
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Emission Color"])
    em = nt.nodes.new("ShaderNodeMath")
    em.operation = "MULTIPLY"
    em.inputs[1].default_value = 6.0
    nt.links.new(sa.outputs["Fac"], em.inputs[0])
    nt.links.new(em.outputs[0], bsdf.inputs["Emission Strength"])
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Coat Weight"].default_value = 0.7
    attrs = {"col": (rgb, "col"), "spd": (d["speed"], "f")}
    if not vowels:
        rot = rng.uniform(-0.6, 0.6, (n, 3)).astype(np.float32)
        S.points("cubes", P, d["size"] / 2 / S.H, attrs, m, instance="cube", rot=rot)
    else:
        bsdf.inputs["Metallic"].default_value = 0.6
        coll = bpy.data.collections.new("letters")
        for ch in "AEIOU":
            cu = bpy.data.curves.new(ch, "FONT")
            cu.body, cu.align_x, cu.align_y = ch, "CENTER", "CENTER"
            cu.extrude, cu.bevel_depth, cu.bevel_resolution = 0.12, 0.025, 2
            coll.objects.link(bpy.data.objects.new(ch, cu))
        rot = np.stack([np.full(n, math.pi / 2) + rng.uniform(-0.35, 0.35, n),
                        rng.uniform(-0.35, 0.35, n), rng.uniform(-0.5, 0.5, n)], 1).astype(np.float32)
        attrs["letter"] = (d["letter"], "int")
        S.points("letters", P, d["size"] * 1.5 / S.H, attrs, m, instance="letters", rot=rot, letters=coll)
    S.three_point()


def strand_geometry(S, d):
    P = S.bl(d["points"])
    K = P.shape[1]
    t = np.linspace(0, 1, K, dtype=np.float32)[None]
    r = 1.7 * (float(d["root_w"]) + (float(d["tip_w"]) - float(d["root_w"])) * t) / 2 / S.H
    return P, np.repeat(r, len(P), 0)


def main():
    args = [a for a in sys.argv[sys.argv.index("--") + 1:]] if "--" in sys.argv else sys.argv[1:]
    paths = [a for a in args if a.endswith(".npz")]
    path = paths[0]
    global FAST
    FAST = "--fast" in args
    samples = int(args[args.index("--samples") + 1]) if "--samples" in args else 48
    hq = "--hq" in args
    global LIGHT
    LIGHT = float(args[args.index("--light") + 1]) if "--light" in args else 0.3
    pct = int(args[args.index("--percent") + 1]) if "--percent" in args else 100
    d = np.load(path, allow_pickle=True)
    S = Scene(d)
    S.sc.cycles.samples = samples
    S.sc.cycles.use_adaptive_sampling = True
    S.sc.render.resolution_percentage = pct
    kind = str(d["creature"])
    {"strand": lambda: build_strand(S, d, hq=hq), "fiber": lambda: build_strand(S, d, fiber=True),
     "circuit": lambda: build_circuit(S, d), "swarm": lambda: build_swarm(S, d),
     "vowels": lambda: build_swarm(S, d, vowels=True)}[kind]()
    S.sc.render.use_persistent_data = True
    for k, pth in enumerate(paths):
        if k > 0:
            # Sequence mode (strand --hq): keep the scene, only move the strands.
            dk = np.load(pth, allow_pickle=True)
            P, r = strand_geometry(S, dk)
            c = bpy.data.objects["strands"].data
            c.position_data.foreach_set("vector", P.astype(np.float32).ravel())
            c.attributes["radius"].data.foreach_set("value", r.astype(np.float32).ravel())
            c.update_tag()
        S.sc.render.filepath = pth[:-4] + ("_hq" if hq else "") + "_render.png"
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED {S.sc.render.filepath} in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
