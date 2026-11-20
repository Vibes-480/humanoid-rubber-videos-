#!/usr/bin/env python3
"""Export creature geometry for a few frames so Blender can render them as real 3D CGI,
and composite Blender's renders back onto the cleaned background.

    python cgi_export.py export   input.mp4 --creature strand --frames 40 100 --out cgi_out
    .venv-blender/bin/python cgi_render.py cgi_out/strand_f0040.npz      (Blender, Cycles)
    python cgi_export.py composite cgi_out/strand_f0040.npz

The simulation (pose, strand physics, swarm springs, circuit pulses) is the one in hairify.py,
run from frame 0 up to each requested frame; Blender only does modelling, shading and lighting.
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import cv2
import numpy as np

import hairify as H


def build(args):
    info = H.probe_video(args.input)
    fps = float(info.fps)
    scale = args.scale
    out_w = int(round(info.width * scale / 2) * 2)
    out_h = int(round(info.height * scale / 2) * 2)
    last = max(args.frames) + 1
    cache = Path(H.CONFIG["cache_dir"])
    track = H.cached_pose(info, info.n_frames, H.CONFIG["pose"], cache)
    pose = H.smooth_pose(track, fps, H.CONFIG["smoothing"], info.height)
    rcfg = H.CONFIG["removal"]
    need = np.zeros(info.n_frames, bool)
    need[:last] = True
    need[H.plate_sample_frames(info.n_frames, rcfg)] = True
    scene = H.cached_scene(info, need, rcfg, cache)
    plate = H.cached_plate(info, scene, rcfg, cache)
    remover = H.PersonRemover(scene, plate, rcfg, scale)
    hits = H.detect_hits(info, last, H.CONFIG["audio"])
    body = H.Body(pose, out_w / info.width, fps, H.CONFIG["body"])
    cargs = types.SimpleNamespace(color=args.color, density=args.density,
                                  hair_length=args.hair_length, stiffness=args.stiffness)
    creature = H.CREATURES[args.creature](body, cargs, hits, fps, (out_w, out_h))
    grade = H.DEFAULT_GRADE[args.creature] if args.grade == "auto" else args.grade
    return info, fps, out_w, out_h, body, creature, remover, grade, hits


def export(args):
    info, fps, W, Hh, body, cr, remover, grade, hits = build(args)
    args.out.mkdir(parents=True, exist_ok=True)
    want = set(args.frames)
    visible = ~body.occluded                    # hidden body parts (legs behind the pad) are skipped
    names = np.array([b[0] for b in H.BONES])
    for i, frame in H.iter_frames(info.path, max(want) + 1):
        cr.step(i)
        if i not in want:
            continue
        small = cv2.resize(frame, (W, Hh), interpolation=cv2.INTER_AREA)
        bg = H.grade_frame(remover(small, i), grade)
        stem = args.out / f"{args.creature}_f{i:04d}"
        cv2.imwrite(str(stem) + "_bg.png", bg)
        d = dict(creature=args.creature, width=W, height=Hh, frame=i, Wbody=float(body.W[i]),
                 color=cr.color[::-1] / 255.0, grade=grade)          # colours as RGB 0..1
        A, B, R = body.A[i], body.B[i], body.R[i]
        d.update(bone_a=A[visible], bone_b=B[visible], bone_r=R[visible],
                 bone_sphere=H.BONE_SPHERE[visible], bone_name=names[visible])
        if args.creature in ("strand", "fiber"):
            keep = visible[cr.bones]
            rc = H.CONFIG["render"]
            var = 1 + rc["color_variation"] * cr.var[keep]
            d.update(points=cr.P[keep], var=var[:, ::-1],
                     root_w=rc["root_width"] * body.W[i], tip_w=rc["tip_width"] * body.W[i])
            if args.creature == "fiber":
                fc = H.CONFIG["fiber"]
                hue = cr.hue[keep] + fc["hue_drift"] * i / fps
                d.update(fiber_rgb=H.hsv_deg_to_bgr(hue, 0.85, 1.0)[:, ::-1] / 255.0,
                         width=fc["width"] * body.W[i], tip=fc["tip_size"] * body.W[i],
                         env=float(hits.env[i]))
        elif args.creature == "circuit":
            n, mp_ = cr.tr_pts.shape[:2]
            keep = visible[cr.tr_bones]
            fb = np.repeat(cr.tr_bones, mp_)
            pos, nrm = body.surface(i, fb, cr.tr_pts[..., 0].ravel(), cr.tr_pts[..., 1].ravel())
            pos = (pos + nrm * 0.012 * body.W[i]).reshape(n, mp_, 3)   # sit just on the skin
            d.update(traces=pos[keep], counts=cr.tr_cnt[keep],
                     trace_w=H.CONFIG["circuit"]["trace_width"] * body.W[i],
                     body_rgb=H.hex_to_bgr(H.CONFIG["circuit"]["body_color"])[::-1] / 255.0,
                     env=float(hits.env[i]))
            pts = []
            for tr, ps, st in zip(cr.p_tr, cr.p_pos, cr.p_str):
                if not keep[tr]:
                    continue
                m = cr.tr_cnt[tr]
                x = ps * (m - 1)
                j = min(int(x), m - 2)
                pts.append(np.append(pos[tr, j] + (pos[tr, j + 1] - pos[tr, j]) * (x - j), st))
            d.update(pulses=np.array(pts, np.float32).reshape(-1, 4))
            chips = []
            for bo, t0, th0, sz in cr.chips:
                if not visible[bo]:
                    continue
                dt_ = sz * body.W_med / body.len_med[bo]
                dth = sz / H.BONE_RF[bo]
                qp, qn = body.surface(i, np.full(4, bo),
                                      np.array([t0 - dt_, t0 + dt_, t0 + dt_, t0 - dt_], np.float32),
                                      np.array([th0 - dth, th0 - dth, th0 + dth, th0 + dth], np.float32))
                chips.append(np.concatenate([qp + qn * 0.01 * body.W[i], qn], 1))
            d.update(chips=np.array(chips, np.float32).reshape(-1, 4, 6))
        elif args.creature in ("swarm", "vowels"):
            keep = visible[cr.bones]
            spd = np.linalg.norm(cr.V[keep], axis=1) / (body.W[i] * 3.0)
            d.update(points=cr.X[keep], size=cr.size[keep] * body.W[i], rgb=cr.col[keep][:, ::-1] / 255.0,
                     speed=np.clip(spd, 0, 1))
            if args.creature == "vowels":
                d.update(letter=cr.letter[keep])
        np.savez_compressed(str(stem) + ".npz", **d)
        print(f"exported {stem}.npz")


def composite(args):
    for npz in args.npz:
        d = np.load(npz, allow_pickle=True)
        stem = str(npz)[:-4]
        bg = cv2.imread(stem + "_bg.png").astype(np.float32)
        fg = cv2.imread(stem + "_render.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        a = fg[..., 3:4] / 255.0
        out = bg * (1 - a) + fg[..., :3] * a            # Blender writes straight (unpremultiplied) alpha
        if str(d["creature"]) in ("fiber", "circuit", "vowels", "swarm"):
            # Bloom from the brightest parts of the render (the emissive bits).
            lum = fg[..., :3].max(axis=2, keepdims=True) * a
            hot = np.clip((lum - 170) / 85, 0, 1) * fg[..., :3] * a
            h, w = hot.shape[:2]
            small = cv2.resize(hot, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            bloom = cv2.GaussianBlur(small, (0, 0), h / 200) + cv2.GaussianBlur(small, (0, 0), h / 60)
            gain = {"fiber": 0.9, "circuit": 1.0, "vowels": 0.5, "swarm": 0.4}[str(d["creature"])]
            out = out + gain * cv2.resize(bloom, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(stem + "_final.png", np.clip(out, 0, 255).astype(np.uint8))
        print(f"composited {stem}_final.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("input", type=Path)
    e.add_argument("--creature", choices=list(H.CREATURES), default="strand")
    e.add_argument("--frames", type=int, nargs="+", default=[40, 100])
    e.add_argument("--out", type=Path, default=Path("cgi_out"))
    e.add_argument("--scale", type=float, default=0.5)
    e.add_argument("--color", default=None)
    e.add_argument("--density", type=float, default=1.0)
    e.add_argument("--hair-length", type=float, default=1.0)
    e.add_argument("--stiffness", type=float, default=None)
    e.add_argument("--grade", choices=["auto", "none", "cyber"], default="auto")
    c = sub.add_parser("composite")
    c.add_argument("npz", type=Path, nargs="+")
    args = p.parse_args()
    export(args) if args.cmd == "export" else composite(args)


if __name__ == "__main__":
    main()
