#!/usr/bin/env python3
"""hairify.py - replace a person in a video with a jiggly rubber-strand creature.

Pipeline (see README.md):
    1. pose tracking      (MediaPipe Pose Landmarker, One Euro smoothing, gap handling)
    2. person removal     (segmentation mask -> clean background plate / inpaint)
    3. creature body      (capsules on the skeleton, strand roots on their surface)
    4. strand physics     (vectorised Verlet chains)
    5. rendering          (tapered glossy strands, core body, contact shadow)
    6. encode             (ffmpeg, H.264, original audio)

Use --stage to stop early and render a debug video for a single stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# =============================================================================
# CONFIG - every tunable value lives here. CLI flags override a few of them.
# =============================================================================
CONFIG = {
    "pose": {
        "model": "full",                 # lite | full | heavy (heavy = best, slowest)
        "model_dir": str(HERE / "models"),
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "smoothing": {
        # One Euro filter, coordinates normalised by frame height so the values
        # are resolution independent. Lower min_cutoff = smoother when still,
        # higher beta = less lag when moving fast.
        "min_cutoff": 1.2,               # Hz
        "beta": 3.0,
        "d_cutoff": 1.0,                 # Hz
        "max_interp_gap_s": 0.5,         # gaps shorter than this are interpolated
        "fade_s": 0.35,                  # fade-out/in time when tracking is lost/regained
    },
    "removal": {
        "analysis_height": 270,          # masks / motion are computed and cached at this height
        "mask_threshold": 0.35,          # person probability above which a pixel is removed
        "temporal_dilate": 1,            # also remove where the person was +-N frames ago (motion blur)
        "dilate_px": 0.02,               # mask grow, fraction of frame height
        "feather_px": 0.015,             # mask edge softness, fraction of frame height
        "plate_samples": 60,             # frames sampled for the median background plate
        "plate_min_coverage": 0.25,      # fraction of samples a pixel must be seen in, else inpaint
        "inpainter": "rows",             # key in INPAINTERS (rows | pyramid | telea | your own)
        "inpaint_radius": 5,
        "color_match": True,             # match plate exposure to each frame (auto-exposure drift)
        "static_threshold_px": 3.0,      # camera shift below this counts as static
        "warp_smooth_frames": 5,         # temporal smoothing of camera motion estimates
        "min_inliers": 25,               # background feature matches needed to align a frame
    },
    # Lengths, sizes and widths below are in units of W, the body scale (~shoulder width).
    "body": {
        "radius_scale": 1.0,             # capsule thickness multiplier
        "z_scale": 0.6,                  # trust in MediaPipe depth (noisy)
        "scale_smooth_s": 0.5,           # temporal smoothing of the body scale
        "occlusion_vis": 0.75,           # bones with lower median visibility are clipped to the person mask
    },
    "creature": {                        # --creature strand (rubber hair)
        "strand_count": 1800,            # total strands at --density 1.0
        "segments": 12,                  # points per strand
        "hair_length": 0.65,             # strand length (W) at --hair-length 1.0
        "length_jitter": 0.3,
        "fan_out": 0.6,                  # 0 = strands follow the 3D surface normal, 1 = flattened
                                         # sideways so front strands fan out instead of facing the lens
        "default_stiffness": 0.15,       # floppy: hangs and swings like hair / cooked spaghetti
    },
    "physics": {                         # strand simulation (Verlet + dynamic follow-the-leader)
        "sim_rate": 120.0,               # physics steps/s, independent of the video fps
        "gravity": 30.0,                 # W / s^2 (sets how fast hanging strands swing)
        "damping_floppy": 0.006,         # velocity loss per step at stiffness 0 (low = keeps swinging)
        "damping_stiff": 0.03,           # ... at stiffness 1
        "k_shape_max": 0.22,             # pull to the straight rest pose at stiffness 1 (x stiffness^3)
        "tip_softness": 0.6,             # tips are this much softer than roots
        "follicle_points": 2,            # points near the root held in the skin direction
        "k_follicle_min": 0.08,
        "k_follicle_max": 0.4,
        "k_bend_min": 0.12,              # smooth curves at stiffness 0
        "k_bend_max": 0.3,
        "ftl_damping": 0.9,              # dynamic follow-the-leader velocity correction (0..1)
        "z_flatten": 0.04,               # per step, pull strand depth toward the root's (fewer
                                         # foreshortened zigzags; 0 = full 3D)
    },
    "render": {                          # strand shading
        "color": "#ff3fa4",
        "root_darken": 0.55,             # root colour = base * (1 - root_darken)
        "tip_lighten": 0.30,
        "color_variation": 0.10,
        "root_width": 0.022,             # W
        "tip_width": 0.007,
        "light_dir": (-0.4, -0.7, 0.6),  # x right, y down, z away from camera
        "specular": 0.6,
        "shininess": 30.0,
        "core_darkness": 0.9,            # core body colour = root colour * this
        "shadow_opacity": 0.45,
    },
    "fiber": {                           # --creature fiber (fiber-optic neon)
        "strand_count": 2000,
        "segments": 10,
        "hair_length": 0.6,
        "length_jitter": 0.35,
        "default_stiffness": 0.45,
        "width": 0.006,                  # W
        "tip_size": 0.012,               # W
        "hue_spread": 45.0,              # degrees around --color
        "hue_drift": 15.0,               # degrees / s
        "base_glow": 0.45,
        "pulse_gain": 0.6,               # extra brightness of hit pulses
        "fan_out": 0.6,
        "pulse_speed": 2.2,              # fiber lengths / s
        "glow": 0.7,
    },
    "circuit": {                         # --creature circuit (circuit-board skin)
        "traces": 650,
        "body_color": "#141a2e",
        "trace_width": 0.008,            # W
        "step": (0.05, 0.16),            # trace segment length range (W)
        "min_segments": 3,
        "max_points": 9,
        "turn_prob": 0.45,               # chance of a 45 degree turn per segment
        "chips": 8,
        "base_brightness": 0.45,
        "ambient_pulses": 30.0,          # pulses / s
        "hit_pulses": 45,                # pulses per drum hit
        "pulse_speed": 1.6,              # trace lengths / s
        "glow": 1.0,
    },
    "swarm": {                           # --creature swarm (voxel cubes) and vowels
        "count": 3500,
        "cube_size": 0.05,               # W, at --hair-length 1.0
        "hue_spread": 25.0,
        "freq_min": 1.5,                 # spring frequency (Hz) at stiffness 0
        "freq_max": 7.0,                 # ... at stiffness 1
        "damping_ratio": 0.3,            # <1 = overshoot and jiggle
        "default_stiffness": 0.5,
        "scatter_speed": 4.0,            # W / s per unit hit strength
        "scatter_radius": 0.8,           # W around the striking hand
        "glow": 0.8,
    },
    "vowels": {                          # --creature vowels (letter swarm)
        "count": 2200,
        "size_scale": 1.6,               # letters relative to cube_size
        "swap_rate": 0.8,                # letter changes / s
    },
    "audio": {                           # drum-hit detection
        "sample_rate": 22050,
        "threshold": 3.0,                # in MADs above the local median of spectral flux
        "median_window_s": 0.4,
        "min_gap_s": 0.06,
        "decay_s": 0.15,
    },
    "grade": {                           # --grade cyber
        "exposure": 0.42,
        "gamma": 1.5,
        "tint": 0.6,
        "shadow_tint": "#0a5a78",
        "highlight_tint": "#7a5ab0",
        "vignette": 0.65,
    },
    "encode": {
        "crf": 18,
        "preset": "veryfast",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    },
    "preview": {"seconds": 5.0, "scale": 0.5},
    "cache_dir": str(HERE / ".hairify_cache"),
}

# MediaPipe 33-point pose topology (subset used for drawing).
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),             # torso
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm/hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm/hand
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),   # left leg/foot
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),   # right leg/foot
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # face
]
LEFT_IDS = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_IDS = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{m}/float16/latest/pose_landmarker_{m}.task")


# =============================================================================
# Environment checks & video I/O
# =============================================================================
def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"error: '{tool}' not found on PATH. Install ffmpeg "
                     "(e.g. 'sudo apt install ffmpeg' or 'brew install ffmpeg').")


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    n_frames: int
    duration: float
    has_audio: bool
    audio_codec: str | None = None


def probe_video(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"error: ffprobe could not read {path}:\n{out.stderr}")
    meta = json.loads(out.stdout)
    vstreams = [s for s in meta["streams"] if s["codec_type"] == "video"]
    if not vstreams:
        sys.exit(f"error: no video stream in {path}")
    vs = vstreams[0]
    fps = Fraction(vs.get("avg_frame_rate") or vs["r_frame_rate"])
    if fps <= 0:
        fps = Fraction(vs["r_frame_rate"])
    duration = float(vs.get("duration") or meta["format"].get("duration") or 0)
    # OpenCV applies rotation metadata, so take the size from a decoded frame.
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        sys.exit(f"error: OpenCV could not decode {path}")
    h, w = frame.shape[:2]
    if n <= 0:
        n = int(round(duration * float(fps)))
    astreams = [s for s in meta["streams"] if s["codec_type"] == "audio"]
    return VideoInfo(path, w, h, fps, n, duration, bool(astreams),
                     astreams[0].get("codec_name") if astreams else None)


def iter_frames(path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


class FFmpegWriter:
    """Pipes raw BGR frames into ffmpeg, muxing the original audio in one go."""

    def __init__(self, out_path: Path, width: int, height: int, fps: Fraction,
                 audio_src: Path | None, duration: float | None, enc: dict,
                 audio_codec: str | None = None):
        self.out_path = out_path
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        if audio_src is not None:
            if duration is not None:
                cmd += ["-t", f"{duration:.3f}"]
            cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?"]
            # MP4-compatible audio is copied bit-exact; trimmed previews re-encode so the cut is exact.
            if duration is None and audio_codec in ("aac", "mp3", "alac"):
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", enc["audio_codec"], "-b:a", enc["audio_bitrate"]]
        cmd += ["-c:v", "libx264", "-crf", str(enc["crf"]), "-preset", enc["preset"],
                "-pix_fmt", enc["pix_fmt"], "-movflags", "+faststart", str(out_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.size = (width, height)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            sys.exit("error: ffmpeg encoding failed")


# =============================================================================
# Stage 1: pose tracking
# =============================================================================
def ensure_model(name: str, model_dir: Path) -> Path:
    path = model_dir / f"pose_landmarker_{name}.task"
    if not path.exists():
        import urllib.request
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading pose model '{name}' ...")
        urllib.request.urlretrieve(MODEL_URL.format(m=name), path)
    return path


@dataclass
class PoseTrack:
    img: np.ndarray        # [N,33,3] x_px, y_px, z_px (z: depth, same scale as x; smaller = closer)
    vis: np.ndarray        # [N,33] visibility 0..1
    world: np.ndarray      # [N,33,3] metric, hip-centred (MediaPipe world landmarks)
    detected: np.ndarray   # [N] bool


def track_pose(info: VideoInfo, n_frames: int, cfg: dict) -> PoseTrack:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model = ensure_model(cfg["model"], Path(cfg["model_dir"]))
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=cfg["min_detection_confidence"],
        min_pose_presence_confidence=cfg["min_presence_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    )
    img = np.full((n_frames, 33, 3), np.nan, np.float32)
    vis = np.zeros((n_frames, 33), np.float32)
    world = np.full((n_frames, 33, 3), np.nan, np.float32)
    detected = np.zeros(n_frames, bool)
    w, h = info.width, info.height
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i, frame in tqdm(iter_frames(info.path, n_frames), total=n_frames,
                             desc="pose tracking", unit="f"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(i * 1000 / info.fps))
            if not res.pose_landmarks:
                continue
            detected[i] = True
            pts = res.pose_landmarks[0]
            img[i] = [(p.x * w, p.y * h, p.z * w) for p in pts]
            vis[i] = [p.visibility or 0.0 for p in pts]
            if res.pose_world_landmarks:
                world[i] = [(p.x, p.y, p.z) for p in res.pose_world_landmarks[0]]
    return PoseTrack(img, vis, world, detected)


def cached_pose(info: VideoInfo, n_frames: int, cfg: dict, cache_dir: Path) -> PoseTrack:
    """Pose tracking is the slow part, so cache it per (file, model, frame count)."""
    st = info.path.stat()
    key = hashlib.sha1(json.dumps([str(info.path.resolve()), st.st_size, st.st_mtime_ns,
                                   n_frames, cfg], sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"pose_{info.path.stem}_{key}.npz"
    if path.exists():
        d = np.load(path)
        print(f"using cached pose track: {path.name}")
        return PoseTrack(d["img"], d["vis"], d["world"], d["detected"])
    track = track_pose(info, n_frames, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=track.img, vis=track.vis, world=track.world,
                        detected=track.detected)
    return track


class OneEuroFilter:
    """Vectorised One Euro filter (Casiez et al. 2012) over an array of signals."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


@dataclass
class SmoothPose:
    img: np.ndarray        # [N,33,3] smoothed, gaps filled
    vis: np.ndarray        # [N,33]
    world: np.ndarray      # [N,33,3]
    alpha: np.ndarray      # [N] creature opacity (fades when tracking is lost)
    valid: np.ndarray      # [N] bool, a pose is available (detected or interpolated/held)
    raw: PoseTrack


def _fill_gaps(det: np.ndarray, max_gap: int):
    """Return (interpolate_mask, long_gap_mask) for missing frames."""
    n = len(det)
    interp = np.zeros(n, bool)
    long_gap = np.zeros(n, bool)
    i = 0
    while i < n:
        if det[i]:
            i += 1
            continue
        j = i
        while j < n and not det[j]:
            j += 1
        bounded = i > 0 and j < n
        if bounded and (j - i) <= max_gap:
            interp[i:j] = True
        else:
            long_gap[i:j] = True
        i = j
    return interp, long_gap


def smooth_pose(track: PoseTrack, fps: float, cfg: dict, height: int) -> SmoothPose:
    n = len(track.detected)
    det = track.detected
    max_gap = max(1, int(round(cfg["max_interp_gap_s"] * fps)))
    interp, long_gap = _fill_gaps(det, max_gap)

    img = track.img.copy()
    vis = track.vis.copy()
    world = track.world.copy()
    idx = np.arange(n)
    if det.any():
        # Linear interpolation across short gaps (per landmark coordinate).
        good = np.flatnonzero(det)
        for arr in (img, world):
            flat = arr.reshape(n, -1)
            for c in range(flat.shape[1]):
                flat[~det, c] = np.interp(idx[~det], good, flat[good, c])
        for c in range(33):
            vis[~det, c] = np.interp(idx[~det], good, track.vis[good, c])
        # np.interp clamps at the ends, so long gaps hold the nearest detected pose.

    # One Euro smoothing in height-normalised units. Reset across long gaps so
    # the filter does not slide the body from where it vanished to where it reappears.
    dt = 1.0 / fps
    f_img = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    f_world = OneEuroFilter(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
    out_img = np.empty_like(img)
    out_world = np.empty_like(world)
    for i in range(n):
        if i > 0 and long_gap[i - 1] and not long_gap[i]:
            f_img.reset(); f_world.reset()
        if np.isnan(img[i]).any():
            out_img[i], out_world[i] = img[i], world[i]
            continue
        out_img[i] = f_img(img[i] / height, dt) * height
        out_world[i] = f_world(np.nan_to_num(world[i]), dt)

    # Creature opacity: 1 when tracked/interpolated, fades during long gaps.
    fade_n = max(1, int(round(cfg["fade_s"] * fps)))
    present = (det | interp).astype(np.float32)
    alpha = np.zeros(n, np.float32)
    a = 0.0
    for i in range(n):            # fade out after loss
        a = 1.0 if present[i] else max(0.0, a - 1.0 / fade_n)
        alpha[i] = a
    a = 0.0
    for i in range(n - 1, -1, -1):  # fade in before re-acquisition
        a = 1.0 if present[i] else max(0.0, a - 1.0 / fade_n)
        alpha[i] = max(alpha[i], a) if not present[i] else 1.0
    if not det.any():
        alpha[:] = 0
    valid = ~np.isnan(out_img).any(axis=(1, 2))
    return SmoothPose(out_img, vis, out_world, alpha, valid, track)


# =============================================================================
# Stage 2: person removal
# =============================================================================
SEG_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                 "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")


def _cache_path(cache_dir: Path, kind: str, info: VideoInfo, n_frames: int, cfg) -> Path:
    st = info.path.stat()
    key = hashlib.sha1(json.dumps([str(info.path.resolve()), st.st_size, st.st_mtime_ns,
                                   n_frames, cfg], sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir / f"{kind}_{info.path.stem}_{key}.npz"


# --- Inpainters ---------------------------------------------------------------
# An inpainter is any function (bgr_uint8_image, mask_uint8) -> bgr_uint8_image that
# fills the pixels where mask > 0. Register a better one (e.g. LaMa) here and select
# it with CONFIG["removal"]["inpainter"].
def inpaint_telea(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    return cv2.inpaint(img, (mask > 0).astype(np.uint8), radius, cv2.INPAINT_TELEA)


def inpaint_pyramid(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Telea at low resolution, upsampled into the hole. Large holes come out as smooth,
    plausible gradients instead of the streaks full-res Telea produces, and it is fast."""
    h, w = mask.shape[:2]
    hole = (mask > 0).astype(np.uint8)
    if not hole.any():
        return img.copy()
    scale = min(1.0, 400.0 / max(h, w))
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_hole = (cv2.resize(hole, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    small_hole = cv2.dilate(small_hole, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, small_hole, radius, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 1.0)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    # Feather the seam so the low-res fill blends into the full-res surroundings.
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), max(1.0, 2.0 / scale))
    soft = np.maximum(soft, hole.astype(np.float32))[..., None]
    return (img * (1 - soft) + up * soft).astype(np.uint8)


def inpaint_rows(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill each row of the hole by blending between the background colours just left and
    right of it, then smooth vertically. Person-shaped holes are tall and narrow, so the
    horizontal direction is the short one; this also stops foreground objects below the
    person (a table, a drum pad) from bleeding upwards the way radial inpainters do."""
    hole = mask > 0
    if not hole.any():
        return img.copy()
    h, w = hole.shape
    k = max(3, 2 * radius)                       # boundary pixels averaged on each side
    out = img.astype(np.float32)
    filled = np.zeros(h, bool)
    for y in np.flatnonzero(hole.any(axis=1)):
        r = np.concatenate([[0], hole[y].astype(np.int8), [0]])
        starts, ends = np.flatnonzero(np.diff(r) == 1), np.flatnonzero(np.diff(r) == -1)
        for a, b in zip(starts, ends):
            left = np.median(out[y, max(0, a - k):a], axis=0) if a > 0 else None
            right = np.median(out[y, b:min(w, b + k)], axis=0) if b < w else None
            if left is None and right is None:
                continue
            left = right if left is None else left
            right = left if right is None else right
            t = np.linspace(0, 1, b - a + 2)[1:-1, None]
            out[y, a:b] = left * (1 - t) + right * t
        filled[y] = True
    # Rows that are hole from edge to edge: copy the nearest filled row.
    empty = hole.all(axis=1) | ~filled & hole.any(axis=1)
    if empty.any() and filled.any():
        src = np.flatnonzero(~empty)
        near = src[np.abs(src[None, :] - np.flatnonzero(empty)[:, None]).argmin(axis=1)]
        out[empty] = out[near]
    # Vertical smoothing removes row-to-row streaks; composite only inside the hole.
    sm = cv2.GaussianBlur(out, (0, 0), sigmaX=max(1.0, 0.004 * w), sigmaY=max(2.0, 0.03 * h))
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), max(1.0, 0.004 * h))
    soft = np.maximum(soft, cv2.erode(hole.astype(np.uint8), np.ones((5, 5), np.uint8)))[..., None]
    res = np.where(hole[..., None], sm, out)
    return np.clip(img * (1 - soft) + res * soft, 0, 255).astype(np.uint8)


INPAINTERS = {"rows": inpaint_rows, "telea": inpaint_telea, "pyramid": inpaint_pyramid}


# --- Analysis pass: masks + camera motion --------------------------------------
@dataclass
class SceneAnalysis:
    masks: np.ndarray      # [N,h,w] uint8 person probability (0..255) at analysis resolution
    have: np.ndarray       # [N] bool, frame has been analysed (analysis is incremental)
    warps: np.ndarray      # [N,3,3] full-res transform frame -> reference frame (frame 0)
    aligned: np.ndarray    # [N] bool, warp is trustworthy
    static: bool           # camera considered static (all warps = identity)


def ensure_seg_model(model_dir: Path) -> Path:
    path = model_dir / "selfie_multiclass_256x256.tflite"
    if not path.exists():
        import urllib.request
        model_dir.mkdir(parents=True, exist_ok=True)
        print("downloading segmentation model ...")
        urllib.request.urlretrieve(SEG_MODEL_URL, path)
    return path


def plate_sample_frames(n_total: int, cfg: dict) -> np.ndarray:
    k = min(cfg["plate_samples"], n_total)
    return np.unique(np.linspace(0, n_total - 1, k).round().astype(int))


def analyze_scene(info: VideoInfo, sc: SceneAnalysis, need: np.ndarray, cfg: dict) -> int:
    """Segment the person and estimate camera motion for frames in `need` that are not
    analysed yet (in place). Frame 0 is always analysed: it is the motion reference."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    todo = need & ~sc.have
    todo[0] = not sc.have[0]
    if not todo.any():
        return 0
    model = ensure_seg_model(Path(CONFIG["pose"]["model_dir"]))
    opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                        output_confidence_masks=True)
    ah, aw = sc.masks.shape[1:]
    s = info.width / aw                                     # analysis -> full-res scale
    S, S_inv = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    excl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ref = None
    last = int(np.flatnonzero(todo).max())
    cap = cv2.VideoCapture(str(info.path))
    bar = tqdm(total=int(todo.sum()), desc="segmenting", unit="f")
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        for i in range(last + 1):
            if not (todo[i] or (i == 0 and ref is None)):
                if not cap.grab():                          # skip without decoding
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
            if todo[i]:
                res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                bg = np.squeeze(res.confidence_masks[0].numpy_view())  # category 0 = background
                sc.masks[i] = np.clip((1.0 - bg) * 255, 0, 255).astype(np.uint8)
            # Camera motion: match background features against frame 0.
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            fmask = np.where(cv2.dilate(sc.masks[i], excl) > 64, 0, 255).astype(np.uint8)
            kp, des = orb.detectAndCompute(gray, fmask)
            if i == 0:
                ref = (kp, des)
                sc.warps[0], sc.aligned[0] = np.eye(3), True
            else:
                ok = False
                if des is not None and ref[1] is not None and len(kp) >= 10:
                    m = bf.match(des, ref[1])
                    if len(m) >= cfg["min_inliers"]:
                        src = np.float32([kp[x.queryIdx].pt for x in m])
                        dst = np.float32([ref[0][x.trainIdx].pt for x in m])
                        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                             ransacReprojThreshold=2.0)
                        if A is not None and inl is not None and inl.sum() >= cfg["min_inliers"]:
                            sc.warps[i] = S @ np.vstack([A, [0, 0, 1]]) @ S_inv
                            ok = True
                sc.aligned[i] = ok
            if todo[i]:
                sc.have[i] = True
                bar.update(1)
    bar.close()
    cap.release()
    return int(todo.sum())


def finalize_motion(sc: SceneAnalysis, cfg: dict, width: int, height: int) -> SceneAnalysis:
    """Smooth the raw per-frame camera motion and decide whether the camera is static."""
    warps, aligned = sc.warps.copy(), sc.aligned & sc.have
    # Smooth the estimates over time (moving average of the affine parameters over
    # consecutive analysed frames) so estimation noise does not make the fill jitter.
    ks = cfg["warp_smooth_frames"]
    if ks > 1 and aligned.sum() > ks:
        idx = np.flatnonzero(aligned)
        p = warps[idx][:, :2].reshape(len(idx), -1)
        pad = np.pad(p, ((ks // 2, ks // 2), (0, 0)), mode="edge")
        p = np.stack([np.convolve(pad[:, c], np.ones(ks) / ks, "valid") for c in range(6)], 1)
        warps[idx, :2] = p.reshape(-1, 2, 3)
    # Static camera: frame-corner displacement stays under the threshold.
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], float).T
    disp = np.abs(np.einsum("nij,jk->nik", warps, corners)[:, :2] - corners[:2]).max(axis=(1, 2))
    p95 = float(np.percentile(disp[aligned], 95)) if aligned.any() else 0.0
    static = p95 < cfg["static_threshold_px"]
    n_bad = int((sc.have & ~sc.aligned).sum())
    print(f"camera: {'static' if static else 'moving'} (95th pct shift {p95:.2f}px, "
          f"{n_bad} frames could not be aligned)")
    if static:
        warps[:] = np.eye(3)
        aligned = sc.have.copy()
    return SceneAnalysis(sc.masks, sc.have, warps, aligned, static)


def cached_scene(info: VideoInfo, need: np.ndarray, cfg: dict, cache_dir: Path,
                 use_cache: bool = True) -> SceneAnalysis:
    """Incremental, cached scene analysis: only frames in `need` that were never analysed
    are processed, so a --preview run costs ~preview length + plate samples."""
    n = len(need)
    keycfg = {k: cfg[k] for k in ("analysis_height", "min_inliers")}
    path = _cache_path(cache_dir, "scene", info, n, keycfg)
    ah = cfg["analysis_height"]
    aw = int(round(info.width * ah / info.height))
    if use_cache and path.exists():
        d = np.load(path)
        sc = SceneAnalysis(d["masks"], d["have"], d["warps"], d["aligned"], False)
    else:
        sc = SceneAnalysis(np.zeros((n, ah, aw), np.uint8), np.zeros(n, bool),
                           np.tile(np.eye(3), (n, 1, 1)), np.zeros(n, bool), False)
    done = analyze_scene(info, sc, need, cfg)
    if done:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, masks=sc.masks, have=sc.have, warps=sc.warps, aligned=sc.aligned)
    else:
        print(f"using cached scene analysis: {path.name}")
    return finalize_motion(sc, cfg, info.width, info.height)


# --- Masks ----------------------------------------------------------------------
def person_mask(scene: SceneAnalysis, i: int, size: tuple[int, int], cfg: dict) -> np.ndarray:
    """Soft person mask (float32 0..1) at `size`: temporal max over analysed neighbours,
    threshold, dilate, feather. Morphology runs at <=540p and is upsampled."""
    w, h = size
    t = cfg["temporal_dilate"]
    lo, hi = max(0, i - t), min(len(scene.masks), i + t + 1)
    nb = [j for j in range(lo, hi) if scene.have[j]] or [i]
    m = scene.masks[nb].max(axis=0)                 # covers motion blur and flicker
    ws = min(1.0, 540 / h)
    mw, mh = int(w * ws), int(h * ws)
    m = cv2.resize(m, (mw, mh), interpolation=cv2.INTER_LINEAR)
    hard = (m > cfg["mask_threshold"] * 255).astype(np.uint8)
    d = max(1, int(cfg["dilate_px"] * mh))
    hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    f = max(1.0, cfg["feather_px"] * mh)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), f / 2)
    soft = np.maximum(soft, cv2.erode(hard, np.ones((3, 3), np.uint8)).astype(np.float32))
    if (mw, mh) != (w, h):
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    return soft


# --- Background plate ------------------------------------------------------------
@dataclass
class Plate:
    image: np.ndarray      # [H,W,3] uint8, reference-frame coordinates, holes filled
    raw: np.ndarray        # [H,W,3] uint8, median before hole filling
    seen: np.ndarray       # [H,W] uint8 255 where the real background was observed
    coverage: np.ndarray   # [H,W] int number of unmasked samples


def build_plate(info: VideoInfo, n_frames: int, scene: SceneAnalysis, cfg: dict) -> Plate:
    pick = [i for i in plate_sample_frames(n_frames, cfg) if scene.have[i] and scene.aligned[i]]
    W, H = info.width, info.height
    stack = np.zeros((len(pick), H, W, 3), np.uint8)
    valid = np.zeros((len(pick), H, W), bool)
    j = 0
    cap = cv2.VideoCapture(str(info.path))
    pos = 0
    for i in tqdm(pick, desc="building plate", unit="f"):
        while pos < i:                                  # grab() skips without decoding
            cap.grab(); pos += 1
        ok, frame = cap.read(); pos += 1
        if not ok:
            break
        m = person_mask(scene, i, (W, H), cfg) < 0.01
        if not scene.static:
            A = scene.warps[i][:2]
            frame = cv2.warpAffine(frame, A, (W, H), flags=cv2.INTER_LINEAR)
            m = cv2.warpAffine(m.astype(np.uint8), A, (W, H), flags=cv2.INTER_NEAREST,
                               borderValue=0).astype(bool)
        stack[j], valid[j] = frame, m
        j += 1
    cap.release()
    stack, valid = stack[:j], valid[:j]
    coverage = valid.sum(axis=0)
    # Masked median: invalid samples get a sentinel above any pixel value, then after a
    # sort along the sample axis the median of the valid ones sits at index (count-1)//2.
    plate = np.zeros((H, W, 3), np.uint8)
    rows = 128
    for y in range(0, H, rows):
        v = valid[:, y:y + rows]
        st = stack[:, y:y + rows].astype(np.uint16)
        st[~v] = 1000
        st.sort(axis=0)
        mid = ((np.maximum(coverage[y:y + rows], 1) - 1) // 2)[None, ..., None]
        plate[y:y + rows] = np.take_along_axis(st, np.broadcast_to(mid, (1,) + st.shape[1:]), 0)[0].clip(0, 255)
    hole = (coverage < max(3, cfg["plate_min_coverage"] * j)).astype(np.uint8) * 255
    pct = 100.0 * (hole > 0).mean()
    print(f"plate: {j} samples, {pct:.1f}% of the frame never seen -> inpainted ({cfg['inpainter']})")
    raw = plate.copy()
    if hole.any():
        plate = INPAINTERS[cfg["inpainter"]](plate, hole, cfg["inpaint_radius"])
    return Plate(plate, raw, 255 - hole, coverage)


def cached_plate(info: VideoInfo, scene: SceneAnalysis, cfg: dict, cache_dir: Path,
                 use_cache: bool = True) -> Plate:
    keycfg = {k: cfg[k] for k in ("analysis_height", "min_inliers", "mask_threshold",
                                  "temporal_dilate", "dilate_px", "feather_px", "plate_samples",
                                  "plate_min_coverage", "inpainter", "inpaint_radius",
                                  "static_threshold_px", "warp_smooth_frames")}
    keycfg["static"] = scene.static
    path = _cache_path(cache_dir, "plate", info, len(scene.masks), keycfg)
    if use_cache and path.exists():
        print(f"using cached background plate: {path.name}")
        d = np.load(path)
        return Plate(d["image"], d["raw"], d["seen"], d["coverage"])
    plate = build_plate(info, len(scene.masks), scene, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=plate.image, raw=plate.raw, seen=plate.seen, coverage=plate.coverage)
    return plate


# --- Per-frame removal --------------------------------------------------------------
class PersonRemover:
    """Per-frame person removal: fill the masked area from the (aligned) background plate,
    falling back to inpainting where the plate cannot help."""

    def __init__(self, scene: SceneAnalysis, plate: Plate, cfg: dict, scale: float = 1.0):
        """`scale` < 1 processes frames at reduced size (frames must be resized to match)."""
        self.cfg = cfg
        self.gain = None                                  # smoothed exposure gain (per channel)
        if scale != 1.0:
            ph, pw = plate.image.shape[:2]
            size = (int(round(pw * scale / 2) * 2), int(round(ph * scale / 2) * 2))
            sx, sy = size[0] / pw, size[1] / ph
            S, S_inv = np.diag([sx, sy, 1.0]), np.diag([1 / sx, 1 / sy, 1.0])
            plate = Plate(cv2.resize(plate.image, size, interpolation=cv2.INTER_AREA),
                          plate.raw, plate.seen, plate.coverage)
            scene = SceneAnalysis(scene.masks, scene.have, S @ scene.warps @ S_inv,
                                  scene.aligned, scene.static)
        self.scene, self.plate = scene, plate

    def _match_exposure(self, frame, fill, soft):
        # Per-channel gain measured in a ring of real background just outside the mask,
        # at quarter resolution, smoothed over time so it cannot flicker.
        h, w = frame.shape[:2]
        q = (max(8, w // 4), max(8, h // 4))
        fs, ps = cv2.resize(frame, q, interpolation=cv2.INTER_AREA), cv2.resize(fill, q, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(soft, q, interpolation=cv2.INTER_LINEAR) > 0
        r = max(3, int(0.03 * q[1]))
        ring = (cv2.dilate(ms.astype(np.uint8), np.ones((r, r), np.uint8)) > 0) & ~ms
        if ring.sum() < 50:
            return fill
        g = np.median(fs[ring].astype(np.float32), 0) / np.maximum(np.median(ps[ring].astype(np.float32), 0), 1)
        g = np.clip(g, 0.8, 1.25)
        self.gain = g if self.gain is None else 0.85 * self.gain + 0.15 * g
        if np.abs(self.gain - 1).max() < 0.005:
            return fill
        return cv2.convertScaleAbs(fill.astype(np.float32) * self.gain.astype(np.float32))

    def __call__(self, frame: np.ndarray, i: int) -> np.ndarray:
        scene, cfg = self.scene, self.cfg
        h, w = frame.shape[:2]
        soft = person_mask(scene, i, (w, h), cfg)
        x, y, bw, bh = cv2.boundingRect((soft > 0).astype(np.uint8))
        if bw == 0:
            return frame
        if scene.static:
            fill, fill_ok = self.plate.image, None
        elif scene.aligned[i]:
            A_inv = cv2.invertAffineTransform(scene.warps[i][:2])
            fill = cv2.warpAffine(self.plate.image, A_inv, (w, h), flags=cv2.INTER_LINEAR)
            fill_ok = cv2.warpAffine(np.full((h, w), 255, np.uint8), A_inv, (w, h),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
        else:
            fill, fill_ok = None, None
        if fill is None or (fill_ok is not None and (fill_ok[soft > 0] == 0).any()):
            # Camera moved too much to align, or the warped plate does not reach this area.
            hole = (soft > 0).astype(np.uint8) * 255
            if fill is not None:
                hole[fill_ok > 0] = 0
                src = np.where(fill_ok[..., None] > 0, fill, frame)
            else:
                src = frame
            fill = INPAINTERS[cfg["inpainter"]](src, hole, cfg["inpaint_radius"])
        elif cfg["color_match"]:
            fill = self._match_exposure(frame, fill, soft)
        out = frame.copy()
        sl = (slice(y, y + bh), slice(x, x + bw))
        a = soft[sl][..., None]
        out[sl] = (frame[sl] * (1 - a) + fill[sl] * a).astype(np.uint8)
        return out


# =============================================================================
# Audio: drum hits / onsets (drive pulses, flashes and scatter in the sci-fi creatures)
# =============================================================================
@dataclass
class Hits:
    times: np.ndarray       # [H] seconds
    strength: np.ndarray    # [H] ~0.2..1.5
    per_frame: np.ndarray   # [N] strongest hit starting in each video frame (0 = none)
    env: np.ndarray         # [N] decaying hit envelope (0..~2)


def detect_hits(info: VideoInfo, n_frames: int, cfg: dict) -> Hits:
    fps = float(info.fps)
    empty = Hits(np.zeros(0), np.zeros(0), np.zeros(n_frames, np.float32), np.zeros(n_frames, np.float32))
    if not info.has_audio:
        return empty
    sr = cfg["sample_rate"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(info.path), "-vn", "-ac", "1",
                          "-ar", str(sr), "-t", f"{n_frames / fps + 0.5:.3f}", "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    n_fft, hop = 1024, 256
    if len(x) < n_fft * 4:
        return empty
    nfr = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    flux = np.zeros(nfr, np.float32)
    prev = None
    for c in range(0, nfr, 2048):                       # spectral flux, chunked
        idx = np.arange(n_fft)[None, :] + hop * np.arange(c, min(nfr, c + 2048))[:, None]
        mag = np.log1p(100 * np.abs(np.fft.rfft(x[idx] * win, axis=1)))
        if prev is not None:
            mag = np.vstack([prev, mag])
        d = np.maximum(0, np.diff(mag, axis=0)).sum(1)
        flux[c + (0 if prev is not None else 1): c + (0 if prev is not None else 1) + len(d)] = d
        prev = mag[-1:]
    t = (np.arange(nfr) * hop + n_fft / 2) / sr
    w = max(3, int(cfg["median_window_s"] * sr / hop))
    from numpy.lib.stride_tricks import sliding_window_view
    med = np.median(sliding_window_view(np.pad(flux, (w // 2, w - w // 2 - 1), mode="edge"), w), axis=1)
    mad = np.median(np.abs(flux - np.median(flux))) + 1e-6
    thr = med + cfg["threshold"] * mad
    k = max(1, int(0.03 * sr / hop))
    loc_max = flux == np.max(sliding_window_view(np.pad(flux, (k, k), mode="edge"), 2 * k + 1), axis=1)
    cand = np.flatnonzero((flux > thr) & loc_max)
    peaks, last = [], -1e9
    for p in cand:                                      # enforce a minimum gap
        if t[p] - last >= cfg["min_gap_s"]:
            peaks.append(p)
            last = t[p]
    if not peaks:
        return empty
    peaks = np.array(peaks)
    st = flux[peaks] / (np.percentile(flux[peaks], 90) + 1e-6)
    st = np.clip(st, 0.2, 1.5).astype(np.float32)
    times = t[peaks]
    fr_t = np.arange(n_frames) / fps
    per_frame = np.zeros(n_frames, np.float32)
    fi = np.clip(np.round(times * fps).astype(int), 0, n_frames - 1)
    np.maximum.at(per_frame, fi, st)
    env = np.zeros(n_frames, np.float32)
    decay = cfg["decay_s"]
    for th, s in zip(times, st):
        a = int(np.ceil(th * fps))
        b = min(n_frames, a + int(decay * 6 * fps) + 1)
        if a < n_frames:
            env[a:b] += s * np.exp(-(fr_t[a:b] - th) / decay)
    print(f"audio: {len(times)} hits detected ({len(times) / max(fr_t[-1], 1e-3):.1f}/s)")
    return Hits(times, st, per_frame, env)


# =============================================================================
# Stage 3: creature body (capsules on the skeleton)
# =============================================================================
def hex_to_bgr(h: str) -> np.ndarray:
    h = h.lstrip("#")
    if len(h) != 6:
        sys.exit(f"error: bad colour '{h}', use #rrggbb")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)


def bgr_to_hsv_deg(bgr: np.ndarray) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(np.uint8([[np.clip(bgr, 0, 255)]]), cv2.COLOR_BGR2HSV)[0, 0]
    return hsv[0] * 2.0, hsv[1] / 255.0, hsv[2] / 255.0


def hsv_deg_to_bgr(h, s, v) -> np.ndarray:
    """Vectorised HSV (degrees, 0..1, 0..1) -> BGR float32 0..255."""
    h = np.asarray(h, np.float32) % 360
    s, v = np.broadcast_arrays(np.asarray(s, np.float32), np.asarray(v, np.float32))
    hsv = np.stack(np.broadcast_arrays(h / 2, s * 255, v * 255), -1).reshape(-1, 1, 3)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.reshape(np.broadcast(h, s).shape + (3,)).astype(np.float32)


# Extra virtual landmarks appended after MediaPipe's 33.
VIRTUAL_POINTS = {"mid_sh": (11, 12), "mid_hip": (23, 24), "head": (7, 8),
                  "hand_l": (17, 19), "hand_r": (18, 20)}
_VIDX = {name: 33 + k for k, name in enumerate(VIRTUAL_POINTS)}
# (name, joint a, joint b, radius as fraction of body scale W). a == b -> sphere.
BONES = [
    ("torso", "mid_sh", "mid_hip", 0.30), ("torso_l", 11, 23, 0.17), ("torso_r", 12, 24, 0.17),
    ("shoulders", 11, 12, 0.13), ("hips", 23, 24, 0.16), ("neck", "mid_sh", "head", 0.11),
    ("head", "head", "head", 0.30),
    ("upper_arm_l", 11, 13, 0.11), ("upper_arm_r", 12, 14, 0.11),
    ("forearm_l", 13, 15, 0.085), ("forearm_r", 14, 16, 0.085),
    ("hand_l", 15, "hand_l", 0.075), ("hand_r", 16, "hand_r", 0.075),
    ("thigh_l", 23, 25, 0.15), ("thigh_r", 24, 26, 0.15),
    ("shin_l", 25, 27, 0.11), ("shin_r", 26, 28, 0.11),
    ("foot_l", 29, 31, 0.065), ("foot_r", 30, 32, 0.065),
]
BONE_IA = np.array([_VIDX.get(a, a) for _, a, _, _ in BONES])
BONE_IB = np.array([_VIDX.get(b, b) for _, _, b, _ in BONES])
BONE_RF = np.array([r for *_, r in BONES], np.float32)
BONE_SPHERE = BONE_IA == BONE_IB


class Body:
    """Per-frame capsule geometry in output pixels. Depth z uses MediaPipe's z (same scale
    as x, smaller = closer to the camera), scaled down because it is the noisiest axis."""

    def __init__(self, pose: SmoothPose, scale: float, fps: float, cfg: dict):
        P = pose.img * scale
        P[..., 2] *= cfg["z_scale"]
        n = len(P)
        E = np.concatenate([P, np.zeros((n, len(VIRTUAL_POINTS), 3), np.float32)], 1)
        V = np.concatenate([pose.vis, np.zeros((n, len(VIRTUAL_POINTS)), np.float32)], 1)
        for name, (a, b) in VIRTUAL_POINTS.items():
            E[:, _VIDX[name]] = (P[:, a] + P[:, b]) / 2
            V[:, _VIDX[name]] = np.minimum(pose.vis[:, a], pose.vis[:, b])
        self.valid = pose.valid & ~np.isnan(E).any(axis=(1, 2))
        E = np.nan_to_num(E)
        # Body scale W: shoulder width, or torso length for side-on poses; smoothed.
        sw = np.linalg.norm(E[:, 11, :2] - E[:, 12, :2], axis=1)
        tl = np.linalg.norm(E[:, _VIDX["mid_sh"], :2] - E[:, _VIDX["mid_hip"], :2], axis=1)
        raw = np.maximum(sw, 0.75 * tl)
        idx = np.arange(n)
        if self.valid.any():
            raw = np.interp(idx, idx[self.valid], raw[self.valid])
            win = max(1, int(fps))
            from numpy.lib.stride_tricks import sliding_window_view
            raw = np.median(sliding_window_view(np.pad(raw, (win // 2, win - win // 2 - 1), mode="edge"), win), 1)
            sig = max(1.0, cfg["scale_smooth_s"] * fps)
            k = np.exp(-0.5 * (np.arange(-3 * int(sig), 3 * int(sig) + 1) / sig) ** 2)
            raw = np.convolve(np.pad(raw, len(k) // 2, mode="edge"), k / k.sum(), "valid")
        self.W = np.maximum(raw, 8.0).astype(np.float32)
        self.A, self.B = E[:, BONE_IA], E[:, BONE_IB]            # [N,nb,3]
        self.R = BONE_RF[None] * self.W[:, None] * cfg["radius_scale"]
        self.bone_vis = np.minimum(V[:, BONE_IA], V[:, BONE_IB])
        self.n_bones = len(BONES)
        v = self.valid if self.valid.any() else np.ones(n, bool)
        self.len_med = np.median(np.linalg.norm(self.B[v] - self.A[v], axis=-1), axis=0) + 1e-3
        self.W_med = float(np.median(self.W[v]))
        # Bones the person's own body hides (e.g. legs behind a table) are clipped to the
        # person mask so real foreground objects stay in front of the creature.
        self.occluded = np.median(self.bone_vis[v], axis=0) < cfg["occlusion_vis"]

    def area_weights(self, extra: np.ndarray | None = None) -> np.ndarray:
        rf = BONE_RF * self.W_med
        a = np.where(BONE_SPHERE, 4 * np.pi * rf ** 2, 2 * np.pi * rf * self.len_med)
        if extra is not None:
            a = a * extra
        return a / a.sum()

    def frames(self, i: int):
        A, B = self.A[i], self.B[i]
        d = B - A
        ln = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where((ln > 1e-3) & ~BONE_SPHERE[:, None], d / np.maximum(ln, 1e-3), [0.0, -1.0, 0.0])
        u = np.cross(d, [0.0, 0.0, 1.0])
        bad = np.linalg.norm(u, axis=1) < 1e-3
        u[bad] = np.cross(d[bad], [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v = np.cross(d, u)                  # theta = 90deg points toward the camera (-z)
        return A, B, d, u, v

    def surface(self, i: int, bone, t, th, rho=1.0):
        """Points on (rho<1: inside) the capsules. Returns (pos [n,3], normal [n,3])."""
        A, B, d, u, v = self.frames(i)
        R = self.R[i]
        ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
        nperp = ct * u[bone] + st * v[bone]
        rho = np.asarray(rho, np.float32)
        rr = (R[bone] * rho)[:, None] if rho.ndim else R[bone][:, None] * rho
        sph = BONE_SPHERE[bone][:, None]
        phi = np.arccos(np.clip(1 - 2 * t, -1, 1))[:, None]
        n_sph = np.cos(phi) * d[bone] + np.sin(phi) * nperp
        normal = np.where(sph, n_sph, nperp)
        base = np.where(sph, A[bone], A[bone] + t[:, None] * (B[bone] - A[bone]))
        return (base + rr * normal).astype(np.float32), normal.astype(np.float32)


# =============================================================================
# Stage 3: creatures
# =============================================================================
LINE = cv2.LINE_AA


class Creature:
    """Base class. step(i) advances the simulation to frame i (called once per frame, in
    order); draw() paints the bones selected by `sel` onto `img` and emissive light onto
    `glow` (uint8, bloomed and added afterwards)."""
    default_color = "#ff3fa4"

    def __init__(self, body: Body, args, hits: Hits, fps: float, size: tuple[int, int], seed=7):
        self.body, self.args, self.hits, self.fps = body, args, hits, fps
        self.w, self.h = size
        self.rng = np.random.default_rng(seed)
        self.color = hex_to_bgr(args.color or self.default_color)

    def stiffness(self, c: dict) -> float:
        """--stiffness if given, else this creature's default; 0..1."""
        v = self.args.stiffness if self.args.stiffness is not None else c.get("default_stiffness", 0.5)
        return float(np.clip(v, 0, 1))

    def step(self, i: int):
        pass

    def draw(self, img, glow, i, sel):
        raise NotImplementedError

    def draw_core(self, img, i, sel, color, shrink=0.85, opacity=1.0, shade=False):
        """Filled capsules (soft edged). With shade=True, a cylinder-like shading from a
        distance transform gives the body some volume."""
        b = self.body
        A, B, R = b.A[i], b.B[i], b.R[i] * shrink
        idx = [k for k in range(b.n_bones) if sel[k]]
        if not idx:
            return
        pad = int(R.max()) + 4
        xs = np.concatenate([A[idx, 0], B[idx, 0]])
        ys = np.concatenate([A[idx, 1], B[idx, 1]])
        x0, x1 = int(max(0, xs.min() - pad)), int(min(self.w, xs.max() + pad))
        y0, y1 = int(max(0, ys.min() - pad)), int(min(self.h, ys.max() + pad))
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0])
        for k in idx:
            a = tuple(int(c) for c in A[k, :2] - off)
            r = int(max(1, R[k]))
            if BONE_SPHERE[k]:
                cv2.circle(m, a, r, 255, -1, LINE)
            else:
                cv2.line(m, a, tuple(int(c) for c in B[k, :2] - off), 255, 2 * r, LINE)
        soft = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.012 * b.W[i])).astype(np.float32) / 255
        roi = img[y0:y1, x0:x1].astype(np.float32)
        col = np.broadcast_to(np.asarray(color, np.float32), roi.shape)
        if shade:
            dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5)
            s = np.clip(dist / max(4.0, 0.12 * b.W[i]), 0, 1)
            vol = np.sqrt(1 - (1 - s) ** 2)                       # rounded profile
            col = col * (0.35 + 0.65 * vol[..., None])
        a = (soft * opacity)[..., None]
        img[y0:y1, x0:x1] = (roi * (1 - a) + col * a).astype(np.uint8)

    def draw_contact_shadow(self, img, i):
        b = self.body
        k_l, k_r = len(BONES) - 2, len(BONES) - 1                    # foot bones
        if b.bone_vis[i, [k_l, k_r]].max() < 0.5:
            return
        pts = np.concatenate([b.A[i, [k_l, k_r], :2], b.B[i, [k_l, k_r], :2]])
        if pts[:, 1].max() > self.h * 0.995:
            return
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].max())
        W = float(b.W[i])
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(0.9 * W), int(0.12 * W)), 0, 0, 360, 255, -1, LINE)
        m = cv2.GaussianBlur(m, (0, 0), 0.12 * W)
        a = (m.astype(np.float32) / 255 * CONFIG["render"]["shadow_opacity"])[..., None]
        img[:] = (img * (1 - a)).astype(np.uint8)


# --- Strand creature (rubber hair) and fiber-optic variant -----------------------
class StrandCreature(Creature):
    default_color = "#ff3fa4"
    cfg_key = "creature"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        c = CONFIG[self.cfg_key]
        body, rng, args = self.body, self.rng, self.args
        n = max(10, int(c["strand_count"] * args.density))
        self.K = K = int(c["segments"])
        self.bones = rng.choice(body.n_bones, n, p=body.area_weights())
        self.t = rng.random(n).astype(np.float32)
        self.th = (rng.random(n) * 2 * np.pi).astype(np.float32)
        jit = c["length_jitter"]
        self.L = (c["hair_length"] * args.hair_length * (1 + jit * (2 * rng.random(n) - 1))).astype(np.float32)
        self.var = rng.standard_normal((n, 3)).astype(np.float32)
        ph = CONFIG["physics"]
        s = self.stiffness(c)
        # Three stiffness terms, all scaled by --stiffness:
        #  shape:    pull toward the straight rest pose (spikes). ~0 when floppy.
        #  follicle: keeps the first segments pointing out of the skin.
        #  bend:     each segment continues the previous one -> smooth curves, no kinks.
        k_shape = ph["k_shape_max"] * s ** 3
        prof = 1 - ph["tip_softness"] * np.arange(K) / (K - 1)
        kp = k_shape * prof
        nf = ph["follicle_points"]
        k_fol = ph["k_follicle_min"] + (ph["k_follicle_max"] - ph["k_follicle_min"]) * s
        kp[1:1 + nf] = np.maximum(kp[1:1 + nf], k_fol * np.linspace(1, 0.5, nf))
        kp[0] = 0
        self.kprof = kp.astype(np.float32)
        self.k_bend = float(ph["k_bend_min"] + (ph["k_bend_max"] - ph["k_bend_min"]) * s)
        self.damping = float(ph["damping_floppy"] + (ph["damping_stiff"] - ph["damping_floppy"]) * s)
        self.dt = 1.0 / ph["sim_rate"]
        self.P = self.Pp = None
        self.acc = 0.0
        self.root = self.nrm = None

    def _rest_dir(self, nrm):
        # Flatten the toward/away-from-camera component so strands on the front of the
        # body fan out sideways (reads much better in 2D than spikes aimed at the lens).
        f = CONFIG[self.cfg_key].get("fan_out", 0.0)
        d = nrm * np.array([1.0, 1.0, 1.0 - f], np.float32)
        side = np.linalg.norm(d[:, :2], axis=1, keepdims=True)
        # Strands aimed straight at the camera get a deterministic sideways direction.
        fallback = np.stack([np.cos(self.th * 3.7), np.sin(self.th * 3.7), np.zeros_like(self.th)], 1)
        d = np.where(side < 0.25, d + fallback * (0.25 - side) * 2 * f, d)
        return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)

    def _rest(self, root, nrm, seg):
        steps = np.arange(self.K, dtype=np.float32)
        return root[:, None] + self._rest_dir(nrm)[:, None] * (seg[:, None] * steps)[..., None]

    def step(self, i):
        b = self.body
        if not b.valid[i]:
            self.P = None
            return
        root, nrm = b.surface(i, self.bones, self.t, self.th)
        seg = self.L * b.W[i] / (self.K - 1)
        if self.P is None:
            self.P = self._rest(root, nrm, seg)
            self.Pp = self.P.copy()
            self.root, self.nrm, self.acc = root, nrm, 0.0
            return
        ph = CONFIG["physics"]
        self.acc += 1.0 / self.fps
        n = max(1, int(self.acc / self.dt))
        self.acc -= n * self.dt
        g = ph["gravity"] * float(b.W[i]) * self.dt * self.dt
        damp = 1.0 - self.damping
        kb, ftl, zf = self.k_bend, ph["ftl_damping"], ph["z_flatten"]
        r0, n0 = self.root, self.nrm
        P, Pp, kp = self.P, self.Pp, self.kprof[None, :, None]
        seg_c = seg[:, None]
        corr = np.zeros_like(P)
        for s in range(n):
            f = (s + 1) / n
            r = r0 + (root - r0) * f
            nn = n0 + (nrm - n0) * f
            nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-6
            vel = (P - Pp) * damp                            # Verlet: inertia ...
            Pp = P
            P = P + vel
            P[..., 1] += g                                   # ... gravity
            if zf > 0:
                P[:, 2:, 2] += (P[:, 1:2, 2] - P[:, 2:, 2]) * zf
            P += (self._rest(r, nn, seg) - P) * kp           # shape + follicle springs
            P[:, 0] = r
            prev_dir = self._rest_dir(nn)
            for j in range(1, self.K):
                if kb > 0 and j >= 2:                        # bending: continue previous segment
                    P[:, j] += (P[:, j - 1] + prev_dir * seg_c - P[:, j]) * kb
                d = P[:, j] - P[:, j - 1]                    # segment length (follow the leader)
                ln = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
                prev_dir = d / ln
                new = P[:, j - 1] + prev_dir * seg_c
                corr[:, j] = new - P[:, j]
                P[:, j] = new
            # Dynamic follow-the-leader (Mueller et al. 2012): hand each point's
            # correction back to its parent's velocity so the chain keeps its energy
            # and swings freely instead of being dragged dead by the constraint.
            Pp[:, :-1] += ftl * corr[:, 1:]
            Pp[:, 0] = r
        self.P, self.Pp, self.root, self.nrm = P, Pp, root, nrm

    # -- shading helpers
    def _tangent_light(self, P):
        rc = CONFIG["render"]
        T = P[:, 1:] - P[:, :-1]
        T /= np.linalg.norm(T, axis=-1, keepdims=True) + 1e-6
        L = np.asarray(rc["light_dir"], np.float32)
        L /= np.linalg.norm(L)
        H = L + np.array([0, 0, -1.0], np.float32)
        H /= np.linalg.norm(H)
        diffuse = np.sqrt(np.clip(1 - (T @ L) ** 2, 0, 1))
        spec = np.sqrt(np.clip(1 - (T @ H) ** 2, 0, 1)) ** rc["shininess"]
        return T, diffuse, spec

    def _order(self, sel_s):
        idx = np.flatnonzero(sel_s)
        depth = self.P[idx, :, 2].mean(1)
        back = self.nrm[idx, 2] > 0.35                       # pointing away from camera
        o_back = idx[back][np.argsort(-depth[back])]
        o_front = idx[~back][np.argsort(-depth[~back])]
        return o_back, o_front

    def draw(self, img, glow, i, sel):
        if self.P is None:
            return
        rc = CONFIG["render"]
        sel_s = sel[self.bones]
        if not sel_s.any():
            return
        W = float(self.body.W[i])
        base = self.color
        root_c = base * (1 - rc["root_darken"])
        tip_c = base + (255 - base) * rc["tip_lighten"]
        K = self.K
        g = np.linspace(0, 1, K - 1, dtype=np.float32)[None, :, None]
        T, diffuse, spec = self._tangent_light(self.P)
        var = 1 + rc["color_variation"] * self.var[:, None, :]
        col = (root_c + (tip_c - root_c) * g) * var * (0.35 + 0.65 * diffuse[..., None])
        hi = np.clip(col + rc["specular"] * 255 * spec[..., None], 0, 255)
        col = np.clip(col, 0, 255)
        widths = np.maximum(1, np.round((rc["root_width"] + (rc["tip_width"] - rc["root_width"]) * g[0, :, 0]) * W)).astype(int)
        hw = max(1, int(round(widths[: (K - 1) // 2].mean() / 3)))
        # Highlight: one continuous line per strand, offset toward the light, coloured by
        # the strand's average specular term (per-segment dashes looked like a ladder).
        L2 = -np.asarray(rc["light_dir"][:2], np.float32)
        L2 /= np.linalg.norm(L2) + 1e-6
        pts = np.round(self.P[..., :2]).astype(np.int32)
        hl_pts = np.round(self.P[:, :-1, :2] - L2 * (widths[None, :, None] * 0.28)).astype(np.int32)
        spec_m = spec.mean(1)
        hi_c = np.clip(col.mean(1) + rc["specular"] * 255 * spec_m[:, None], 0, 255)
        o_back, o_front = self._order(sel_s)

        def paint(order):
            P_l, C_l = pts[order].tolist(), col[order].tolist()
            wl = widths.tolist()
            for p, c in zip(P_l, C_l):
                for j in range(K - 1):
                    cv2.line(img, p[j], p[j + 1], c[j], wl[j], LINE)
            glossy = order[spec_m[order] > 0.2]
            for k in glossy.tolist():
                cv2.polylines(img, [hl_pts[k]], False, hi_c[k].tolist(), hw, LINE)

        paint(o_back)
        core = np.clip(base * (1 - rc["root_darken"]) * rc["core_darkness"], 0, 255)
        self.draw_core(img, i, sel, core, shrink=0.9)
        paint(o_front)


class FiberCreature(StrandCreature):
    """Fiber-optic neon: dark thin fibers with glowing, colour-shifting tips and light
    pulses racing root -> tip on every drum hit."""
    default_color = "#00e5ff"
    cfg_key = "fiber"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        c = CONFIG["fiber"]
        h, s, v = bgr_to_hsv_deg(self.color)
        self.hue0 = h
        self.hue = (h + c["hue_spread"] * self.rng.uniform(-1, 1, len(self.bones))).astype(np.float32)
        self.delay = self.rng.uniform(0, 0.12, len(self.bones)).astype(np.float32)

    def draw(self, img, glow, i, sel):
        if self.P is None:
            return
        c = CONFIG["fiber"]
        sel_s = sel[self.bones]
        if not sel_s.any():
            return
        W = float(self.body.W[i])
        K = self.K
        tnow = i / self.fps
        hue = self.hue + c["hue_drift"] * tnow
        g = np.linspace(0, 1, K, dtype=np.float32)
        # Light pulses from recent hits travelling along each fiber.
        boost = np.zeros((len(self.bones), K), np.float32)
        hits = self.hits
        if len(hits.times):
            recent = (hits.times <= tnow) & (hits.times > tnow - 1.2)
            for th, st in zip(hits.times[recent], hits.strength[recent]):
                pos = (tnow - th - self.delay) * c["pulse_speed"]
                boost += c["pulse_gain"] * st * np.exp(-0.5 * ((g[None] - pos[:, None]) / 0.12) ** 2)
        env = float(hits.env[i]) if len(hits.env) else 0.0
        bright = (0.25 + 0.75 * g ** 2)[None] * (c["base_glow"] + 0.2 * min(env, 1.5)) + boost
        emis = hsv_deg_to_bgr(hue[:, None].repeat(K, 1), 0.85 - 0.35 * np.clip(boost, 0, 1), 1.0)
        emis = np.clip(emis * np.clip(bright, 0, 1.3)[..., None], 0, 255)
        fiber_c = np.clip(self.color * 0.12 + 18, 0, 255)
        width = max(1, int(round(c["width"] * W)))
        tip_r = max(1, int(round(c["tip_size"] * W)))
        pts = np.round(self.P[..., :2]).astype(np.int32)
        o_back, o_front = self._order(sel_s)

        def paint(order):
            P_l, E_l = pts[order].tolist(), emis[order].tolist()
            for p, e in zip(P_l, E_l):
                for j in range(K - 1):
                    cv2.line(img, p[j], p[j + 1], fiber_c.tolist(), width, LINE)
                for j in range(K // 2, K - 1):           # light leaking from the outer half
                    cv2.line(img, p[j], p[j + 1], e[j + 1], width, LINE)
                    cv2.line(glow, p[j], p[j + 1], e[j + 1], width + 1, LINE)
                cv2.circle(img, p[-1], tip_r, e[-1], -1, LINE)
                cv2.circle(glow, p[-1], tip_r * 2, e[-1], -1, LINE)

        paint(o_back)
        self.draw_core(img, i, sel, (12, 8, 10), shrink=0.9)
        paint(o_front)


# --- Circuit-board skin ------------------------------------------------------------
class CircuitCreature(Creature):
    """Dark shaded body covered in PCB traces, vias and chips; light pulses run along the
    traces, ambiently and in bursts on drum hits."""
    default_color = "#19f0ff"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        c = CONFIG["circuit"]
        b, rng = self.body, self.rng
        n = max(10, int(c["traces"] * self.args.density))
        maxp = c["max_points"]
        bones = rng.choice(b.n_bones, n, p=b.area_weights())
        pts = np.zeros((n, maxp, 2), np.float32)            # (t, theta) per point
        cnt = np.zeros(n, int)
        for k in range(n):
            bo = bones[k]
            rf = BONE_RF[bo] * CONFIG["body"]["radius_scale"]
            sph = BONE_SPHERE[bo]
            # Walk in arc-length units (multiples of W) on the unrolled surface.
            ax_len = (np.pi * rf) if sph else b.len_med[bo] / b.W_med
            a = rng.uniform(0.1, 0.9) * ax_len
            th = rng.uniform(0, 2 * np.pi)
            ang = rng.integers(0, 8) * np.pi / 4
            nseg = rng.integers(c["min_segments"], maxp)
            out = [(a, th)]
            for _ in range(nseg):
                if rng.random() < c["turn_prob"]:
                    ang += rng.choice([-1, 1]) * np.pi / 4
                ln = rng.uniform(*c["step"]) * self.args.hair_length
                na = a + np.cos(ang) * ln
                nth = th + np.sin(ang) * ln / rf
                if not (0.02 * ax_len < na < 0.98 * ax_len):
                    break
                a, th = na, nth
                out.append((a, th))
            if len(out) < 2:
                out.append((a, th + 0.05 / rf))
            o = np.array(out, np.float32)
            if sph:
                o[:, 0] = (1 - np.cos(o[:, 0] / rf)) / 2      # polar angle -> t
            else:
                o[:, 0] = o[:, 0] / ax_len
            cnt[k] = len(o)
            pts[k, :len(o)] = o
            pts[k, len(o):] = o[-1]
        self.tr_bones, self.tr_pts, self.tr_cnt = bones, pts, cnt
        # Chips: small quads on the torso / upper arms.
        nchip = int(c["chips"] * self.args.density)
        chip_bones = rng.choice([0, 1, 2, 7, 8], nchip)
        self.chips = [(bo, rng.uniform(0.15, 0.85), rng.uniform(np.pi * 0.2, np.pi * 0.8),
                       rng.uniform(0.05, 0.10)) for bo in chip_bones]
        # Pulses (parallel arrays).
        self.p_tr = np.zeros(0, int)
        self.p_pos = np.zeros(0, np.float32)
        self.p_spd = np.zeros(0, np.float32)
        self.p_str = np.zeros(0, np.float32)
        hh, ss, vv = bgr_to_hsv_deg(self.color)
        self.hue = hh

    def _spawn(self, k, strength):
        if k <= 0:
            return
        c = CONFIG["circuit"]
        tr = self.rng.integers(0, len(self.tr_bones), k)
        self.p_tr = np.concatenate([self.p_tr, tr])
        self.p_pos = np.concatenate([self.p_pos, np.zeros(k, np.float32)])
        self.p_spd = np.concatenate([self.p_spd, (c["pulse_speed"] * self.rng.uniform(0.6, 1.4, k)).astype(np.float32)])
        self.p_str = np.concatenate([self.p_str, np.full(k, strength, np.float32)])

    def step(self, i):
        c = CONFIG["circuit"]
        dt = 1.0 / self.fps
        self.p_pos += self.p_spd * dt
        keep = self.p_pos < 1.0
        self.p_tr, self.p_pos, self.p_spd, self.p_str = (self.p_tr[keep], self.p_pos[keep],
                                                        self.p_spd[keep], self.p_str[keep])
        self._spawn(self.rng.poisson(c["ambient_pulses"] * dt * self.args.density), 0.7)
        hs = float(self.hits.per_frame[i]) if len(self.hits.per_frame) else 0.0
        if hs > 0:
            self._spawn(int(c["hit_pulses"] * hs * self.args.density), 0.7 + 0.5 * hs)

    def draw(self, img, glow, i, sel):
        b = self.body
        if not b.valid[i]:
            return
        c = CONFIG["circuit"]
        W = float(b.W[i])
        self.draw_core(img, i, sel, hex_to_bgr(c["body_color"]), shrink=1.0, shade=True)
        sel_t = sel[self.tr_bones]
        if not sel_t.any():
            return
        n, mp_ = self.tr_pts.shape[:2]
        flat_b = np.repeat(self.tr_bones, mp_)
        pos, nrm = b.surface(i, flat_b, self.tr_pts[..., 0].ravel(), self.tr_pts[..., 1].ravel())
        pos = pos.reshape(n, mp_, 3)
        facing = (-nrm[:, 2]).reshape(n, mp_)               # >0 faces the camera
        env = float(self.hits.env[i]) if len(self.hits.env) else 0.0
        level = c["base_brightness"] + (1 - c["base_brightness"]) * min(1.0, env * 0.6)
        tw = max(1, int(round(c["trace_width"] * W)))
        via = max(2, int(round(tw * 1.8)))
        p2 = np.round(pos[..., :2]).astype(np.int32)
        base = self.color
        fl = facing.tolist()
        pl = p2.tolist()
        cl = self.tr_cnt.tolist()
        for k in np.flatnonzero(sel_t).tolist():
            p, f, m = pl[k], fl[k], cl[k]
            for j in range(m - 1):
                ff = min(f[j], f[j + 1])
                if ff <= 0.05:
                    continue
                col = (base * (level * ff ** 0.5)).tolist()
                cv2.line(img, p[j], p[j + 1], col, tw, LINE)
                cv2.line(glow, p[j], p[j + 1], (base * (0.35 * level * ff)).tolist(), tw, LINE)
            for j in (0, m - 1):
                if f[j] > 0.1:
                    col = (base * (level * f[j] ** 0.5)).tolist()
                    cv2.circle(img, p[j], via, col, max(1, tw // 2 + 1), LINE)
        # Chips
        for bo, t0, th0, sz in self.chips:
            if not sel[bo]:
                continue
            rf = BONE_RF[bo]
            dt_ = sz * b.W_med / b.len_med[bo]
            dth = sz / rf
            q_t = np.array([t0 - dt_, t0 + dt_, t0 + dt_, t0 - dt_], np.float32)
            q_th = np.array([th0 - dth, th0 - dth, th0 + dth, th0 + dth], np.float32)
            qp, qn = b.surface(i, np.full(4, bo), q_t, q_th)
            f = float((-qn[:, 2]).min())
            if f <= 0.15:
                continue
            quad = np.round(qp[:, :2]).astype(np.int32)
            cv2.fillConvexPoly(img, quad, (22, 22, 26), LINE)
            cv2.polylines(img, [quad], True, (base * level * f).tolist(), max(1, tw // 2), LINE)
        # Pulses
        if len(self.p_tr):
            sel_p = sel_t[self.p_tr]
            for tr, ps, st in zip(self.p_tr[sel_p].tolist(), self.p_pos[sel_p].tolist(), self.p_str[sel_p].tolist()):
                m = cl[tr]
                x = ps * (m - 1)
                j = min(int(x), m - 2)
                fr = x - j
                f = min(fl[tr][j], fl[tr][j + 1])
                if f <= 0.05:
                    continue
                a, bb = pos[tr, j, :2], pos[tr, j + 1, :2]
                pt = tuple(int(v) for v in np.round(a + (bb - a) * fr))
                hot = np.clip(base * 0.5 + 160, 0, 255) * st * f
                cv2.circle(img, pt, tw + 1, np.clip(hot, 0, 255).tolist(), -1, LINE)
                cv2.circle(glow, pt, tw * 4, np.clip(base * st * f * 1.2, 0, 255).tolist(), -1, LINE)


# --- Voxel swarm -----------------------------------------------------------------------
class SwarmCreature(Creature):
    """Thousands of small cubes sprung to points inside the body capsules. They lag and
    overshoot on fast moves and are blasted outward around the striking hand on drum hits."""
    default_color = "#ff3fa4"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        c = CONFIG["swarm"]
        b, rng = self.body, self.rng
        count = CONFIG["vowels"]["count"] if isinstance(self, VowelSwarmCreature) else c["count"]
        n = max(10, int(count * self.args.density))
        self.bones = rng.choice(b.n_bones, n, p=b.area_weights())
        self.t = rng.random(n).astype(np.float32)
        self.th = (rng.random(n) * 2 * np.pi).astype(np.float32)
        self.rho = np.sqrt(rng.uniform(0.25, 1.0, n)).astype(np.float32)
        self.size = (c["cube_size"] * self.args.hair_length * rng.uniform(0.7, 1.3, n)).astype(np.float32)
        h, s, v = bgr_to_hsv_deg(self.color)
        hue = h + c["hue_spread"] * rng.uniform(-1, 1, n)
        self.col = hsv_deg_to_bgr(hue, np.clip(s * rng.uniform(0.8, 1.1, n), 0, 1),
                                  np.clip(v * rng.uniform(0.65, 1.05, n), 0, 1))
        stiff = self.stiffness(c)
        f_hz = c["freq_min"] + (c["freq_max"] - c["freq_min"]) * stiff
        self.omega = (2 * np.pi * f_hz * rng.uniform(0.8, 1.2, n)).astype(np.float32)
        self.zeta = c["damping_ratio"]
        self.dt = 1.0 / CONFIG["physics"]["sim_rate"]
        self.X = self.V = None
        self.tgt = None
        self.acc = 0.0
        self.prev_wrists = None

    def step(self, i):
        b, c = self.body, CONFIG["swarm"]
        if not b.valid[i]:
            self.X = None
            return
        tgt, _ = b.surface(i, self.bones, self.t, self.th, self.rho)
        W = float(b.W[i])
        wr = np.stack([b.B[i, 11], b.B[i, 12]])                  # hand tips
        if self.X is None:
            self.X, self.V, self.tgt, self.acc = tgt.copy(), np.zeros_like(tgt), tgt, 0.0
            self.prev_wrists = wr
            return
        # Drum hit: blast cubes outward around the hand that is moving down fastest.
        hs = float(self.hits.per_frame[i]) if len(self.hits.per_frame) else 0.0
        if hs > 0:
            vy = wr[:, 1] - self.prev_wrists[:, 1]
            center = wr[int(np.argmax(vy))]
            d = self.X - center
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
            fall = np.clip(1 - dist / (c["scatter_radius"] * W), 0, 1) ** 2
            rnd = self.rng.standard_normal(self.X.shape).astype(np.float32) * 0.6
            self.V += (d / dist + rnd) * fall * (c["scatter_speed"] * W * hs)
        self.prev_wrists = wr
        self.acc += 1.0 / self.fps
        n = max(1, int(self.acc / self.dt))
        self.acc -= n * self.dt
        t0, w2 = self.tgt, (self.omega ** 2)[:, None]
        c2 = (2 * self.zeta * self.omega)[:, None]
        X, V = self.X, self.V
        for s in range(n):
            T = t0 + (tgt - t0) * ((s + 1) / n)
            V += (w2 * (T - X) - c2 * V) * self.dt
            X += V * self.dt
        self.X, self.V, self.tgt = X, V, tgt

    def draw(self, img, glow, i, sel):
        if self.X is None:
            return
        sel_p = sel[self.bones]
        if not sel_p.any():
            return
        W = float(self.body.W[i])
        idx = np.flatnonzero(sel_p)
        idx = idx[np.argsort(-self.X[idx, 2])]                 # back to front
        h = (self.size[idx] * W / 2)
        o = h * 0.55
        x, y = self.X[idx, 0], self.X[idx, 1]
        speed = np.linalg.norm(self.V[idx], axis=1) / (W * 3.0)
        hot = np.clip(speed, 0, 1)[:, None]
        col = self.col[idx]
        front = np.clip(col * 0.85 + hot * 120, 0, 255)
        top = np.clip(col * 1.25 + 30 + hot * 120, 0, 255)
        side = np.clip(col * 0.5 + hot * 80, 0, 255)
        F = np.stack([np.stack([x - h, y - h], 1), np.stack([x + h, y - h], 1),
                      np.stack([x + h, y + h], 1), np.stack([x - h, y + h], 1)], 1)
        Tq = np.stack([np.stack([x - h, y - h], 1), np.stack([x + h, y - h], 1),
                       np.stack([x + h + o, y - h - o], 1), np.stack([x - h + o, y - h - o], 1)], 1)
        Sq = np.stack([np.stack([x + h, y - h], 1), np.stack([x + h + o, y - h - o], 1),
                       np.stack([x + h + o, y + h - o], 1), np.stack([x + h, y + h], 1)], 1)
        F, Tq, Sq = (np.round(q).astype(np.int32) for q in (F, Tq, Sq))
        for f, t, s_, cf, ct, cs in zip(F, Tq, Sq, front.tolist(), top.tolist(), side.tolist()):
            cv2.fillConvexPoly(img, t, ct)
            cv2.fillConvexPoly(img, s_, cs)
            cv2.fillConvexPoly(img, f, cf)
        fast = np.flatnonzero(hot[:, 0] > 0.25)
        for k in fast.tolist():
            cv2.circle(glow, (int(x[k]), int(y[k])), int(h[k] * 2 + 1),
                       (col[k] * hot[k, 0]).tolist(), -1, LINE)


class VowelSwarmCreature(SwarmCreature):
    """Same swarm physics, but every particle is a vowel (A E I O U) - a typographic
    creature that spells nothing and scatters into letters on each drum hit."""
    default_color = "#ffd23f"
    LETTERS = "AEIOU"
    FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        n = len(self.bones)
        self.letter = self.rng.integers(0, len(self.LETTERS), n)
        self.font = self.rng.integers(0, len(self.FONTS), n)
        self.size *= CONFIG["vowels"]["size_scale"]
        self.swap_rate = CONFIG["vowels"]["swap_rate"]

    def step(self, i):
        super().step(i)
        if self.X is not None:                               # letters flicker to other vowels
            k = self.rng.random(len(self.letter)) < self.swap_rate / self.fps
            self.letter[k] = self.rng.integers(0, len(self.LETTERS), int(k.sum()))

    def draw(self, img, glow, i, sel):
        if self.X is None:
            return
        sel_p = sel[self.bones]
        if not sel_p.any():
            return
        W = float(self.body.W[i])
        idx = np.flatnonzero(sel_p)
        idx = idx[np.argsort(-self.X[idx, 2])]
        speed = np.clip(np.linalg.norm(self.V[idx], axis=1) / (W * 3.0), 0, 1)
        # Depth shading: letters further back are darker.
        z = self.X[idx, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-6)
        col = self.col[idx] * (1.05 - 0.45 * zn[:, None]) + speed[:, None] * 120
        col = np.clip(col, 0, 255)
        dark = np.clip(self.col[idx] * 0.2, 0, 255)
        scale = self.size[idx] * W / 22.0                      # Hershey glyphs are ~22px at scale 1
        thick = np.maximum(1, np.round(scale * 2.2)).astype(int)
        x, y = self.X[idx, 0], self.X[idx, 1]
        for k, (xx, yy, sc, th, c, d, l, f, sp) in enumerate(zip(
                x.tolist(), y.tolist(), scale.tolist(), thick.tolist(), col.tolist(), dark.tolist(),
                self.letter[idx].tolist(), self.font[idx].tolist(), speed.tolist())):
            ch = self.LETTERS[l]
            font = self.FONTS[f]
            org = (int(xx - sc * 8), int(yy + sc * 8))
            cv2.putText(img, ch, org, font, sc, d, th + 2, LINE)    # dark outline for legibility
            cv2.putText(img, ch, org, font, sc, c, th, LINE)
            if sp > 0.25:
                cv2.putText(glow, ch, org, font, sc, (np.array(c) * sp).tolist(), th + 2, LINE)


CREATURES = {"strand": StrandCreature, "fiber": FiberCreature, "circuit": CircuitCreature,
             "swarm": SwarmCreature, "vowels": VowelSwarmCreature}
DEFAULT_GRADE = {"strand": "none", "fiber": "cyber", "circuit": "cyber", "swarm": "none",
                 "vowels": "cyber"}


# --- Post: bloom and colour grade ---------------------------------------------------
def add_glow(img: np.ndarray, glow: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0 or not glow.any():
        return img
    h, w = img.shape[:2]
    small = cv2.resize(glow, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(small, (0, 0), max(1.0, h / 400))
    b2 = cv2.GaussianBlur(small, (0, 0), max(2.0, h / 120))
    bloom = cv2.resize(cv2.addWeighted(b1, 1.0, b2, 1.0, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(img, 1.0, glow, 0.5 * strength, 0)
    return cv2.addWeighted(out, 1.0, bloom, strength, 0)


_GRADE_CACHE: dict = {}


def grade_frame(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    gc = CONFIG["grade"]
    h, w = img.shape[:2]
    key = (h, w)
    if key not in _GRADE_CACHE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        _GRADE_CACHE[key] = (1 - gc["vignette"] * np.clip(r - 0.3, 0, 1) ** 1.5)[..., None]
        lut = np.arange(256, dtype=np.float32) / 255
        _GRADE_CACHE["lut"] = (lut ** gc["gamma"] * gc["exposure"])
    vig = _GRADE_CACHE[key]
    f = cv2.LUT(img, np.clip(_GRADE_CACHE["lut"] * 255, 0, 255).astype(np.uint8)).astype(np.float32) / 255
    lum = f.mean(axis=2, keepdims=True)
    sh, hi = hex_to_bgr(gc["shadow_tint"]) / 255, hex_to_bgr(gc["highlight_tint"]) / 255
    tint = sh + (hi - sh) * np.clip(lum * 2.5, 0, 1)
    f = f * (1 - gc["tint"]) + f * tint * 2.0 * gc["tint"]
    return np.clip(f * vig * 255, 0, 255).astype(np.uint8)


# =============================================================================
# Debug renderers
# =============================================================================
def draw_skeleton(frame: np.ndarray, pose: SmoothPose, i: int) -> np.ndarray:
    out = frame.copy()
    h = frame.shape[0]
    s = max(1.0, h / 540)                      # line scale
    raw = pose.raw
    # Raw detections: small grey dots, to show what the filter removes.
    if raw.detected[i]:
        for p in raw.img[i]:
            cv2.circle(out, (int(p[0]), int(p[1])), int(2 * s), (150, 150, 150), -1, cv2.LINE_AA)
    if pose.valid[i] and pose.alpha[i] > 0:
        layer = out.copy()
        pts = pose.img[i]
        for a, b in POSE_CONNECTIONS:
            col = (60, 200, 255) if a in LEFT_IDS else (255, 160, 60) if a in RIGHT_IDS else (230, 230, 230)
            cv2.line(layer, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])),
                     col, max(1, int(2 * s)), cv2.LINE_AA)
        z = pts[:, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-6)  # 0 = closest
        for k, p in enumerate(pts):
            r = int((3 + 4 * (1 - zn[k])) * s)   # nearer joints drawn bigger
            v = pose.vis[i, k]
            col = (0, int(255 * v), int(255 * (1 - v)))  # green = visible, red = occluded
            cv2.circle(layer, (int(p[0]), int(p[1])), r, col, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, pose.alpha[i], out, 1 - pose.alpha[i], 0, out)
    if raw.detected[i]:
        status, col = "DETECTED", (80, 220, 80)
    elif pose.alpha[i] > 0:
        status, col = "INTERPOLATED", (0, 200, 255)
    else:
        status, col = "LOST", (60, 60, 255)
    txt = f"frame {i}  {status}  alpha={pose.alpha[i]:.2f}"
    cv2.putText(out, txt, (int(10 * s), int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                (0, 0, 0), int(3 * s), cv2.LINE_AA)
    cv2.putText(out, txt, (int(10 * s), int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                col, max(1, int(s)), cv2.LINE_AA)
    return out


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replace a person in a video with a rubbery hair creature.")
    p.add_argument("input", type=Path, help="input video (one person, ideally full body)")
    p.add_argument("-o", "--output", type=Path, default=None, help="output .mp4 (default: <input>_hairy.mp4)")
    p.add_argument("--creature", choices=list(CREATURES), default="strand",
                   help="creature style (default %(default)s)")
    p.add_argument("--color", default=None, help="base colour, hex (default depends on the creature)")
    p.add_argument("--grade", choices=["auto", "none", "cyber"], default="auto",
                   help="background colour grade (auto: cyber for fiber/circuit/vowels)")
    p.add_argument("--hair-length", type=float, default=1.0,
                   help="strand length / trace length / cube size multiplier (default 1.0)")
    p.add_argument("--stiffness", type=float, default=None,
                   help="0 = floppy hair / cooked spaghetti, 1 = stiff spikes (default per creature)")
    p.add_argument("--density", type=float, default=1.0,
                   help="strand / trace / particle count multiplier (default 1.0)")
    p.add_argument("--preview", action="store_true",
                   help=f"only the first {CONFIG['preview']['seconds']:g}s at half resolution")
    p.add_argument("--keep-person", action="store_true", help="skip person removal (draw creature on top)")
    p.add_argument("--stage", choices=["skeleton", "background", "final"], default="final",
                   help="render a debug video for one stage (default: final)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default=CONFIG["pose"]["model"],
                   help="pose model size (default %(default)s)")
    p.add_argument("--no-cache", action="store_true", help="re-run pose tracking even if cached")
    return p.parse_args(argv)


def render_creature(bg, i, creature, grade, glow_strength, occl, pose, scene, rcfg):
    bg = grade_frame(bg, grade)
    creature.step(i)
    alpha = float(pose.alpha[i]) if creature.body.valid[i] else 0.0
    if alpha <= 0:
        return bg
    h, w = bg.shape[:2]
    out = bg.copy()
    glow = np.zeros_like(out)
    creature.draw_contact_shadow(out, i)
    if occl.any():
        # Hidden body parts first, kept only where the person was (so e.g. a table or
        # drum pad in front of the legs stays in front of the creature).
        tmp, g2 = out.copy(), np.zeros_like(out)
        creature.draw(tmp, g2, i, occl)
        keep = person_mask(scene, i, (w, h), rcfg)[..., None]
        out = (out * (1 - keep) + tmp * keep).astype(np.uint8)
        glow = (g2 * keep).astype(np.uint8)
    creature.draw(out, glow, i, ~occl)
    out = add_glow(out, glow, glow_strength)
    if alpha < 1:
        out = cv2.addWeighted(bg, 1 - alpha, out, alpha, 0)
    return out


class StageTimer:
    def __init__(self):
        self.t0 = self.t = time.time()

    def __call__(self, name: str):
        now = time.time()
        print(f"  [{name}: {now - self.t:.1f}s]")
        self.t = now


def main(argv=None):
    t0 = time.time()
    lap = StageTimer()
    args = parse_args(argv)
    check_ffmpeg()
    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")
    CONFIG["pose"]["model"] = args.model

    info = probe_video(args.input)
    fps = float(info.fps)
    n_frames = info.n_frames
    out_scale = 1.0
    out_duration = None
    if args.preview:
        n_frames = min(n_frames, int(round(CONFIG["preview"]["seconds"] * fps)))
        out_scale = CONFIG["preview"]["scale"]
        out_duration = n_frames / fps
    out_w = int(round(info.width * out_scale / 2) * 2)   # even sizes for yuv420p
    out_h = int(round(info.height * out_scale / 2) * 2)
    if args.output is None:
        suffix = f"_{args.creature}" if args.stage == "final" else f"_{args.stage}"
        suffix += "_preview" if args.preview else ""
        args.output = args.input.with_name(f"{args.input.stem}_hairy{suffix}.mp4")
    print(f"input : {info.path.name}  {info.width}x{info.height} @ {fps:.3f} fps, "
          f"{info.n_frames} frames, audio={'yes' if info.has_audio else 'no'}")
    print(f"output: {args.output}  {out_w}x{out_h}, {n_frames} frames, stage={args.stage}")

    # --- Stage 1: pose ------------------------------------------------------
    cache_dir = Path(CONFIG["cache_dir"])
    if args.no_cache:
        track = track_pose(info, n_frames, CONFIG["pose"])
    else:
        track = cached_pose(info, n_frames, CONFIG["pose"], cache_dir)
    n_frames = min(n_frames, len(track.detected))
    pose = smooth_pose(track, fps, CONFIG["smoothing"], info.height)
    print(f"pose detected in {track.detected.sum()}/{len(track.detected)} frames")
    lap("pose")

    # --- Stage 2: person removal ---------------------------------------------
    scene = plate = remover = None
    rcfg = CONFIG["removal"]
    if args.stage != "skeleton" and not args.keep_person:
        # The plate samples the whole clip even in preview mode: the person may only move
        # out of the way after the preview window.
        need = np.zeros(info.n_frames, bool)
        need[:n_frames] = True                              # frames we render
        need[plate_sample_frames(info.n_frames, rcfg)] = True  # frames the plate is built from
        scene = cached_scene(info, need, rcfg, cache_dir, not args.no_cache)
        lap("segmentation")
        plate = cached_plate(info, scene, rcfg, cache_dir, not args.no_cache)
        remover = PersonRemover(scene, plate, rcfg, out_scale)
        plate_png = args.output.with_name(args.output.stem + "_plate.png")
        cv2.imwrite(str(plate_png), plate.image)
        print(f"background plate saved: {plate_png}")
        lap("background plate")
    # --- Stage 3: creature --------------------------------------------------
    creature = None
    if args.stage == "final":
        hits = detect_hits(info, n_frames, CONFIG["audio"])
        body = Body(pose, out_w / info.width, fps, CONFIG["body"])
        if scene is not None and body.occluded.any():
            print("clipped to the person mask (hidden in the video): " +
                  ", ".join(BONES[k][0] for k in np.flatnonzero(body.occluded)))
        creature = CREATURES[args.creature](body, args, hits, fps, (out_w, out_h))
        grade = DEFAULT_GRADE[args.creature] if args.grade == "auto" else args.grade
        glow_strength = CONFIG.get(args.creature, {}).get("glow", 0.0)
        if args.creature == "vowels":
            glow_strength = CONFIG["swarm"]["glow"]
        occl = body.occluded if scene is not None else np.zeros(body.n_bones, bool)
        lap("creature setup")

    # --- Render -------------------------------------------------------------
    writer = FFmpegWriter(args.output, out_w, out_h, info.fps,
                          info.path if info.has_audio else None, out_duration, CONFIG["encode"],
                          info.audio_codec)
    for i, frame in tqdm(iter_frames(info.path, n_frames), total=n_frames, desc="rendering", unit="f"):
        if out_scale != 1.0 and args.stage != "skeleton":
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if args.stage == "skeleton":
            out = draw_skeleton(frame, pose, i)
        else:
            out = frame if scene is None else remover(frame, i)
        if creature is not None:
            out = render_creature(out, i, creature, grade, glow_strength, occl, pose, scene, rcfg)
        writer.write(out)
    writer.close()
    lap("render + encode")
    print(f"done: {args.output}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
