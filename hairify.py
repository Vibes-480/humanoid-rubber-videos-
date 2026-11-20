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
        "mask_threshold": 0.35,
        "dilate_px": 0.02,               # fraction of frame height
        "feather_px": 0.015,
        "plate_samples": 60,             # frames sampled for the median background plate
        "plate_min_coverage": 3,         # min unmasked samples per pixel, else inpaint
        "inpaint_radius": 5,
    },
    "creature": {
        "strand_count": 3000,            # total strands at --density 1.0
        "segments": 8,                   # points per strand (6-10)
        "hair_length": 0.09,             # strand length as fraction of body height at --hair-length 1.0
        "length_jitter": 0.25,
        "core_scale": 1.0,
    },
    "physics": {
        "substeps": 4,
        "sim_rate": 120.0,               # physics steps/s (independent of video fps)
        "gravity": 2.5,                  # body heights / s^2
        "damping": 0.06,                 # velocity loss per 1/120 s step
        "stiffness": 0.5,                # pull toward rest direction (0..1, --stiffness)
        "constraint_iters": 2,
    },
    "render": {
        "color": "#ff3fa4",
        "root_darken": 0.45,             # root colour = base * (1 - root_darken)
        "tip_lighten": 0.30,
        "color_variation": 0.08,
        "root_width": 0.006,             # fraction of frame height
        "tip_width": 0.0015,
        "light_dir": (-0.4, -0.7, 0.6),
        "specular": 0.6,
        "shininess": 24.0,
        "core_color": "#2a0716",
        "shadow_opacity": 0.45,
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
    return VideoInfo(path, w, h, fps, n, duration,
                     any(s["codec_type"] == "audio" for s in meta["streams"]))


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
                 audio_src: Path | None, duration: float | None, enc: dict):
        self.out_path = out_path
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
        if audio_src is not None:
            if duration is not None:
                cmd += ["-t", f"{duration:.3f}"]
            cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", enc["audio_codec"], "-b:a", enc["audio_bitrate"], "-shortest"]
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
    p.add_argument("--color", default=CONFIG["render"]["color"], help="strand base colour, hex (default %(default)s)")
    p.add_argument("--hair-length", type=float, default=1.0, help="strand length multiplier (default 1.0)")
    p.add_argument("--stiffness", type=float, default=CONFIG["physics"]["stiffness"],
                   help="0 = floppy, 1 = stiff spikes (default %(default)s)")
    p.add_argument("--density", type=float, default=1.0, help="strand count multiplier (default 1.0)")
    p.add_argument("--preview", action="store_true",
                   help=f"only the first {CONFIG['preview']['seconds']:g}s at half resolution")
    p.add_argument("--keep-person", action="store_true", help="skip person removal (draw creature on top)")
    p.add_argument("--stage", choices=["skeleton", "background", "final"], default="final",
                   help="render a debug video for one stage (default: final)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default=CONFIG["pose"]["model"],
                   help="pose model size (default %(default)s)")
    p.add_argument("--no-cache", action="store_true", help="re-run pose tracking even if cached")
    return p.parse_args(argv)


def main(argv=None):
    t0 = time.time()
    args = parse_args(argv)
    check_ffmpeg()
    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")
    CONFIG["pose"]["model"] = args.model
    CONFIG["render"]["color"] = args.color
    CONFIG["physics"]["stiffness"] = float(np.clip(args.stiffness, 0, 1))

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
        suffix = "" if args.stage == "final" else f"_{args.stage}"
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

    if args.stage != "skeleton":
        sys.exit(f"stage '{args.stage}' is not implemented yet - run with --stage skeleton")

    # --- Render -------------------------------------------------------------
    writer = FFmpegWriter(args.output, out_w, out_h, info.fps,
                          info.path if info.has_audio else None, out_duration, CONFIG["encode"])
    for i, frame in tqdm(iter_frames(info.path, n_frames), total=n_frames, desc="rendering", unit="f"):
        writer.write(draw_skeleton(frame, pose, i))
    writer.close()
    print(f"done: {args.output}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
