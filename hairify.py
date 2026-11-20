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
    if args.stage == "final":
        sys.exit("stage 'final' is not implemented yet - use --stage skeleton or --stage background")

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
        writer.write(out)
    writer.close()
    lap("render + encode")
    print(f"done: {args.output}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
