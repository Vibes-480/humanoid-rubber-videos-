#!/usr/bin/env bash
# Create .venv with a MediaPipe-compatible Python, install deps, fetch the pose models.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    v=$("$cand" -c 'import sys; print("%d%02d" % sys.version_info[:2])')
    if [ "$v" -ge 309 ] && [ "$v" -le 312 ]; then PY="$cand"; break; fi
  fi
done
[ -n "$PY" ] || { echo "error: need Python 3.9-3.12 for MediaPipe"; exit 1; }
echo "using $PY ($($PY --version))"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found (apt install ffmpeg / brew install ffmpeg)"; exit 1; }

"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

mkdir -p models
for m in full heavy; do
  f="models/pose_landmarker_$m.task"
  [ -f "$f" ] || curl -fL -o "$f" "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task"
done

# On headless Linux MediaPipe needs libEGL / libGLESv2 even on CPU.
if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libEGL.so.1; then
  echo "note: install EGL libs first: sudo apt install libegl1 libgles2"
fi
echo "ready: .venv/bin/python hairify.py input.mp4 --preview"
