# hairify

Replace the person in a short video with a CGI creature made of rubbery, jiggling
hair/spike strands, in the spirit of Method Studios' "AICP 2016 Sponsor Reel".

> **Status:** stage (a) pose tracking + skeleton overlay is done. Person removal (b)
> and the creature (c) come next. Until then, use `--stage skeleton`.

## Setup

Requires **Python 3.9–3.12** (MediaPipe wheels) and **ffmpeg** on your PATH.

```bash
./setup.sh               # creates .venv, installs requirements.txt, downloads pose models
```

Manual equivalent:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On headless Linux, MediaPipe also needs the EGL/GLES libraries (it uses them even on CPU):
`sudo apt install libegl1 libgles2`.

## Usage

```bash
.venv/bin/python hairify.py input.mp4 -o output.mp4 --color "#ff3fa4" \
    --hair-length 1.0 --stiffness 0.5 --density 1.0

# fast iteration: first 5 s at half resolution
.venv/bin/python hairify.py input.mp4 --preview

# debug videos for each stage
.venv/bin/python hairify.py input.mp4 --preview --stage skeleton
```

The output has the same resolution, fps and length as the input, with the original audio
(H.264, CRF 18, preset veryfast, yuv420p). Pose tracking is cached in `.hairify_cache/`, so
re-rendering the same clip with different look settings skips the slow part.

## Options

| Option | Default | What it does |
|---|---|---|
| `input` | | Input video: one person, ideally full body, static camera. |
| `-o, --output` | `<input>_hairy.mp4` | Output MP4 path. |
| `--color` | `#ff3fa4` | Strand base colour (hex). Roots are darker, tips lighter. |
| `--hair-length` | `1.0` | Strand length multiplier. |
| `--stiffness` | `0.5` | 0 = floppy wet noodles, 1 = stiff spikes. |
| `--density` | `1.0` | Strand count multiplier (1.0 ≈ 3000 strands). |
| `--preview` | off | Render only the first 5 s at half resolution. |
| `--keep-person` | off | Skip person removal; draw the creature on top (debugging). |
| `--stage` | `final` | `skeleton` = pose debug overlay, `background` = person removed, `final` = creature. |
| `--model` | `full` | Pose model: `lite` (fastest), `full`, `heavy` (most accurate). |
| `--no-cache` | off | Re-run pose tracking even if a cached result exists. |

All other tunables (smoothing, strand/segment counts, gravity, damping, colours, widths,
encoder settings) are in the `CONFIG` dict at the top of `hairify.py`.

## Skeleton debug video (`--stage skeleton`)

- grey dots: raw MediaPipe landmarks; coloured skeleton: One Euro–smoothed pose
  (orange = left side, blue = right side)
- joint colour: green = visible, red = occluded; joint size grows as the joint gets closer to the camera (z)
- top-left label: `DETECTED`, `INTERPOLATED` (short gap, filled), or `LOST` (creature faded out)

## How tracking handles dropouts

- Gaps shorter than `max_interp_gap_s` (0.5 s) are linearly interpolated.
- Longer gaps hold the last pose and fade the creature out over `fade_s`; it fades back in
  when the person is detected again, and the smoothing filter restarts at that point.
- Tracking always runs at the input's full resolution, even with `--preview`.
