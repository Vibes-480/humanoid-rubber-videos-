# hairify

Replace the person in a short video with a CGI creature made of rubbery, jiggling
hair/spike strands, in the spirit of Method Studios' "AICP 2016 Sponsor Reel".

Five creature styles: rubber **strand** hair, **fiber**-optic neon, **circuit**-board skin, a
voxel **swarm** of cubes and a **vowels** swarm of letters. The sci-fi styles react to drum hits /
beats detected in the soundtrack.

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

# other creatures
.venv/bin/python hairify.py input.mp4 --creature fiber
.venv/bin/python hairify.py input.mp4 --creature circuit --color "#39ff88"
.venv/bin/python hairify.py input.mp4 --creature swarm --stiffness 0.3
.venv/bin/python hairify.py input.mp4 --creature vowels --grade none

# fast iteration: first 5 s at half resolution
.venv/bin/python hairify.py input.mp4 --preview

# debug videos for each stage
.venv/bin/python hairify.py input.mp4 --preview --stage skeleton
.venv/bin/python hairify.py input.mp4 --preview --stage background
```

The output has the same resolution, fps and length as the input, with the original audio
(H.264, CRF 18, preset veryfast, yuv420p; AAC/MP3 audio is copied untouched). Pose tracking,
segmentation and the background plate are cached in `.hairify_cache/`, so re-rendering the
same clip with different look settings skips the slow parts. Delete that folder (or pass
`--no-cache`) to start over.

## Options

| Option | Default | What it does |
|---|---|---|
| `input` | | Input video: one person, ideally full body, static camera. |
| `-o, --output` | `<input>_hairy.mp4` | Output MP4 path. |
| `--creature` | `strand` | `strand`, `fiber`, `circuit`, `swarm` or `vowels` (see below). |
| `--color` | per creature | Base colour (hex). |
| `--grade` | `auto` | Background grade: `none`, `cyber` (dark teal/violet, vignette). `auto` = cyber for fiber, circuit and vowels. |
| `--hair-length` | `1.0` | Strand length (strand, fiber), trace length (circuit), cube/letter size (swarm, vowels). |
| `--stiffness` | per creature | 0 = floppy hair / cooked spaghetti, 1 = stiff spikes (strand default 0.15, fiber 0.45). For swarm/vowels: how tightly particles follow the body (default 0.5). |
| `--density` | `1.0` | Count multiplier: strands, traces or particles. |
| `--preview` | off | Render only the first 5 s at half resolution. |
| `--keep-person` | off | Skip person removal; draw the creature on top (debugging). |
| `--stage` | `final` | `skeleton` = pose debug overlay, `background` = person removed, `final` = creature. |
| `--model` | `full` | Pose model: `lite` (fastest), `full`, `heavy` (most accurate). |
| `--no-cache` | off | Re-run pose tracking even if a cached result exists. |

All other tunables (smoothing, strand/segment counts, gravity, damping, colours, widths,
encoder settings) are in the `CONFIG` dict at the top of `hairify.py`.

## Creatures

| `--creature` | Default colour | What it is |
|---|---|---|
| `strand` | `#ff3fa4` | Rubbery strands on the body capsules that hang, swing and flop like hair or cooked spaghetti (raise `--stiffness` for wobbly spikes). Glossy shading (dark roots, light tips, specular highlight). The body itself is invisible: its shape is implied by where the strands grow (set `core_shrink` > 0 for a fill under the strands). |
| `fiber` | `#00e5ff` | Same physics; dark fibers with glowing, colour-shifting tips and light pulses running root→tip on every drum hit. Bloom. |
| `circuit` | `#19f0ff` | Shaded dark body covered in PCB traces, vias and chips; light pulses run along the traces, in bursts on drum hits. |
| `swarm` | `#ff3fa4` | Thousands of shaded cubes sprung to points inside the body. Fast moves make them trail; drum hits blast them away from the striking hand. |
| `vowels` | `#ffd23f` | The same swarm made of the letters A E I O U, flickering between vowels. |

How the body works: capsules along the skeleton (torso, neck, head, arms, hands, legs, feet),
sized from the body scale W (≈ shoulder width, smoothed). Each strand root / trace / particle keeps
a fixed position on its capsule, so it follows the body. Body parts the tracker reports as
hidden (e.g. legs behind a table) are clipped to the person's mask, so real objects in front of the
person stay in front of the creature. Physics runs at a fixed 120 Hz with substeps, independent of
the video fps.

Strand physics: Verlet integration at a fixed 120 Hz; the root is pinned to the skin, the first
points are held in the skin direction (follicle), every segment gently continues the previous one
(bending stiffness, keeps curves smooth), gravity pulls the rest, segment lengths are enforced
with dynamic follow-the-leader (Müller et al. 2012) so chains swing freely without exploding.
`--stiffness` blends in a pull toward the straight rest pose for spikes.

Drum hits come from spectral-flux onset detection on the audio track (`CONFIG["audio"]`).

## Skeleton debug video (`--stage skeleton`)

- grey dots: raw MediaPipe landmarks; coloured skeleton: One Euro–smoothed pose
  (orange = left side, blue = right side)
- joint colour: green = visible, red = occluded; joint size grows as the joint gets closer to the camera (z)
- top-left label: `DETECTED`, `INTERPOLATED` (short gap, filled), or `LOST` (creature faded out)

## Person removal (`--stage background`)

1. **Segmentation**: MediaPipe `selfie_multiclass_256x256` gives a person probability mask per
   frame. Only the frames that are rendered plus the plate samples are segmented, and the
   results are cached per frame, so a first `--preview` on a new clip stays quick.
2. **Camera motion**: ORB features on the background (person masked out) are matched to frame 0
   and a similarity transform is fitted with RANSAC. If the 95th-percentile shift is under
   `static_threshold_px` the camera is treated as static; otherwise the plate is aligned to
   every frame, and frames that cannot be aligned fall back to inpainting.
3. **Background plate**: per-pixel median over `plate_samples` frames, ignoring masked pixels.
   Pixels seen in less than `plate_min_coverage` of the samples (e.g. behind someone who never
   moves) are inpainted once, so the fill is stable over time. It is saved as `<output>_plate.png`.
4. **Compositing**: the mask is maxed over ±`temporal_dilate` frames (covers motion blur),
   dilated, feathered, and the plate is blended in, with a per-channel gain that tracks the
   camera's auto-exposure.

### Inpainters

`CONFIG["removal"]["inpainter"]` picks a function from the `INPAINTERS` dict:

| Name | Notes |
|---|---|
| `rows` (default) | Blends each row between the colours left and right of the hole, then smooths vertically. Best for person-shaped holes; does not smear things below the person (floor, tables) upwards. |
| `pyramid` | OpenCV Telea at low resolution, upsampled. Smooth, but radial streaks on large holes. |
| `telea` | OpenCV Telea at full resolution. |

To plug in a better one (e.g. LaMa), add a function `fn(bgr_uint8, mask_uint8, radius) -> bgr_uint8`
to `INPAINTERS` and select it in the config.

## How tracking handles dropouts

- Gaps shorter than `max_interp_gap_s` (0.5 s) are linearly interpolated.
- Longer gaps hold the last pose and fade the creature out over `fade_s`; it fades back in
  when the person is detected again, and the smoothing filter restarts at that point.
- Tracking always runs at the input's full resolution, even with `--preview`.
