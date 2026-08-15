# Motion Study Runbook

Everything needed to go from a camera on a tripod to a report a supervisor will
accept. Read section 1 before filming anything — camera discipline decides the
accuracy of this system more than any setting in it.

---

## 1. How to film

The zones you draw are **pixel coordinates**. They stay valid only while the
camera does not move. Everything else follows from that.

**Rules**

| Rule | Why it matters |
|---|---|
| **Fixed tripod. Never handheld.** | A moving camera invalidates every zone, and every number downstream is then wrong without looking wrong. |
| **Do not move, pan or zoom mid-recording.** | Same reason. If you must reposition, stop recording and start a new file — one file per camera position. |
| **Get the whole operation in frame**, including the aisle workers walk through. | Anything outside the frame is reported as "left", which inflates idle time and starts arguments. |
| **Mount high — a ceiling corner or a mezzanine beats eye level.** | Steel stock, jigs and racks hide workers at eye level. Occlusion is the single biggest accuracy limit. |
| **Keep the station well lit; avoid shooting into a bright doorway.** | A backlit worker becomes a silhouette the detector misses. |
| **Record at least 5–6 full cycles.** | Fewer than 5 and the spread statistics mean nothing. |
| **1080p at 25–30 fps is plenty.** | 4K costs processing time and buys nothing — we analyse at 2 fps. |
| **Tell the team they are being filmed and why.** | It is process improvement, not performance policing. It also gets far better cooperation, and footage of identifiable people deserves that basic courtesy. |

**One camera position = one config file.** If you film the cutting bench from
two angles, that is two config files.

---

## 2. First-time setup

Already done on this machine, but for a new one:

```bash
winget install Python.Python.3.12 --architecture x64
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -e .
```

Model weights (~40 MB) download automatically on the first run and are cached.

---

## 3. Running a study

### Step 1 — Drop the video in `input\`

### Step 2 — Draw the stations (once per camera position)

```bash
python -m mstudy zones input\your_video.mp4 -c configs\your_line.yaml
```

A window opens on a frame from the video. Click to place points, **ENTER** to
close a zone and name it, **S** to save and quit.

Draw one zone per station (cutting bench, press, welding jig, material rack,
finished stack) plus one for the aisle, marked as `walkway`. Zones are tested in
order, so **list a specific station before any broad aisle polygon that overlaps
it**.

Then open the config and fill in:
- `steps:` — friendly work-element names your IE team already uses
- `cycles.boundary_zone:` — the station where a new cycle begins, usually where
  the worker picks up the next piece
- `time_study.rating_factor:` — see section 5

### Step 3 — Check the setup before committing an hour

```bash
python -m mstudy track input\your_video.mp4 -c configs\your_line.yaml --quick
```

Processes the first 2 minutes only. Confirm workers are being detected before
running the full job.

### Step 4 — The full run

```bash
python -m mstudy run input\your_video.mp4 -c configs\your_line.yaml
```

This tracks, asks you to name the workers, and writes the report. It is
resumable — if it is interrupted, run the same command again and it picks up
from the last checkpoint.

### Step 5 — Read `runs\<video-name>\report.html`

Optional, and worth it for the first few videos:

```bash
python -m mstudy annotate input\your_video.mp4 -c configs\your_line.yaml --max-seconds 120
```

Two minutes of your own footage with the labels burned in. Watch it. If the
labels are right there, the numbers are right everywhere.

### Unattended batch

```bash
python -m mstudy watch -c configs\your_line.yaml
```

Watches `input\` and processes anything new. Leave it running overnight.

---

## 4. How long it takes

Measured on this machine (Snapdragon X, x64 emulation, CPU only, `lightweight`
models, 1280×720 footage):

| | Measured |
|---|---|
| Detection | ~0.55 s per analysed frame |
| Pose | ~0.12 s per worker per pose frame |
| **Realistic throughput** | **roughly 1.5–2× the length of the video** |

So a **30-minute video takes about 45–60 minutes**, and a day's filming is an
overnight job. This is a CPU-only laptop running emulated x64 — that is the
cost, and it is why the pipeline is resumable and has a `watch` mode.

**If throughput becomes the bottleneck**, a desktop with an NVIDIA GPU cuts this
roughly tenfold with **no code change**: install `onnxruntime-gpu` and set
`detection.device: cuda` in the config.

**To go faster on this machine**, in order of what actually helps:
1. `video.analysis_fps: 1.0` — halves the work; still 1 s resolution
2. `video.pose_fps: 0.5` — cheaper, but slower to notice a stop starting
3. Film at 720p rather than 1080p — less decoding

---

## 5. Reading the report

**The Gantt chart.** One row per worker across the operation's real clock.
Colour = employee. Texture = activity: solid is working at a station, hatched is
walking, faded is away. **■ amber square = stopped**, **✖ red cross = left**.
Hover anything for exact times. The second tab recolours by work element.

**Work element times.** Read the **`cv_percent`** column first — spread as a
percentage of the mean. A slow step is usually already known about; an
*inconsistent* step is where recoverable time hides.

**Where the time went.** Utilisation is time at a work station as a share of
time on camera, not of the whole recording, so a worker who appears late is not
penalised.

**Yamazumi.** Work content per operator per cycle. Uneven columns mean the line
is unbalanced — the tallest column sets the pace and everyone else waits.

**Spaghetti diagram.** Movement paths. Dense tangles are walking that the layout
is forcing.

### Standard time, and the one thing this system will not do

```
Normal time   = Observed time × Rating factor
Standard time = Normal time × (1 + Allowances)
```

**The rating factor is a human judgement and this system will not invent one.**
It cannot see whether a worker was going flat out or coasting. You enter it in
the config and the report always prints the value used. Any tool that claims to
auto-generate a performance rating from video is guessing, and the first
supervisor who challenges it will be right to.

Set `rating_factor` from a qualified observer's assessment before using any
standard time to set a target.

---

## 6. Validating before anyone trusts it

**Do this once per camera position, before setting any standard from this
system.**

1. Pick one video. Watch it with a stopwatch and hand-time **3 complete cycles**,
   element by element.
2. Open `runs\<video>\segments.csv` and pull the same elements.
3. Compare.

**Acceptance gate: within ±5% on element durations.** If it misses, tune — do
not lower the bar:

| Symptom | Fix |
|---|---|
| Real work reported as stopped | Lower `events.stop_motion_threshold` |
| Idle workers reported as working | Raise `events.stop_motion_threshold` |
| Steps fragmenting into many short bars | Raise `events.min_segment_seconds` |
| Departures reported that did not happen | Raise `events.missing_grace_seconds` |
| Workers missed entirely | `detection.mode: balanced`, lower `min_confidence` |
| Time in the wrong station | Redraw the zone; check station order in the config |

After any change:

```bash
python -m mstudy report input\your_video.mp4 -c configs\your_line.yaml
```

That rebuilds the report from the saved tracks in seconds — it does **not**
reprocess the video.

---

## 7. Troubleshooting

**"No workers were detected."** Zones may be wrong, or the detector is missing
people. Run `annotate` and look. Try `detection.mode: balanced`.

**Too many tracks on the contact sheet.** Normal — the tracker gives a new
number each time someone is re-acquired. Give the same name to every thumbnail
of the same person; they merge onto one Gantt row.

**A visitor or supervisor walked through.** Type `x` at their track to exclude
them.

**"Could not open video."** Re-encode it:

```bash
ffmpeg -i input\your_video.mp4 -c:v libx264 -an input\fixed.mp4
```

**The camera got bumped mid-recording.** Split the file at that point and treat
the halves as two camera positions with two configs. The zones cannot be right
for both.

---

## 8. Privacy

Footage of identifiable workers stays on this machine. `input\` and `runs\` are
git-ignored, so no video and no analysis output is ever committed. Nothing is
uploaded anywhere — all processing is local.
