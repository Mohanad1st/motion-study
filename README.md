# motion-study

Video-based **motion and time study** for steel scaffolding manufacturing.

Drop a video of a production operation into `input\`, run one command, get an
interactive report: per-worker processing times, work elements across the
operation timeline, detected stops and departures, cycle times, labour balance
and movement paths.

```bash
python -m mstudy run input\cutting_line.mp4 -c configs\cutting_line.yaml
```

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for how to film, run and validate.

---

## See it before filming anything

```bash
python -m mstudy demo
```

Builds a full report from a synthetic three-worker cuplock-ledger line. No
video, no models, no GPU. Opens at `runs\demo\report.html`.

---

## What it produces

Per video, in `runs\<video-name>\`:

| File | What it is |
|---|---|
| `report.html` | The deliverable. Self-contained, opens offline. |
| `segments.csv` | Every worker/activity/start/end/duration — pivot this in Excel. |
| `stops.csv`, `cycles.csv`, `step_statistics.csv`, `worker_statistics.csv` | Raw tables. |
| `events.jsonl` | Machine-readable event stream. |
| `annotated.mp4` | The video with names, steps and a clock burned in (optional). |
| `tracks.parquet` | Raw detections — re-run the analysis without re-processing the video. |
| `run.json` | Config fingerprint and versions, so any report can be reproduced. |

The report contains a multi-worker **Gantt chart** (colour per employee,
**■** = stopped, **✖** = left), work-element statistics with spread, a
**Yamazumi** labour-balance chart, and a **spaghetti** movement diagram.

---

## How it works

```
video ─► detect (YOLOX) ─► track (ByteTrack) ─► name workers
                                                     │
              zones you drew once ──────────────────►┤
                                                     ▼
                              steps · stops · departures · cycles
                                                     ▼
                                    report.html + annotated.mp4
```

A worker's **step** is the station zone they are standing in. A **stop** is
hand and forearm movement below a threshold while at a station — measured in
body-heights per second, so it works the same near and far from the camera. A
**departure** is being outside every zone, or out of frame.

Each stage writes a file the next one reads, so changing a threshold or a chart
rebuilds in seconds instead of reprocessing the video.

---

## Licensing

Everything is **MIT / Apache-2.0 / BSD**. Nothing here is AGPL, and there is no
PyTorch dependency.

| Component | Licence |
|---|---|
| [rtmlib](https://github.com/Tau-J/rtmlib) — YOLOX detection + RTMPose | Apache-2.0 |
| [supervision](https://github.com/roboflow/supervision) | MIT |
| [trackers](https://github.com/roboflow/trackers) — ByteTrack | Apache-2.0 |
| ONNX Runtime · OpenCV · Plotly · pandas · Jinja2 | MIT / Apache-2.0 / BSD |

Ultralytics YOLO was deliberately **not** used: it is AGPL-3.0, which would
restrict what this system could become later.

---

## Development

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

49 tests, no video or models required. `tests/test_demo_accuracy.py` checks
detected stops, departures, element times and cycle times against synthetic data
with known ground truth.

---

## The one thing this system will not do

It will not estimate a worker's **performance rating**. It cannot see whether
someone was going flat out or coasting. You supply the rating factor; the report
always prints the value used. Standard times are only as defensible as that
human judgement behind them.
