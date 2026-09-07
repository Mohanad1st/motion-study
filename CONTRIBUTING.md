# Contributing

## What this is

A command-line tool that turns video of a production operation into an interactive motion and time
study. It is used to measure processes, and it deliberately refuses to rate people — see the README
for why that boundary exists. **Contributions that erode it will be declined**, however well
implemented: no performance rating, no effort estimation, no per-person productivity scoring, no
leaderboards.

That is the only closed door. Everything else is open, and there is plenty to do.

## Especially welcome

- **Detection and tracking accuracy on real footage.** This is the biggest gap in the project. The
  analysis half is validated against synthetic ground truth (`tests/test_demo_accuracy.py`); the
  detection half has never been measured against a hand-labelled video. A benchmark, a labelled
  clip you can share, or a reproducible accuracy number would be the single most useful
  contribution.
- **Other domains.** The demo models a scaffolding line because that is what it was built for. If
  you run it on a different kind of assembly or fabrication work and it does something wrong, that
  is worth an issue with the config you used.
- **Speed.** Analysis runs at roughly 1.5–2× video length on CPU. `onnxruntime-gpu` with
  `detection.device: cuda` is a config change away, but nobody has published measured numbers.
- **Report clarity.** The report is the deliverable, and it has to survive being shown to a
  supervisor who has never seen a Yamazumi chart. Confusing labels and misleading defaults are bugs.
- **Platform fixes.** Developed on Windows, tested in CI on Linux. macOS is unverified.

## Setting up

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests -q                  # 71 tests, no video or model weights needed
```

`python -m mstudy demo` builds a complete report from synthetic data, so you can work on the
analysis, the report or the charts without any footage at all. Most changes can be developed and
reviewed this way.

On Windows-on-ARM, use an x64 Python — `opencv-python` publishes no Windows-ARM64 wheel.

## Before opening a PR

```bash
pytest tests -q
```

CI runs the same on Python 3.11 and 3.12. Two things to keep true:

- **Never commit footage, a report, or model weights.** `.gitignore` covers `input/`, `runs/`,
  video extensions, `models/` and `*.onnx`. If you add an output path, add it there in the same
  commit.
- **Declare new dependencies in `pyproject.toml`.** A package that is merely installed in your venv
  will pass locally and break every clean install — that exact bug shipped once, which is why CI now
  installs from scratch on every push.

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.

## Reporting something sensitive

Do not open a public issue for a security or privacy problem. See [SECURITY.md](SECURITY.md) — and
note that the privacy boundary around footage matters more here than any conventional vulnerability.

## Licence

By contributing you agree your contribution is licensed under the Apache License 2.0 in
[LICENSE](LICENSE).
