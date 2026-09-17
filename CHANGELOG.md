# Changelog

All notable changes to CLIPWorkbench are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-16

### Added

- Zero-shot CLIP classification, linear-probe and LoRA fine-tuning, ONNX export with FP16/INT8
  quantization, a SQLite-backed model registry, and stratified train/val splits with class weighting.
- FastAPI service (`predict` · `embed` · `finetune` · `status` · `models` · `export` · `tasks`).
- PySide6 desktop windows: admin (dataset, training, registry, export) and app (inference, Top-K, batch).
- `clipworkbench` CLI with `train` · `infer` · `export` · `serve` · `gui`, available both as the console
  script and as `python -m clipworkbench`.
- One-command launchers in the repository root — `start_admin.*`, `start_app.*`, `start_api.*`
  (`.bat` for Windows, `.sh` for Linux/macOS).

### Changed

- **Renamed** the project, distribution, import package, CLI, environment variables, registry database and
  exception base class before the first release. The old names are gone, not aliased — no compatibility
  shim is shipped. Superseded → current:

| Old                                       | New                  |
|-------------------------------------------|----------------------|
| `clip_desktop_lab` (import package)       | `clipworkbench`      |
| `clip-desktop-lab` (PyPI distribution)    | `clipworkbench`      |
| `ClipDesktopLab` (display name)           | `CLIPWorkbench`      |
| `clip-lab` (console script)               | `clipworkbench`      |
| `CLIP_LAB_*` (launcher env variables)     | `CLIPWORKBENCH_*`    |
| `clip_lab.db` (SQLite registry)           | `clipworkbench.db`   |
| `ClipLabError` (exception base class)     | `ClipWorkbenchError` |

### Migration

- Imports: `from clip_desktop_lab...` → `from clipworkbench...`.
- Commands: `clip-lab ...` / `python -m clip_desktop_lab ...` → `clipworkbench ...` /
  `python -m clipworkbench ...`; launcher variables `CLIP_LAB_*` → `CLIPWORKBENCH_*`.
- Registry database: rename `clip_lab.db` → `clipworkbench.db`. The table names (`models` ·
  `predictions` · `corrections`) are unchanged, so the data keeps working; only absolute paths stored
  inside may point at an older checkout.
- Saved fine-tuned models and ONNX exports need no changes — no package name is persisted inside them.
- Recreate the environment so the renamed console script is materialised: `uv sync --all-extras`
  (delete `.venv` first for a guaranteed clean editable install).
