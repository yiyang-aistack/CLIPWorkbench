#!/usr/bin/env bash
# =============================================================================
#  CLIPWorkbench - Application desktop GUI (start_app.sh)
#
#  End-user side: single-image inference, Top-K probabilities, batch folder
#  inference.  Same as:  clipworkbench gui --mode app
#
#  Usage :  ./start_app.sh [extra clipworkbench gui arguments]
#  Env   :  CLIPWORKBENCH_EXTRA    uv extra to sync (default: gui, empty = no extra)
#
#  Run "chmod +x start_*.sh" once after cloning so the scripts are executable.
#  The script always runs from the repository root, so configs/default.yaml
#  and the ./model_cache, ./saved_models, ./onnx_models folders are found.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="app"
EXTRA="${CLIPWORKBENCH_EXTRA-gui}"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

extra_args=()
if [ -n "$EXTRA" ]; then
  extra_args=(--extra "$EXTRA")
fi

# ---- 1. preferred path: uv (creates / refreshes .venv from uv.lock) --------
if command -v uv >/dev/null 2>&1; then
  echo "[clipworkbench] app GUI:  uv run ${EXTRA:+--extra "$EXTRA" }clipworkbench gui --mode $MODE"
  exec uv run ${extra_args[@]+"${extra_args[@]}"} clipworkbench gui --mode "$MODE" "$@"
fi

# ---- 2. interpreter of the local .venv ------------------------------------
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import PySide6, clipworkbench" >/dev/null 2>&1; then
  echo "[clipworkbench] app GUI:  .venv/bin/python -m clipworkbench.cli gui --mode $MODE"
  exec "$VENV_PY" -m clipworkbench.cli gui --mode "$MODE" "$@"
fi

# ---- 3. whichever Python is on PATH ---------------------------------------
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
    "$candidate" -c "import PySide6, clipworkbench" >/dev/null 2>&1; then
    echo "[clipworkbench] app GUI:  $candidate -m clipworkbench.cli gui --mode $MODE"
    exec "$candidate" -m clipworkbench.cli gui --mode "$MODE" "$@"
  fi
done

cat >&2 <<'EOF'

[clipworkbench] ERROR: no Python environment providing PySide6 was found.

  uv sync --all-extras        (recommended: builds .venv from uv.lock)
  pip install -e ".[gui]"     (plain pip alternative)

  Install uv: https://docs.astral.sh/uv/getting-started/installation/
EOF
exit 1
