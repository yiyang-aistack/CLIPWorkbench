#!/usr/bin/env bash
# =============================================================================
#  CLIPWorkbench - FastAPI service (start_api.sh)
#
#  HTTP shell of the shared core: predict / embed / finetune / status /
#  models / export / tasks.  Same as:
#      clipworkbench serve --host 0.0.0.0 --port 8000
#
#  Usage :  ./start_api.sh [extra clipworkbench serve arguments]
#  Env   :  CLIPWORKBENCH_HOST      bind address        (default: 0.0.0.0)
#           CLIPWORKBENCH_PORT      TCP port            (default: 8000)
#           CLIPWORKBENCH_RELOAD    auto-reload on code changes (any value except 0)
#           CLIPWORKBENCH_EXTRA     uv extra to sync    (default: onnx,
#                              empty = core deps only, no ONNX export)
#           DEVICE             cuda | cpu          (empty = auto-detect)
#
#  Run "chmod +x start_*.sh" once after cloning so the scripts are executable.
#  The script always runs from the repository root, so configs/default.yaml
#  and the ./model_cache, ./saved_models, ./onnx_models folders are found.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXTRA="${CLIPWORKBENCH_EXTRA-onnx}"
HOST="${CLIPWORKBENCH_HOST-0.0.0.0}"
PORT="${CLIPWORKBENCH_PORT-8000}"
RELOAD="${CLIPWORKBENCH_RELOAD-}"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

extra_args=()
if [ -n "$EXTRA" ]; then
  extra_args=(--extra "$EXTRA")
fi

# Anything but empty / 0 turns auto-reload on; reload_display keeps the banner
# in sync with the arguments that are really forwarded.
reload_args=()
reload_display=""
if [ -n "$RELOAD" ] && [ "$RELOAD" != "0" ]; then
  reload_args=(--reload)
  reload_display="--reload"
fi

serve_args=(--host "$HOST" --port "$PORT")

# ---- 1. preferred path: uv (creates / refreshes .venv from uv.lock) --------
if command -v uv >/dev/null 2>&1; then
  echo "[clipworkbench] API:  uv run ${EXTRA:+--extra "$EXTRA" }clipworkbench serve --host $HOST --port $PORT ${reload_display}"
  echo "[clipworkbench] health http://localhost:$PORT/health   docs http://localhost:$PORT/docs"
  exec uv run ${extra_args[@]+"${extra_args[@]}"} clipworkbench serve "${serve_args[@]}" \
    ${reload_args[@]+"${reload_args[@]}"} "$@"
fi

# ---- 2. interpreter of the local .venv ------------------------------------
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import fastapi, uvicorn, clipworkbench" >/dev/null 2>&1; then
  echo "[clipworkbench] API:  .venv/bin/python -m clipworkbench.cli serve --host $HOST --port $PORT ${reload_display}"
  echo "[clipworkbench] health http://localhost:$PORT/health   docs http://localhost:$PORT/docs"
  exec "$VENV_PY" -m clipworkbench.cli serve "${serve_args[@]}" \
    ${reload_args[@]+"${reload_args[@]}"} "$@"
fi

# ---- 3. whichever Python is on PATH ---------------------------------------
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
    "$candidate" -c "import fastapi, uvicorn, clipworkbench" >/dev/null 2>&1; then
    echo "[clipworkbench] API:  $candidate -m clipworkbench.cli serve --host $HOST --port $PORT ${reload_display}"
    echo "[clipworkbench] health http://localhost:$PORT/health   docs http://localhost:$PORT/docs"
    exec "$candidate" -m clipworkbench.cli serve "${serve_args[@]}" \
      ${reload_args[@]+"${reload_args[@]}"} "$@"
  fi
done

cat >&2 <<'EOF'

[clipworkbench] ERROR: no Python environment providing FastAPI was found.

  uv sync --all-extras        (recommended: builds .venv from uv.lock)
  pip install -e ".[onnx]"    (plain pip alternative)

  Install uv: https://docs.astral.sh/uv/getting-started/installation/
EOF
exit 1
