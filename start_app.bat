@echo off
REM ============================================================================
REM  CLIPWorkbench - Application desktop GUI (start_app.bat)
REM
REM  End-user side: single-image inference, Top-K probabilities, batch folder
REM  inference.  Same as:  clipworkbench gui --mode app
REM
REM  Usage :  start_app.bat [extra clipworkbench gui arguments]
REM  Env   :  CLIPWORKBENCH_EXTRA    uv extra to sync (default: gui, empty = no extra)
REM           CLIPWORKBENCH_NOPAUSE  1 = never wait for a key press after a failure
REM
REM  The script always runs from the repository root, so configs/default.yaml
REM  and the ./model_cache, ./saved_models, ./onnx_models folders are found.
REM ============================================================================

setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "MODE=app"

if not defined CLIPWORKBENCH_EXTRA set "CLIPWORKBENCH_EXTRA=gui"
set "EXTRA_ARG="
if not "%CLIPWORKBENCH_EXTRA%"=="" set "EXTRA_ARG=--extra %CLIPWORKBENCH_EXTRA%"

REM ---- run from the project root (configs/, saved_models/, ... live here) ----
pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
    echo [clipworkbench] ERROR: cannot enter "%SCRIPT_DIR%"
    exit /b 1
)
set "PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%"

REM ---- 1. preferred path: uv (creates / refreshes .venv from uv.lock) -------
where uv >nul 2>nul
if not errorlevel 1 goto :run_uv

REM ---- 2. interpreter of the local .venv ------------------------------------
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto :try_system
"%VENV_PY%" -c "import PySide6, clipworkbench" >nul 2>nul
if not errorlevel 1 goto :run_venv

REM ---- 3. whichever Python is on PATH ---------------------------------------
:try_system
where python >nul 2>nul
if errorlevel 1 goto :no_env
python -c "import PySide6, clipworkbench" >nul 2>nul
if not errorlevel 1 goto :run_python

:no_env
echo.
echo [clipworkbench] ERROR: no Python environment providing PySide6 was found.
echo.
echo   uv sync --all-extras        ^(recommended: builds .venv from uv.lock^)
echo   pip install -e ".[gui]"     ^(plain pip alternative^)
echo.
echo   Install uv: https://docs.astral.sh/uv/getting-started/installation/
echo.
set "RC=1"
goto :done

:run_uv
echo [clipworkbench] app GUI:  uv run %EXTRA_ARG% clipworkbench gui --mode %MODE%
uv run %EXTRA_ARG% clipworkbench gui --mode %MODE% %*
set "RC=%ERRORLEVEL%"
goto :done

:run_venv
echo [clipworkbench] app GUI:  .venv\Scripts\python.exe -m clipworkbench.cli gui --mode %MODE%
"%VENV_PY%" -m clipworkbench.cli gui --mode %MODE% %*
set "RC=%ERRORLEVEL%"
goto :done

:run_python
echo [clipworkbench] app GUI:  python -m clipworkbench.cli gui --mode %MODE%
python -m clipworkbench.cli gui --mode %MODE% %*
set "RC=%ERRORLEVEL%"
goto :done

:done
if not "%RC%"=="0" (
    echo.
    echo [clipworkbench] app GUI exited with code %RC%
    if not defined CLIPWORKBENCH_NOPAUSE pause
)
popd >nul
endlocal & exit /b %RC%
