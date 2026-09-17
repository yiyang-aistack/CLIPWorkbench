@echo off
REM ============================================================================
REM  CLIPWorkbench - FastAPI service (start_api.bat)
REM
REM  HTTP shell of the shared core: predict / embed / finetune / status /
REM  models / export / tasks.  Same as:
REM      clipworkbench serve --host 0.0.0.0 --port 8000
REM
REM  Usage :  start_api.bat [extra clipworkbench serve arguments]
REM  Env   :  CLIPWORKBENCH_HOST      bind address        (default: 0.0.0.0)
REM           CLIPWORKBENCH_PORT      TCP port            (default: 8000)
REM           CLIPWORKBENCH_RELOAD    auto-reload on code changes (any value except 0)
REM           CLIPWORKBENCH_EXTRA     uv extra to sync   (default: onnx,
REM                              empty = core deps only, no ONNX export)
REM           CLIPWORKBENCH_NOPAUSE   1 = never wait for a key press after a failure
REM           DEVICE             cuda | cpu          (empty = auto-detect)
REM
REM  The script always runs from the repository root, so configs/default.yaml
REM  and the ./model_cache, ./saved_models, ./onnx_models folders are found.
REM ============================================================================

setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"

if not defined CLIPWORKBENCH_EXTRA set "CLIPWORKBENCH_EXTRA=onnx"
if not defined CLIPWORKBENCH_HOST set "CLIPWORKBENCH_HOST=0.0.0.0"
if not defined CLIPWORKBENCH_PORT set "CLIPWORKBENCH_PORT=8000"
if not defined CLIPWORKBENCH_RELOAD set "CLIPWORKBENCH_RELOAD=0"

set "EXTRA_ARG="
if not "%CLIPWORKBENCH_EXTRA%"=="" set "EXTRA_ARG=--extra %CLIPWORKBENCH_EXTRA%"
REM anything but empty / 0 turns auto-reload on
set "RELOAD_ARG="
if not "%CLIPWORKBENCH_RELOAD%"=="" if not "%CLIPWORKBENCH_RELOAD%"=="0" set "RELOAD_ARG=--reload"

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
"%VENV_PY%" -c "import fastapi, uvicorn, clipworkbench" >nul 2>nul
if not errorlevel 1 goto :run_venv

REM ---- 3. whichever Python is on PATH ---------------------------------------
:try_system
where python >nul 2>nul
if errorlevel 1 goto :no_env
python -c "import fastapi, uvicorn, clipworkbench" >nul 2>nul
if not errorlevel 1 goto :run_python

:no_env
echo.
echo [clipworkbench] ERROR: no Python environment providing FastAPI was found.
echo.
echo   uv sync --all-extras        ^(recommended: builds .venv from uv.lock^)
echo   pip install -e ".[onnx]"    ^(plain pip alternative^)
echo.
echo   Install uv: https://docs.astral.sh/uv/getting-started/installation/
echo.
set "RC=1"
goto :done

REM ---- kill any lingering clipworkbench/uvicorn process (Windows .exe lock) ----
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r ":%CLIPWORKBENCH_PORT% " ^| findstr LISTENING') do (
    echo [clipworkbench] killing stale PID %%p on port %CLIPWORKBENCH_PORT%
    taskkill /F /PID %%p >nul 2>nul
)
taskkill /F /IM clipworkbench.exe >nul 2>nul

:run_uv
echo [clipworkbench] API:  uv run %EXTRA_ARG% clipworkbench serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG%
echo [clipworkbench] health http://localhost:%CLIPWORKBENCH_PORT%/health   docs http://localhost:%CLIPWORKBENCH_PORT%/docs
uv run %EXTRA_ARG% clipworkbench serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG% %*
set "RC=%ERRORLEVEL%"
goto :done

:run_venv
echo [clipworkbench] API:  .venv\Scripts\python.exe -m clipworkbench.cli serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG%
echo [clipworkbench] health http://localhost:%CLIPWORKBENCH_PORT%/health   docs http://localhost:%CLIPWORKBENCH_PORT%/docs
"%VENV_PY%" -m clipworkbench.cli serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG% %*
set "RC=%ERRORLEVEL%"
goto :done

:run_python
echo [clipworkbench] API:  python -m clipworkbench.cli serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG%
echo [clipworkbench] health http://localhost:%CLIPWORKBENCH_PORT%/health   docs http://localhost:%CLIPWORKBENCH_PORT%/docs
python -m clipworkbench.cli serve --host %CLIPWORKBENCH_HOST% --port %CLIPWORKBENCH_PORT% %RELOAD_ARG% %*
set "RC=%ERRORLEVEL%"
goto :done

:done
if not "%RC%"=="0" (
    echo.
    echo [clipworkbench] API exited with code %RC%
    if not defined CLIPWORKBENCH_NOPAUSE pause
)
popd >nul
endlocal & exit /b %RC%