# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""FastAPI application factory.

Usage:
    from clipworkbench.api.main import create_app
    app = create_app()
    # Or run directly: uvicorn clipworkbench.api.main:app
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from clipworkbench.api.routes import router


def _print_startup_banner() -> None:
    """Print a human-readable summary of runtime config + model registry."""
    from clipworkbench.core.config import get_settings
    from clipworkbench.models.registry import ModelRegistry

    settings = get_settings()
    device = settings.resolved_device
    db_path = settings.storage.db_path
    models_dir = settings.storage.models_dir

    try:
        models = ModelRegistry().list_models()
    except Exception as exc:  # noqa: BLE001
        models = []
        print(f"  (registry unreadable: {exc})")

    print()
    print("=" * 60)
    print("  CLIPWorkbench API  v0.1.0")
    print("-" * 60)
    print(f"  Device:       {device}")
    print(f"  Registry DB:  {db_path}")
    print(f"  Models dir:   {models_dir}")
    print(f"  Base CLIP:    {settings.model.name}")
    print("-" * 60)

    if not models:
        print("  Registered models: 0")
        print()
        print("  ⚠ No fine-tuned models registered.")
        print("     Fine-tuned /predict and /export will 404 until one is trained.")
        print("     Use Admin GUI (start_admin.bat) or POST /api/v1/finetune")
        print("     Zero-shot CLIP is always available via strategy=zero_shot")
    else:
        print(f"  Registered models: {len(models)}")
        # Find the newest (= the default used when callers omit model_path)
        newest = models[0]
        for i, m in enumerate(models):
            marker = " ★ DEFAULT" if i == 0 else "          "
            mid = m.get("model_id", "?")
            strat = m.get("strategy", "?")
            ncls = m.get("num_classes", "?")
            path = m.get("path", "?")
            created = m.get("created_at", "")[:19]
            print(f"    {marker} {mid}  {strat:<14} ({ncls} cls)  [{created}]")
            print(f"             path: {path}")

        print("-" * 60)
        print(f"  ★ Default model (used when model_path omitted):")
        print(f"      {newest.get('model_id', '?')}  {newest.get('strategy', '?')}  "
              f"({newest.get('num_classes', '?')} cls)")
        print(f"      path: {newest.get('path', '?')}")

    print("-" * 60)
    print("  Zero-shot CLIP: always available (strategy=zero_shot)")
    print("=" * 60)
    print()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_startup_banner()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CLIPWorkbench API",
        description=(
            "Multimodal AI inference and training API. "
            "Zero-shot CLIP, linear probe, and LoRA fine-tuning."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health() -> dict:
        from clipworkbench.core.config import get_settings
        settings = get_settings()
        return {
            "status": "healthy",
            "device": settings.resolved_device,
            "registry_count": _registry_count(),
            "default_model_id": _default_model_id(),
        }

    return app


def _registry_count() -> int:
    """Return the number of registered models (best-effort, never raises)."""
    try:
        from clipworkbench.models.registry import ModelRegistry
        return len(ModelRegistry().list_models())
    except Exception:  # noqa: BLE001
        return 0


def _default_model_id() -> str | None:
    """Return the default (newest) model ID, or None when no models."""
    try:
        from clipworkbench.models.registry import ModelRegistry
        models = ModelRegistry().list_models()
        return models[0]["model_id"] if models else None
    except Exception:  # noqa: BLE001
        return None


# Module-level ASGI instance for `uvicorn clipworkbench.api.main:app`
app = create_app()