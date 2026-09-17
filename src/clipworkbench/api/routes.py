# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""API route definitions.

Endpoints:
    POST /api/v1/predict      — Classify an image (zero-shot or fine-tuned)
                                model_path optional → newest registry model used.
    POST /api/v1/embed        — Get image embeddings (base CLIP or fine-tuned)
                                empty model_path = base CLIP.
    POST /api/v1/finetune     — Start a background training task
    GET  /api/v1/status/{id}  — Query training task status
    GET  /api/v1/models       — List all registered models
    GET  /api/v1/models/default — Return newest registered model
    POST /api/v1/export       — Export a model to ONNX (model_path optional)
    GET  /api/v1/tasks        — List all training tasks
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from clipworkbench.core.config import get_settings
from clipworkbench.core.schema import FineTuneStrategy
from clipworkbench.core.task_queue import task_queue
from clipworkbench.data.dataset import load_dataset
from clipworkbench.data.storage import Storage
from clipworkbench.models.clip_loader import load_clip_model
from clipworkbench.models.inference import predict_image
from clipworkbench.models.onnx_export import export_to_onnx, benchmark_onnx
from clipworkbench.models.registry import ModelRegistry

router = APIRouter()

# Shared storage/registry instances
_storage: Storage | None = None
_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        settings = get_settings()
        _registry = ModelRegistry(
            models_dir=settings.storage.models_dir,
            db_path=settings.storage.db_path,
        )
    return _registry


def _resolve_model_path(model_path: str | None) -> str:
    """Return an explicit model_path, or the newest registered model as fallback.

    Raises HTTPException(404) when no model is available at all.
    """
    if model_path:  # truthy → caller explicitly provided one
        return model_path
    models = _get_registry().list_models()
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No fine-tuned model registered. Train one first or pass model_path explicitly.",
        )
    return models[0]["path"]


# --- Request/Response models ---

class ZeroShotRequest(BaseModel):
    candidate_labels: list[str]
    template: str = "a photo of {}"


class FinetuneRequest(BaseModel):
    dataset_path: str
    strategy: str = "linear_probe"  # linear_probe | lora
    model_save_path: str = ""
    config_override: dict[str, Any] = {}


class ExportRequest(BaseModel):
    model_path: str = ""  # empty → use newest registered model
    quantization: str | None = None  # fp16 | int8 | None


# --- Endpoints ---

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_path: str = Form(""),
    candidate_labels: str = Form(""),  # comma-separated
    strategy: str = Form("linear_probe"),
    top_k: int = Form(5),
    temperature: float = Form(1.0),
):
    """Classify an uploaded image.

    - For zero-shot: provide candidate_labels (comma-separated), leave model_path empty.
    - For fine-tuned: provide model_path, leave candidate_labels empty.
    """
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    labels = [l.strip() for l in candidate_labels.split(",")] if candidate_labels else None
    strat = FineTuneStrategy(strategy)

    resolved_path = _resolve_model_path(model_path) if strat != FineTuneStrategy.zero_shot else None

    result = predict_image(
        image=image,
        model_path=resolved_path,
        candidate_labels=labels,
        strategy=strat,
        top_k=top_k,
        temperature=temperature,
    )

    # Log to storage
    reg = _get_registry()
    reg.storage.log_prediction(
        image_path=file.filename or "upload",
        model_id=resolved_path or "zero_shot",
        top_label=result.best_label,
        probability=result.best_probability,
        full_result=result.model_dump(),
    )

    return result.model_dump()


@router.post("/embed")
async def embed(
    file: UploadFile = File(...),
    model_path: str = Form(""),
):
    """Get image embedding (projection_dim vector)."""
    from clipworkbench.models.inference import load_finetuned_model

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    settings = get_settings()
    device = settings.resolved_device

    resolved_path = _resolve_model_path(model_path) if model_path or True else None

    if model_path or resolved_path:
        model, processor, _ = load_finetuned_model(resolved_path, device)
    else:
        model, processor = load_clip_model(device=device)

    from clipworkbench.models.clip_loader import get_image_embeddings
    emb = get_image_embeddings(model, processor, [image], device)
    return {"embedding": emb[0].cpu().tolist(), "dim": emb.shape[1]}


@router.post("/finetune")
async def finetune(
    request: FinetuneRequest,
    background_tasks: BackgroundTasks,
):
    """Start a background training task."""
    status = task_queue.create()
    status.strategy = FineTuneStrategy(request.strategy)
    task_queue.update(
        status.task_id,
        status="queued",
        message="Training task queued",
    )

    background_tasks.add_task(_run_training, status.task_id, request)
    return {"task_id": status.task_id, "message": "Training started. Use /status/{task_id} to track."}


def _run_training(task_id: str, request: FinetuneRequest) -> None:
    """Background training function (called by FastAPI BackgroundTasks)."""
    import traceback
    from datetime import datetime

    status = task_queue.get(task_id)
    if status is None:
        return

    try:
        settings = get_settings()
        device = settings.resolved_device

        task_queue.update(
            task_id,
            status="preparing",
            message="Loading model...",
            start_time=datetime.now().isoformat(),
        )

        model, processor = load_clip_model(device=device)
        dataset = load_dataset(request.dataset_path, processor)
        stats = dataset.get_stats()

        task_queue.update(
            task_id,
            status="training",
            message=f"Training {request.strategy} on {stats.num_samples} images, {stats.num_classes} classes",
            total_epochs=settings.train.num_epochs,
            dataset_stats=stats.to_dict(),
        )

        save_path = request.model_save_path or str(
            Path(settings.storage.models_dir) / f"model_{task_id[:8]}"
        )

        def callback(epoch: int, total: int, metrics: dict) -> None:
            task_queue.update(
                task_id,
                current_epoch=epoch,
                total_epochs=total,
                progress=epoch / total,
                message=f"Epoch {epoch}/{total} | Val Acc: {metrics.get('val_accuracy', 0):.4f}",
            )
            # Append to history
            status = task_queue.get(task_id)
            if status:
                hist_entry = {
                    "epoch": epoch,
                    "train_loss": metrics.get("train_loss", 0),
                    "val_loss": metrics.get("val_loss", 0),
                    "val_accuracy": metrics.get("val_accuracy", 0),
                    "val_f1": metrics.get("val_f1", 0),
                }
                status.train_history.append(hist_entry)
                status.val_history.append(hist_entry)

        if request.strategy == "lora":
            from clipworkbench.models.lora_finetune import train_lora
            result = train_lora(
                model, processor, dataset, save_path, device,
                request.config_override or None, callback,
            )
        else:
            from clipworkbench.models.linear_probe import train_linear_probe
            result = train_linear_probe(
                model, processor, dataset, save_path, device,
                request.config_override or None, callback,
            )

        # Register model
        reg = _get_registry()
        reg.register(
            path=save_path,
            strategy=FineTuneStrategy(request.strategy),
            class_names=dataset.class_names,
            metrics={"val_accuracy": result["best_val_accuracy"]},
        )

        task_queue.update(
            task_id,
            status="completed",
            message=f"Training completed. Val accuracy: {result['best_val_accuracy']:.4f}",
            progress=1.0,
            end_time=datetime.now().isoformat(),
            result={"save_path": save_path, "best_val_accuracy": result["best_val_accuracy"]},
        )

    except Exception as e:
        traceback.print_exc()
        task_queue.update(
            task_id,
            status="failed",
            message=f"Training failed: {e}",
            error=str(e),
            end_time=datetime.now().isoformat(),
        )


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Query training task status."""
    status = task_queue.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status.model_dump()


@router.get("/tasks")
async def list_tasks():
    """List all training tasks."""
    return {"tasks": [t.model_dump() for t in task_queue.list_all()]}


@router.get("/models")
async def list_models():
    """List all registered models (newest first)."""
    reg = _get_registry()
    return {"models": reg.list_models()}


@router.get("/models/default")
async def get_default_model():
    """Return the newest registered model (the one used when callers omit model_path)."""
    models = _get_registry().list_models()
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No fine-tuned model registered.",
        )
    return {"default": models[0]}


@router.post("/export")
async def export_model(request: ExportRequest):
    """Export a model to ONNX with optional quantization.

    model_path is optional: when omitted, the newest registered model is exported.
    """
    try:
        resolved = _resolve_model_path(request.model_path)
        result = export_to_onnx(resolved, quantization=request.quantization)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark")
async def benchmark(
    onnx_path: str = Form(...),
    num_runs: int = Form(100),
):
    """Benchmark ONNX Runtime inference latency."""
    try:
        result = benchmark_onnx(onnx_path, num_runs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))