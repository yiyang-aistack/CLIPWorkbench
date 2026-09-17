"""Pydantic data models (schema) shared across CLI, API, and GUI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FineTuneStrategy(str, Enum):
    """Fine-tuning strategies, ordered by data/compute requirement."""
    zero_shot = "zero_shot"
    linear_probe = "linear_probe"
    lora = "lora"


class PredictionItem(BaseModel):
    """A single (label, probability) pair in a Top-K prediction."""
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Full prediction output for one image."""
    top_k: list[PredictionItem]
    best_label: str
    best_probability: float
    strategy: FineTuneStrategy
    latency_ms: float = 0.0


class EpochHistory(BaseModel):
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float = 0.0
    val_f1: float = 0.0


class TrainStatus(BaseModel):
    """Training task status tracker (used by API background tasks)."""
    task_id: str
    status: str = "queued"  # queued | preparing | training | completed | failed
    message: str = ""
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    strategy: FineTuneStrategy = FineTuneStrategy.linear_probe
    train_history: list[EpochHistory] = Field(default_factory=list)
    val_history: list[EpochHistory] = Field(default_factory=list)
    dataset_stats: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class ModelInfo(BaseModel):
    """Metadata for a saved model version."""
    model_id: str
    path: str
    strategy: FineTuneStrategy
    num_classes: int
    class_names: list[str]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict[str, float] = Field(default_factory=dict)


class ExportResult(BaseModel):
    """Result of ONNX export + optional quantization."""
    onnx_path: str
    quantized_path: str | None = None
    pytorch_size_mb: float = 0.0
    onnx_size_mb: float = 0.0
    quantized_size_mb: float | None = None
    quantization: str | None = None  # fp16 | int8 | None
