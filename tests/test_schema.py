"""Tests for the core schema module."""

from __future__ import annotations

from clipworkbench.core.schema import (
    EpochHistory,
    ExportResult,
    FineTuneStrategy,
    ModelInfo,
    PredictionItem,
    PredictionResult,
    TrainStatus,
)


def test_prediction_item() -> None:
    item = PredictionItem(label="cat", probability=0.95)
    assert item.label == "cat"
    assert item.probability == 0.95


def test_prediction_result() -> None:
    result = PredictionResult(
        top_k=[PredictionItem(label="cat", probability=0.9)],
        best_label="cat",
        best_probability=0.9,
        strategy=FineTuneStrategy.zero_shot,
    )
    assert result.best_label == "cat"
    assert len(result.top_k) == 1


def test_train_status() -> None:
    status = TrainStatus(task_id="test-123")
    assert status.task_id == "test-123"
    assert status.status == "queued"
    assert status.train_history == []

    status.status = "training"
    assert status.status == "training"


def test_model_info() -> None:
    info = ModelInfo(
        model_id="abc",
        path="/models/abc",
        strategy=FineTuneStrategy.linear_probe,
        num_classes=5,
        class_names=["a", "b", "c", "d", "e"],
    )
    assert info.model_id == "abc"
    assert info.num_classes == 5


def test_export_result() -> None:
    result = ExportResult(
        onnx_path="/models/model.onnx",
        onnx_size_mb=150.0,
        pytorch_size_mb=300.0,
    )
    assert result.onnx_path == "/models/model.onnx"
    assert result.quantized_path is None


def test_epoch_history() -> None:
    hist = EpochHistory(epoch=1, train_loss=0.5, val_loss=0.6)
    assert hist.epoch == 1
    assert hist.train_loss == 0.5


def test_finetune_strategy_enum() -> None:
    assert FineTuneStrategy.zero_shot.value == "zero_shot"
    assert FineTuneStrategy.linear_probe.value == "linear_probe"
    assert FineTuneStrategy.lora.value == "lora"
