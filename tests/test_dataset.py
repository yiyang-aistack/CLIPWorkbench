"""Tests for the dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipworkbench.core.exceptions import DatasetError
from clipworkbench.data.dataset import (
    ClipImageDataset,
    DatasetStats,
    compute_class_weights,
    stratified_split,
)


def _create_fake_dataset(tmp_path: Path, num_classes: int = 3, samples_per_class: int = 10) -> Path:
    """Create a minimal ImageFolder structure in tmp_path."""
    for i in range(num_classes):
        class_dir = tmp_path / f"class_{i}"
        class_dir.mkdir()
        for j in range(samples_per_class):
            # Create a minimal 1x1 pixel image
            (class_dir / f"img_{j}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    return tmp_path


def test_dataset_raises_on_missing_path() -> None:
    with pytest.raises(DatasetError):
        ClipImageDataset("/nonexistent/path/to/dataset")


def test_dataset_raises_on_empty_dir(tmp_path: Path) -> None:
    # Empty directory with no class folders
    with pytest.raises(DatasetError):
        ClipImageDataset(tmp_path)


def test_stratified_split(tmp_path: Path) -> None:
    ds_path = _create_fake_dataset(tmp_path, num_classes=3, samples_per_class=10)
    # Create dataset without processor (uses fallback)
    dataset = ClipImageDataset(ds_path)
    train, val = stratified_split(dataset, val_split=0.2)
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == len(dataset)


def test_class_counts(tmp_path: Path) -> None:
    ds_path = _create_fake_dataset(tmp_path, num_classes=3, samples_per_class=10)
    dataset = ClipImageDataset(ds_path)
    counts = dataset.get_class_counts()
    assert len(counts) == 3
    assert all(v == 10 for v in counts.values())


def test_compute_class_weights() -> None:
    counts = {"a": 10, "b": 20, "c": 30}
    weights = compute_class_weights(counts)
    assert len(weights) == 3
    # Class with fewer samples should get higher weight
    assert weights[0] > weights[1] > weights[2]


def test_dataset_stats(tmp_path: Path) -> None:
    ds_path = _create_fake_dataset(tmp_path, num_classes=3, samples_per_class=10)
    dataset = ClipImageDataset(ds_path)
    stats = dataset.get_stats()
    assert stats.num_classes == 3
    assert stats.num_samples == 30
    assert len(stats.class_names) == 3
