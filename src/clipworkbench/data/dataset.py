"""Image dataset utilities for CLIP fine-tuning.

Supported dataset formats:

1. ImageFolder (directory)
    dataset_root/
    ├── class_a/
    │   ├── img001.jpg
    │   └── img002.png
    ├── class_b/
    │   └── img003.jpg
    └── ...
    Each top-level folder name is treated as the class label.

2. CSV annotation file (.csv)
    image_path,label
    ./images/cat_001.jpg,cat
    ./images/cat_002.jpg,cat
    ./images/dog_001.jpg,dog
    ...
    Optional 3rd column ``split`` can be used for predefined train/val split.

Use :func:`load_dataset` to auto-detect the format from the given path.
Both dataset classes share the same public API (``class_names``, ``samples``,
``class_counts``, ``__getitem__``), so downstream training code is format-agnostic.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset, Subset

from clipworkbench.core.exceptions import DatasetError

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class DatasetStats:
    """Summary statistics for a dataset."""
    num_classes: int = 0
    num_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    class_names: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "num_samples": self.num_samples,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "class_names": self.class_names,
            "class_counts": self.class_counts,
        }


class ClipImageDataset(Dataset):
    """Image dataset that loads from a folder structure.

    Args:
        root_dir: Path to the dataset root containing class folders.
        processor: A HuggingFace CLIPProcessor for image preprocessing.
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        processor: Any | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise DatasetError(f"Dataset root not found: {root_dir}")

        self.processor = processor

        # Discover class folders (sorted for reproducibility)
        self.class_names: list[str] = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not self.class_names:
            raise DatasetError(
                f"No class folders found in {root_dir}. "
                f"Expected subfolders named after classes."
            )
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # Gather samples
        self.samples: list[tuple[str, int]] = []
        self.class_counts: dict[str, int] = {}
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            count = 0
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((str(class_dir / img_name), self.class_to_idx[class_name]))
                    count += 1
            self.class_counts[class_name] = count

        if len(self.samples) == 0:
            raise DatasetError(
                f"No images found in {root_dir}. "
                f"Supported extensions: {IMG_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)  # remove batch dim
            return {"pixel_values": pixel_values, "label": label, "image_path": img_path}
        else:
            import torch
            import numpy as np
            arr = np.array(image.resize((224, 224)))
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return {"pixel_values": tensor, "label": label, "image_path": img_path}

    def get_class_counts(self) -> dict[str, int]:
        """Return per-class sample counts (for class-weight computation)."""
        return dict(self.class_counts)

    def get_stats(self) -> DatasetStats:
        return DatasetStats(
            num_classes=len(self.class_names),
            num_samples=len(self.samples),
            class_names=list(self.class_names),
            class_counts=dict(self.class_counts),
        )


class ClipCsvDataset(Dataset):
    """Image dataset loaded from a CSV annotation file.

    Expected CSV columns (header required)::

        image_path,label[,split]

    - ``image_path`` -- absolute path, or relative to the CSV file's directory.
    - ``label``      -- class name (string).
    - ``split``      -- optional: ``"train"`` / ``"val"``.
      When every row carries a split, callers can bypass auto ``stratified_split``.

    The class exposes the **exact same public attributes** as
    :class:`ClipImageDataset` (``class_names``, ``class_to_idx``, ``samples``,
    ``class_counts``, ``__getitem__`` return dict), so downstream training code
    does not need to know which format is being used.
    """

    def __init__(
        self,
        csv_path: str | os.PathLike,
        processor: Any | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise DatasetError(f"CSV not found: {csv_path}")
        if self.csv_path.suffix.lower() != ".csv":
            raise DatasetError(f"Expected .csv file, got: {csv_path}")

        self.processor = processor
        # 相对 image_path 以此目录为基准
        self._base_dir = self.csv_path.parent

        # 1) 读取 CSV
        rows: list[dict[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"image_path", "label"}
            fieldnames = set(reader.fieldnames or [])
            if not required.issubset(fieldnames):
                raise DatasetError(
                    f"CSV must have columns {required}. Got: {reader.fieldnames}"
                )
            for row in reader:
                rows.append(row)

        if not rows:
            raise DatasetError(f"CSV is empty: {csv_path}")

        # 2) 构建类别（与 ImageFolder 保持一致：字母排序、0 起始索引）
        self.class_names: list[str] = sorted({r["label"].strip() for r in rows})
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # 3) 构建 samples + class_counts + split 记录
        self.samples: list[tuple[str, int]] = []
        self.class_counts: dict[str, int] = {n: 0 for n in self.class_names}
        self._splits: list[str | None] = []

        for r in rows:
            img_path = r["image_path"].strip()
            label = r["label"].strip()

            p = Path(img_path)
            if not p.is_absolute():
                p = self._base_dir / p

            self.samples.append((str(p), self.class_to_idx[label]))
            self.class_counts[label] += 1
            sp = (r.get("split") or "").strip().lower()
            self._splits.append(sp if sp in {"train", "val"} else None)

        if len(self.samples) == 0:
            raise DatasetError(
                f"No valid rows in CSV: {csv_path}"
            )

        # 4) 校验图片文件存在（尽早报出，避免训练中途崩溃）
        missing = [(i, p) for i, (p, _) in enumerate(self.samples) if not Path(p).is_file()]
        if missing:
            preview = "; ".join(f"{i}:{p}" for i, p in missing[:3])
            raise DatasetError(
                f"{len(missing)} image path(s) not found. E.g.: {preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return {"pixel_values": pixel_values, "label": label, "image_path": img_path}

        import numpy as np
        import torch
        arr = np.array(image.resize((224, 224)))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return {"pixel_values": tensor, "label": label, "image_path": img_path}

    def get_class_counts(self) -> dict[str, int]:
        return dict(self.class_counts)

    def get_stats(self) -> DatasetStats:
        return DatasetStats(
            num_classes=len(self.class_names),
            num_samples=len(self.samples),
            class_names=list(self.class_names),
            class_counts=dict(self.class_counts),
        )

    def has_predefined_split(self) -> bool:
        """Whether the CSV provides a per-row ``split`` column.

        When True, callers may skip :func:`stratified_split` and use
        :meth:`split_by_predefined` instead.
        """
        return any(s is not None for s in self._splits)

    def split_by_predefined(self) -> tuple[Subset, Subset]:
        """Split using the ``split`` column in the CSV.

        Rows with anything other than ``train`` / ``val`` are skipped.
        Raises :class:`DatasetError` when no split information is available.
        """
        if not self.has_predefined_split():
            raise DatasetError("CSV has no predefined split column")
        train_idx = [i for i, s in enumerate(self._splits) if s == "train"]
        val_idx = [i for i, s in enumerate(self._splits) if s == "val"]
        if not train_idx:
            raise DatasetError("Predefined split produced empty train set")
        if not val_idx:
            # 允许没有 val（回退到全部 train）
            val_idx = []
        return Subset(self, train_idx), Subset(self, val_idx)


def load_dataset(
    path: str | os.PathLike,
    processor: Any | None = None,
) -> "ClipImageDataset | ClipCsvDataset":
    """Auto-detect dataset format from *path* and instantiate the right class.

    - directory      → :class:`ClipImageDataset`
    - ``.csv`` file  → :class:`ClipCsvDataset`

    Raises :class:`DatasetError` for anything else.
    """
    p = Path(path)
    if p.is_dir():
        return ClipImageDataset(p, processor=processor)
    if p.is_file() and p.suffix.lower() == ".csv":
        return ClipCsvDataset(p, processor=processor)
    raise DatasetError(
        f"Unsupported dataset path: {path}. "
        f"Expected a directory (ImageFolder) or a .csv file."
    )


def stratified_split(
    dataset: "ClipImageDataset | ClipCsvDataset",
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Stratified train/validation split.

    Ensures each class has proportional representation in both splits.
    Guarantees at least 1 sample in train and 1 in val when a class has >=2 samples.
    """
    rng = random.Random(seed)

    # Group indices by class
    class_groups: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        label = dataset.samples[idx][1]
        class_groups.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for label, indices in class_groups.items():
        n = len(indices)
        rng.shuffle(indices)
        val_size = max(1, int(n * val_split)) if n >= 2 else 0
        train_size = n - val_size
        if train_size < 1 and n >= 1:
            # All to train if only 1 sample
            val_size = 0
            train_size = n
        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def compute_class_weights(class_counts: dict[str, int]) -> "list[float]":
    """Inverse-frequency class weights to handle imbalance.

    weight_i = total / (num_classes * count_i)
    """
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = []
    for _, count in sorted(class_counts.items()):
        count = max(count, 1)
        weights.append(total / (num_classes * count))
    return weights