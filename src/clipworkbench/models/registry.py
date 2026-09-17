"""Model version registry: track, list, and load saved models.

Wraps the SQLite Storage to provide a clean API for model lifecycle:
  - register: Save metadata after training completes
  - list: List all registered models
  - get: Retrieve a specific model's metadata
  - delete: Remove a model from the registry
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from clipworkbench.core.schema import FineTuneStrategy, ModelInfo
from clipworkbench.data.storage import Storage


class ModelRegistry:
    """Manages model metadata and file lifecycle."""

    def __init__(
        self,
        models_dir: str = "./saved_models",
        db_path: str = "./clipworkbench.db",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(db_path)

    def register(
        self,
        path: str,
        strategy: FineTuneStrategy,
        class_names: list[str],
        metrics: dict[str, float] | None = None,
        model_id: str | None = None,
    ) -> ModelInfo:
        """Register a newly trained model in the registry."""
        model_id = model_id or str(uuid.uuid4())[:8]
        num_classes = len(class_names)
        self.storage.save_model_info(
            model_id=model_id,
            path=path,
            strategy=strategy.value,
            num_classes=num_classes,
            class_names=class_names,
            metrics=metrics or {},
        )
        return ModelInfo(
            model_id=model_id,
            path=path,
            strategy=strategy,
            num_classes=num_classes,
            class_names=class_names,
            metrics=metrics or {},
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models, newest first."""
        return self.storage.list_models()

    def get(self, model_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific model."""
        return self.storage.get_model_info(model_id)

    def delete(self, model_id: str) -> bool:
        """Delete a model from the registry and remove its files."""
        info = self.storage.get_model_info(model_id)
        if info is None:
            return False
        model_path = Path(info["path"])
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)
        self.storage.delete_model_info(model_id)
        return True

    def resolve_path(self, model_id: str) -> str | None:
        """Get the filesystem path for a model by ID."""
        info = self.get(model_id)
        return info["path"] if info else None
