# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""Application configuration via pydantic-settings + YAML.

Usage:
    from clipworkbench.core.config import get_settings
    settings = get_settings()
    print(settings.model.name)  # openai/clip-vit-base-patch32
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """CLIP model configuration."""
    name: str = "openai/clip-vit-base-patch32"
    local_cache_dir: str = "./model_cache"
    image_size: int = 224


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 20
    logging_steps: int = 10
    val_split: float = 0.15
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    use_early_stopping: bool = True
    patience: int = 5
    delta: float = 1e-4
    # Linear probe
    linear_probe_epochs: int = 10
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    # Inference
    inference_temperature: float = 1.0
    top_k: int = 5


class StorageConfig(BaseModel):
    """Storage paths."""
    models_dir: str = "./saved_models"
    onnx_dir: str = "./onnx_models"
    db_path: str = "./clipworkbench.db"


class Settings(BaseSettings):
    """Top-level application settings."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    device: str = ""  # auto-detect if empty

    @property
    def resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def save_yaml(self, path: str) -> None:
        """Write current settings to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def _load_yaml_config(yaml_path: str | None = None) -> dict:
    """Load a YAML config file, returning an empty dict if not found."""
    if yaml_path is None:
        # Look for configs/default.yaml relative to project root
        candidates = [
            Path("configs/default.yaml"),
            Path("config.yaml"),
        ]
        for c in candidates:
            if c.exists():
                yaml_path = str(c)
                break
    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings(yaml_path: str | None = None) -> Settings:
    """Get cached Settings instance, optionally from a YAML file."""
    data = _load_yaml_config(yaml_path)
    settings = Settings(**data) if data else Settings()
    # Ensure storage directories exist
    for d in [settings.storage.models_dir, settings.storage.onnx_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return settings