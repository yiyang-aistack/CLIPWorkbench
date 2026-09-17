# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""CLIP model and processor loading with local caching.

Supports HuggingFace transformers CLIP models (default).
The model is downloaded once and cached locally for offline use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clipworkbench.core.config import get_settings
from clipworkbench.core.exceptions import ModelNotFoundError


def load_clip_model(
    model_name: str | None = None,
    cache_dir: str | None = None,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load a CLIP model and processor.

    Tries local cache first, then downloads from HuggingFace Hub.
    After download, saves to cache_dir for future offline use.

    Args:
        model_name: HuggingFace model ID (e.g. "openai/clip-vit-base-patch32").
        cache_dir: Local directory to cache the model.
        device: "cuda" or "cpu". Auto-detects if None.

    Returns:
        (model, processor) tuple.
    """
    from transformers import CLIPModel, CLIPProcessor

    settings = get_settings()
    model_name = model_name or settings.model.name
    cache_dir = cache_dir or settings.model.local_cache_dir
    device = device or settings.resolved_device

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    local_path = Path(cache_dir) / model_name.replace("/", "_")

    if local_path.exists():
        model = CLIPModel.from_pretrained(str(local_path))
        processor = CLIPProcessor.from_pretrained(str(local_path))
    else:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.save_pretrained(str(local_path))
        processor.save_pretrained(str(local_path))

    model.to(device)
    return model, processor


def get_image_embeddings(
    model: Any,
    processor: Any,
    images: list[Any],
    device: str = "cpu",
) -> Any:
    """Compute image embeddings (projection_dim) for a list of PIL images.

    Returns a tensor of shape (len(images), projection_dim).
    """
    import torch

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        embeddings = model.visual_projection(outputs.pooler_output)
    return embeddings


def get_text_embeddings(
    model: Any,
    processor: Any,
    texts: list[str],
    device: str = "cpu",
) -> Any:
    """Compute text embeddings (projection_dim) for a list of text prompts.

    Returns a tensor of shape (len(texts), projection_dim).
    """
    import torch

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embeddings = model.text_projection(outputs.pooler_output)
    return embeddings