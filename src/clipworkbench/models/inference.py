# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""Inference engine: zero-shot and fine-tuned CLIP prediction.

Supports three modes:
  1. Zero-shot: text prompts only (no training needed)
  2. Linear probe: frozen CLIP + trained linear head
  3. LoRA: merged CLIP + trained linear head

All modes support temperature scaling and Top-K output.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from clipworkbench.core.config import get_settings
from clipworkbench.core.exceptions import ModelNotFoundError
from clipworkbench.core.schema import FineTuneStrategy, PredictionItem, PredictionResult
from clipworkbench.models.linear_probe import LinearProbeHead


# Model cache (avoid reloading on every API call)
_model_cache: dict[str, tuple[Any, Any, dict]] = {}


def load_finetuned_model(model_path: str, device: str | None = None) -> tuple[Any, Any, dict]:
    """Load a fine-tuned CLIP model + processor + class info.

    Uses an in-memory cache so repeated API calls don't reload.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    if not Path(model_path).exists():
        raise ModelNotFoundError(f"Model path not found: {model_path}")

    from transformers import CLIPModel, CLIPProcessor
    import torch

    settings = get_settings()
    device = device or settings.resolved_device

    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)

    # Load class info
    class_info_path = Path(model_path) / "class_info.json"
    if not class_info_path.exists():
        raise ModelNotFoundError(f"class_info.json not found in {model_path}")
    with open(class_info_path, encoding="utf-8") as f:
        class_info = json.load(f)

    class_names: list[str] = class_info["class_names"]
    num_classes = len(class_names)

    # Rebuild and load classification head
    feature_dim = model.config.projection_dim
    head = LinearProbeHead(feature_dim, num_classes)
    head_path = Path(model_path) / "classification_head.pt"
    if head_path.exists():
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.to(device)
    model.to(device)
    model.eval()

    result = (model, processor, {"head": head, "class_names": class_names, "device": device})
    _model_cache[model_path] = result
    return result


def predict_finetuned(
    model: Any,
    processor: Any,
    head: LinearProbeHead,
    image: Image.Image,
    class_names: list[str],
    device: str = "cpu",
    temperature: float = 1.0,
    top_k: int = 5,
) -> PredictionResult:
    """Run fine-tuned classification on a single image."""
    import torch

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(vision_out.pooler_output)
        logits = head(features)
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    # Sort by probability
    ranked = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    top_k = min(top_k, len(ranked))
    items = [PredictionItem(label=name, probability=round(float(p), 4)) for name, p in ranked[:top_k]]

    return PredictionResult(
        top_k=items,
        best_label=items[0].label,
        best_probability=items[0].probability,
        strategy=FineTuneStrategy.linear_probe,
        latency_ms=round(latency_ms, 2),
    )


def predict_zero_shot(
    model: Any,
    processor: Any,
    image: Image.Image,
    candidate_labels: list[str],
    device: str = "cpu",
    top_k: int = 5,
    template: str = "a photo of {}",
) -> PredictionResult:
    """Run zero-shot CLIP classification using text prompts.

    Args:
        model: CLIPModel.
        processor: CLIPProcessor.
        image: PIL Image.
        candidate_labels: List of class name strings.
        device: "cuda" or "cpu".
        top_k: Number of top results to return.
        template: Prompt template, e.g. "a photo of a {}".
    """
    import torch

    text_prompts = [template.format(label) for label in candidate_labels]

    start = time.perf_counter()
    inputs = processor(
        text=text_prompts, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_texts)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    ranked = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)
    top_k = min(top_k, len(ranked))
    items = [PredictionItem(label=name, probability=round(float(p), 4)) for name, p in ranked[:top_k]]

    return PredictionResult(
        top_k=items,
        best_label=items[0].label,
        best_probability=items[0].probability,
        strategy=FineTuneStrategy.zero_shot,
        latency_ms=round(latency_ms, 2),
    )


def predict_image(
    image: Image.Image,
    model_path: str | None = None,
    candidate_labels: list[str] | None = None,
    strategy: FineTuneStrategy = FineTuneStrategy.linear_probe,
    top_k: int = 5,
    temperature: float = 1.0,
    device: str | None = None,
) -> PredictionResult:
    """Unified prediction entry point.

    - If strategy == zero_shot: requires candidate_labels, uses base CLIP.
    - If strategy == linear_probe or lora: requires model_path, uses fine-tuned model.
    """
    import torch

    settings = get_settings()
    device = device or settings.resolved_device

    if strategy == FineTuneStrategy.zero_shot:
        from clipworkbench.models.clip_loader import load_clip_model
        model, processor = load_clip_model(device=device)
        if not candidate_labels:
            raise ValueError("Zero-shot prediction requires candidate_labels")
        return predict_zero_shot(model, processor, image, candidate_labels, device, top_k)

    else:
        if not model_path:
            raise ValueError("Fine-tuned prediction requires model_path")
        model, processor, info = load_finetuned_model(model_path, device)
        return predict_finetuned(
            model=model,
            processor=processor,
            head=info["head"],
            image=image,
            class_names=info["class_names"],
            device=info["device"],
            temperature=temperature,
            top_k=top_k,
        )