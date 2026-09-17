# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""Linear probe fine-tuning: freeze CLIP backbone, train a linear head.

This is the recommended starting point when you have a small labeled dataset
(10-100 images per class). It's fast, stable, and avoids overfitting.

Architecture:
    Image → CLIP vision encoder → visual_projection → linear_head → logits
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clipworkbench.core.config import get_settings
from clipworkbench.core.schema import FineTuneStrategy
from clipworkbench.data.dataset import (
    ClipImageDataset,
    compute_class_weights,
    stratified_split,
)


class LinearProbeHead(nn.Module):
    """Single linear layer on top of CLIP image embeddings."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def collate_fn(batch: list[dict], processor: Any) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader — stacks pixel_values and labels."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def train_linear_probe(
    model: Any,
    processor: Any,
    dataset: ClipImageDataset,
    save_path: str,
    device: str | None = None,
    config_override: dict | None = None,
    status_callback: Any | None = None,
) -> dict[str, Any]:
    """Train a linear probe head on top of a frozen CLIP model.

    Args:
        model: A HuggingFace CLIPModel.
        processor: A HuggingFace CLIPProcessor.
        dataset: A ClipImageDataset with folder-based labels.
        save_path: Directory to save model weights + class info.
        device: Override device ("cuda" or "cpu").
        config_override: Dict of training config overrides.
        status_callback: Optional callable(epoch, metrics) for progress reporting.

    Returns:
        Dict with training history and final metrics.
    """
    settings = get_settings()
    cfg = settings.train
    if config_override:
        for k, v in config_override.items():
            setattr(cfg, k, v)
    device = device or settings.resolved_device

    # Freeze the entire CLIP backbone
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Create linear probe head
    feature_dim = model.config.projection_dim
    num_classes = len(dataset.class_names)
    head = LinearProbeHead(feature_dim, num_classes).to(device)

    # Stratified split
    train_subset, val_subset = stratified_split(dataset, cfg.val_split)
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    # Class weights for imbalanced data
    weights = compute_class_weights(dataset.get_class_counts())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer only trains the head
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # LR scheduler
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp and device == "cuda" else None

    train_history: list[dict] = []
    val_history: list[dict] = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(cfg.num_epochs):
        # --- Training ---
        head.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        vision_out = model.vision_model(pixel_values=pixel_values)
                        features = model.visual_projection(vision_out.pooler_output)
                    logits = head(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    vision_out = model.vision_model(pixel_values=pixel_values)
                    features = model.visual_projection(vision_out.pooler_output)
                logits = head(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                vision_out = model.vision_model(pixel_values=pixel_values)
                features = model.visual_projection(vision_out.pooler_output)
                logits = head(features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / total if total > 0 else 0.0

        # F1 (macro)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0

        epoch_hist = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        train_history.append(epoch_hist)
        val_history.append(epoch_hist)

        if status_callback:
            status_callback(epoch + 1, cfg.num_epochs, epoch_hist)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

    # Save best model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if best_state:
        head.load_state_dict(best_state)

    # Save: CLIP backbone + processor + head + class info
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    torch.save(head.state_dict(), os.path.join(save_path, "classification_head.pt"))

    class_info = {"class_names": dataset.class_names}
    with open(os.path.join(save_path, "class_info.json"), "w", encoding="utf-8") as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)

    return {
        "strategy": FineTuneStrategy.linear_probe.value,
        "save_path": save_path,
        "best_val_accuracy": best_val_acc,
        "train_history": train_history,
        "val_history": val_history,
        "dataset_stats": dataset.get_stats().to_dict(),
    }