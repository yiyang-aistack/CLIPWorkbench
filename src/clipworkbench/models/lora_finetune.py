# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""LoRA/PEFT fine-tuning for CLIP.

When you have more data (50+ images per class) and want to adapt
the CLIP backbone itself, LoRA injects low-rank adapters into
attention layers (q_proj, v_proj) and trains them alongside the
classification head.

Training phases:
  Phase 1 (warmup): Train classification head only (frozen backbone + LoRA).
  Phase 2 (joint): Unfreeze LoRA params + head, joint optimization.
"""

from __future__ import annotations

import json
import os
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
from clipworkbench.models.linear_probe import LinearProbeHead, collate_fn


def train_lora(
    model: Any,
    processor: Any,
    dataset: ClipImageDataset,
    save_path: str,
    device: str | None = None,
    config_override: dict | None = None,
    status_callback: Any | None = None,
) -> dict[str, Any]:
    """Fine-tune CLIP with LoRA adapters + a linear classification head.

    Args:
        model: A HuggingFace CLIPModel.
        processor: A HuggingFace CLIPProcessor.
        dataset: A ClipImageDataset with folder-based labels.
        save_path: Directory to save the merged model + class info.
        device: Override device.
        config_override: Dict of training config overrides.
        status_callback: Optional callable(epoch, total, metrics).

    Returns:
        Dict with training history and final metrics.
    """
    from peft import LoraConfig, get_peft_model, PeftModel

    settings = get_settings()
    cfg = settings.train
    if config_override:
        for k, v in config_override.items():
            setattr(cfg, k, v)
    device = device or settings.resolved_device

    # Phase split: first half trains head only, second half joint
    warmup_epochs = max(1, cfg.num_epochs // 2)

    # 1. Freeze all backbone params
    for param in model.parameters():
        param.requires_grad = False

    # 2. Inject LoRA into vision encoder
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    model.vision_model = get_peft_model(model.vision_model, lora_config)
    model.vision_model.print_trainable_parameters()

    # 3. Create classification head
    feature_dim = model.config.projection_dim
    num_classes = len(dataset.class_names)
    head = LinearProbeHead(feature_dim, num_classes).to(device)

    # 4. Data loaders
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

    # 5. Loss with class weights
    weights = compute_class_weights(dataset.get_class_counts())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 6. Two-phase optimizers
    def get_warmup_optimizer() -> torch.optim.Optimizer:
        return torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def get_joint_optimizer() -> torch.optim.Optimizer:
        lora_params = [p for p in model.vision_model.parameters() if p.requires_grad]
        head_params = list(head.parameters())
        return torch.optim.AdamW(lora_params + head_params, lr=cfg.learning_rate * 0.5, weight_decay=cfg.weight_decay)

    optimizer = get_warmup_optimizer()
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.use_amp and device == "cuda" else None

    train_history: list[dict] = []
    val_history: list[dict] = []
    best_val_acc = 0.0
    best_head_state = None

    for epoch in range(cfg.num_epochs):
        # Switch to joint optimizer at the boundary
        if epoch == warmup_epochs:
            print(">>> Switching to joint LoRA + head training")
            optimizer = get_joint_optimizer()
            joint_steps = len(train_loader) * (cfg.num_epochs - warmup_epochs)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=joint_steps
            )
            if scaler:
                scaler = torch.amp.GradScaler("cuda")

        phase = "warmup" if epoch < warmup_epochs else "joint"
        model.vision_model.train() if epoch >= warmup_epochs else model.vision_model.eval()
        head.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[{phase}] Epoch {epoch+1}/{cfg.num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.autocast("cuda"):
                    vision_out = model.vision_model(pixel_values=pixel_values)
                    features = model.visual_projection(vision_out.pooler_output)
                    logits = head(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                vision_out = model.vision_model(pixel_values=pixel_values)
                features = model.visual_projection(vision_out.pooler_output)
                logits = head(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.vision_model.eval()
        head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
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
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} [{phase}] | "
            f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )

    # Merge LoRA weights back into the base model
    print("Merging LoRA adapters...")
    if isinstance(model.vision_model, PeftModel):
        model.vision_model = model.vision_model.merge_and_unload()

    # Restore best head
    if best_head_state:
        head.load_state_dict(best_head_state)

    # Save
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    torch.save(head.state_dict(), os.path.join(save_path, "classification_head.pt"))

    class_info = {"class_names": dataset.class_names}
    with open(os.path.join(save_path, "class_info.json"), "w", encoding="utf-8") as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)

    return {
        "strategy": FineTuneStrategy.lora.value,
        "save_path": save_path,
        "best_val_accuracy": best_val_acc,
        "train_history": train_history,
        "val_history": val_history,
        "dataset_stats": dataset.get_stats().to_dict(),
    }