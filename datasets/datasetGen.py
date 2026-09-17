"""Generate a synthetic sample dataset for GUI testing.

Creates:
  1) ImageFolder directory: datasets/samples/<class>/<idx>.jpg
  2) CSV annotation file:    datasets/samples/annotations.csv

The CSV has columns ``image_path,label,split`` where ``image_path`` is a path
**relative to the CSV file** (so it works after moving the whole folder).

Each class uses distinct colors/patterns so zero-shot / linear-probe training
can verify the pipeline end-to-end.

Run once:
    uv run python datasets/datasetGen.py
"""
from __future__ import annotations

import csv
import math
import zlib
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "samples"

# 6 classes × 10 images = 60 samples
CLASSES = {
    "cat":    {"bg": "#fce4d6", "fg": "#ff7043", "pattern": "cat"},
    "dog":    {"bg": "#e3f2fd", "fg": "#1976d2", "pattern": "dog"},
    "bird":   {"bg": "#e8f5e9", "fg": "#388e3c", "pattern": "bird"},
    "fish":   {"bg": "#e0f7fa", "fg": "#00838f", "pattern": "fish"},
    "car":    {"bg": "#fce4ec", "fg": "#c2185b", "pattern": "car"},
    "flower": {"bg": "#fff8e1", "fg": "#f9a825", "pattern": "flower"},
}
SAMPLES_PER_CLASS = 10
IMG_SIZE = (320, 320)


def _draw_cat(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon cat: circle head + two triangle ears + whiskers."""
    cx, cy = w // 2, h // 2 + 20
    r = min(w, h) // 3
    # ears
    draw.polygon([(cx - r, cy - r + 5), (cx - r // 2, cy - r - 40), (cx - 5, cy - r + 5)], fill=fg)
    draw.polygon([(cx + r, cy - r + 5), (cx + r // 2, cy - r - 40), (cx + 5, cy - r + 5)], fill=fg)
    # head
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    # eyes
    for dx in (-r // 3, r // 3):
        draw.ellipse((cx + dx - 6, cy - r // 3 - 6, cx + dx + 6, cy - r // 3 + 6), fill="white")
        draw.ellipse((cx + dx - 2, cy - r // 3 - 3, cx + dx + 2, cy - r // 3 + 3), fill="black")
    # nose + mouth
    draw.polygon([(cx - 5, cy + 5), (cx + 5, cy + 5), (cx, cy + 15)], fill="white")
    # whiskers
    for dy in (-5, 0, 5):
        draw.line([(cx - r + 10, cy + dy), (cx - r - 20, cy + dy)], fill="white", width=2)
        draw.line([(cx + r - 10, cy + dy), (cx + r + 20, cy + dy)], fill="white", width=2)


def _draw_dog(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon dog: rounded head + floppy ears + snout."""
    cx, cy = w // 2, h // 2 + 10
    r = min(w, h) // 3
    # floppy ears (ovals on sides)
    draw.ellipse((cx - r - 25, cy - 10, cx - r + 5, cy + r + 15), fill=fg)
    draw.ellipse((cx + r - 5, cy - 10, cx + r + 25, cy + r + 15), fill=fg)
    # head
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    # snout
    draw.ellipse((cx - r // 2, cy + r // 4, cx + r // 2, cy + r), fill="white")
    # eyes
    for dx in (-r // 3, r // 3):
        draw.ellipse((cx + dx - 5, cy - r // 3 - 5, cx + dx + 5, cy - r // 3 + 5), fill="white")
        draw.ellipse((cx + dx - 2, cy - r // 3 - 2, cx + dx + 2, cy - r // 3 + 2), fill="black")
    # nose
    draw.ellipse((cx - 10, cy + r // 3 - 3, cx + 10, cy + r // 3 + 8), fill="black")


def _draw_bird(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon bird: body oval + wing + beak."""
    cx, cy = w // 2, h // 2
    r = min(w, h) // 3
    # body
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    # head
    draw.ellipse((cx - r - 15, cy - r // 2 - 15, cx - r + 15, cy - r // 2 + 15), fill=fg)
    # wing
    draw.ellipse((cx - 10, cy - 5, cx + r - 5, cy + r - 10), fill="white")
    # beak (triangle)
    draw.polygon([(cx - r - 15, cy - r // 2), (cx - r - 35, cy - r // 2 + 5),
                  (cx - r - 15, cy - r // 2 + 15)], fill="orange")
    # eye
    draw.ellipse((cx - r - 8, cy - r // 2 - 8, cx - r + 8, cy - r // 2 + 8), fill="white")
    draw.ellipse((cx - r - 3, cy - r // 2 - 4, cx - r + 1, cy - r // 2), fill="black")


def _draw_fish(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon fish: oval body + triangle tail."""
    cx, cy = w // 2, h // 2
    r = min(w, h) // 3
    # body
    draw.ellipse((cx - r - 20, cy - r // 2, cx + r, cy + r // 2), fill=fg)
    # tail
    draw.polygon([
        (cx - r - 20, cy - r // 2),
        (cx - r - 50, cy),
        (cx - r - 20, cy + r // 2),
    ], fill=fg)
    # eye
    draw.ellipse((cx + r // 2, cy - 10, cx + r // 2 + 18, cy + 8), fill="white")
    draw.ellipse((cx + r // 2 + 8, cy - 5, cx + r // 2 + 14, cy + 5), fill="black")
    # fin
    draw.polygon([
        (cx - 10, cy + r // 2),
        (cx + 10, cy + r // 2),
        (cx, cy + r // 2 + 25),
    ], fill="white")


def _draw_car(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon car: rectangular body + cabin + wheels."""
    cx, cy = w // 2, h // 2 + 10
    body_top = cy - 30
    body_bot = cy + 30
    body_left = cx - 90
    body_right = cx + 90
    # lower body
    draw.rectangle((body_left, body_top + 20, body_right, body_bot), fill=fg)
    # cabin (trapezoid)
    draw.polygon([
        (body_left + 20, body_top + 20),
        (body_left + 50, body_top - 10),
        (body_right - 50, body_top - 10),
        (body_right - 20, body_top + 20),
    ], fill=fg)
    # windshield
    draw.polygon([
        (body_left + 30, body_top + 18),
        (body_left + 55, body_top - 5),
        (body_right - 55, body_top - 5),
        (body_right - 30, body_top + 18),
    ], fill="#b3e5fc")
    # wheels
    for wx in (body_left + 25, body_right - 25):
        draw.ellipse((wx - 18, body_bot, wx + 18, body_bot + 36), fill="#333")
        draw.ellipse((wx - 8, body_bot + 10, wx + 8, body_bot + 26), fill="#888")
    # headlight
    draw.ellipse((body_right - 8, body_top + 25, body_right + 10, body_top + 40), fill="#fff9c4")


def _draw_flower(draw: ImageDraw.ImageDraw, w: int, h: int, fg: str) -> None:
    """Simple cartoon flower: 5 petals + center + stem."""
    cx, cy = w // 2, h // 2 - 20
    petal_r = 45
    # stem
    draw.rectangle((cx - 6, cy + petal_r, cx + 6, h - 20), fill="#2e7d32")
    # leaves
    draw.ellipse((cx - 45, cy + petal_r + 40, cx - 5, cy + petal_r + 70), fill="#43a047")
    draw.ellipse((cx + 5, cy + petal_r + 60, cx + 45, cy + petal_r + 90), fill="#43a047")
    # 5 petals
    for i in range(5):
        angle = math.radians(i * 72 - 90)
        px = cx + int(math.cos(angle) * petal_r * 0.8)
        py = cy + int(math.sin(angle) * petal_r * 0.8)
        draw.ellipse((px - 35, py - 25, px + 35, py + 25), fill=fg)
    # center
    draw.ellipse((cx - 25, cy - 25, cx + 25, cy + 25), fill="#fff59d")


DRAW_FNS = {
    "cat": _draw_cat,
    "dog": _draw_dog,
    "bird": _draw_bird,
    "fish": _draw_fish,
    "car": _draw_car,
    "flower": _draw_flower,
}


def make_image(pattern: str, bg: str, fg: str, seed: int) -> Image.Image:
    """Create one synthetic image. `seed` slightly varies position/size."""
    w, h = IMG_SIZE

    # deterministic jitter per seed
    rng = random.Random(seed)
    jitter_x = rng.randint(-20, 20)
    jitter_y = rng.randint(-20, 20)
    scale = rng.uniform(0.85, 1.15)

    # shift draw origin by jitter (simplified: redraw with offset)
    draw_fn = DRAW_FNS[pattern]
    # quick trick: draw full image then paste onto jittered background? just draw with
    # a small offset into a fresh image via translate – instead we simply translate by
    # redrawing with shifted coordinates inside a larger temp canvas.
    # Simpler approach: temporarily move entire image by pasting later.
    temp = Image.new("RGB", (w, h), bg)
    tdraw = ImageDraw.Draw(temp)
    # Use offset wrapper: we'll just re-call with an offset applied below.
    # For simplicity, draw on temp, then scale + paste onto img with jitter.
    draw_fn(tdraw, w, h, fg)
    new_w, new_h = int(w * scale), int(h * scale)
    temp = temp.resize((new_w, new_h), Image.Resampling.BILINEAR)
    img = Image.new("RGB", (w, h), bg)
    paste_x = max(0, (w - new_w) // 2 + jitter_x)
    paste_y = max(0, (h - new_h) // 2 + jitter_y)
    img.paste(temp, (paste_x, paste_y))
    return img


CSV_NAME = "annotations.csv"
VAL_FRACTION = 0.2  # 20% per class → val, 80% → train


def main() -> None:
    print(f"Generating sample dataset at: {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)

    # 收集所有 CSV 行：(相对 image_path, label, split)
    csv_rows: list[tuple[str, str, str]] = []

    total = 0
    for cls_name, cfg in CLASSES.items():
        cls_dir = ROOT / cls_name
        cls_dir.mkdir(exist_ok=True)

        # 每类前 N 张 train, 后 N 张 val（确定性划分）
        n_val = max(1, int(SAMPLES_PER_CLASS * VAL_FRACTION))
        n_train = SAMPLES_PER_CLASS - n_val

        for i in range(SAMPLES_PER_CLASS):
            img = make_image(
                cfg["pattern"], cfg["bg"], cfg["fg"],
                seed=zlib.crc32(cls_name.encode()) + i,
            )
            out_path = cls_dir / f"{cls_name}_{i:02d}.jpg"
            img.save(out_path, quality=92)

            # 相对 CSV 文件所在目录 (ROOT) 的路径，使用 forward-slash 兼容跨平台
            rel_path = out_path.relative_to(ROOT).as_posix()
            split = "train" if i < n_train else "val"
            csv_rows.append((rel_path, cls_name, split))
            total += 1

        print(
            f"  {cls_name:>7s}: {SAMPLES_PER_CLASS} images "
            f"(train={n_train}, val={n_val})  →  {cls_dir}"
        )

    # ---- 写 annotations.csv ----
    csv_path = ROOT / CSV_NAME
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "split"])
        writer.writerows(csv_rows)

    train_count = sum(1 for *_, s in csv_rows if s == "train")
    val_count = sum(1 for *_, s in csv_rows if s == "val")

    print()
    print(f"Done. {len(CLASSES)} classes × {SAMPLES_PER_CLASS} = {total} images.")
    print(f"  train: {train_count}  |  val: {val_count}")
    print()
    print(f"ImageFolder import path:  {ROOT}")
    print(f"CSV     import path:  {csv_path}")


if __name__ == "__main__":
    main()