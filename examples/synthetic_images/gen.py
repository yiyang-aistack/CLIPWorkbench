"""Generate synthetic image dataset for testing.

Creates colored shapes as proxy "classes" in an ImageFolder structure.
No external dependencies beyond Pillow.

Usage:
    python examples/synthetic_images/gen.py --output ./data --classes 5 --samples 20
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ["circle", "rectangle", "triangle", "cross"]
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0),
]


def generate_image(shape: str, color: tuple[int, int, int], size: int = 64) -> Image.Image:
    """Draw a single colored shape on a white background."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = size // 6

    if shape == "circle":
        draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    elif shape == "rectangle":
        draw.rectangle([margin, margin, size - margin, size - margin], fill=color)
    elif shape == "triangle":
        draw.polygon(
            [(size // 2, margin), (margin, size - margin), (size - margin, size - margin)],
            fill=color,
        )
    elif shape == "cross":
        thickness = size // 4
        draw.rectangle(
            [size // 2 - thickness, margin, size // 2 + thickness, size - margin], fill=color
        )
        draw.rectangle(
            [margin, size // 2 - thickness, size - margin, size // 2 + thickness], fill=color
        )

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic image dataset.")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--samples", type=int, default=20, help="Samples per class")
    parser.add_argument("--size", type=int, default=64, help="Image size (pixels)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for i in range(args.classes):
        shape = SHAPES[i % len(SHAPES)]
        color = COLORS[i % len(COLORS)]
        class_name = f"class_{i:02d}_{shape}_{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        class_dir = output / class_name
        class_dir.mkdir(exist_ok=True)

        for j in range(args.samples):
            img = generate_image(shape, color, args.size)
            # Add slight random noise for variety
            if random.random() > 0.5:
                img = img.rotate(random.randint(-10, 10))
            img.save(class_dir / f"img_{j:03d}.jpg", quality=90)

    print(f"Generated {args.classes} classes x {args.samples} images = {args.classes * args.samples} total")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
