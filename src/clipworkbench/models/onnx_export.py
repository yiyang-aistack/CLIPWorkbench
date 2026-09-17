# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""ONNX export and quantization for CLIP vision encoder + classification head.

Exports the inference graph (image → vision_model → visual_projection → head → logits)
to ONNX format, with optional FP16/INT8 quantization for CPU acceleration.

Usage:
    from clipworkbench.models.onnx_export import export_to_onnx
    result = export_to_onnx(model_path="saved_models/my_model", quantization="int8")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from clipworkbench.core.exceptions import ExportError
from clipworkbench.core.schema import ExportResult
from clipworkbench.models.linear_probe import LinearProbeHead


def export_to_onnx(
    model_path: str,
    output_dir: str | None = None,
    quantization: str | None = None,  # "fp16" | "int8" | None
) -> ExportResult:
    """Export a fine-tuned CLIP model to ONNX format.

    Args:
        model_path: Path to the saved fine-tuned model directory.
        output_dir: Directory for the ONNX file. Defaults to model_path.
        quantization: Optional quantization mode ("fp16" or "int8").

    Returns:
        ExportResult with file paths and size comparison.
    """
    try:
        import torch
        import onnx
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        raise ExportError(f"Missing dependency: {e}")

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ExportError(f"Model path not found: {model_path}")

    output_dir = Path(output_dir or model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_dir / "model.onnx")

    # Load model
    model = CLIPModel.from_pretrained(str(model_dir))
    processor = CLIPProcessor.from_pretrained(str(model_dir))

    # Load classification head
    with open(model_dir / "class_info.json", encoding="utf-8") as f:
        class_info = json.load(f)
    num_classes = len(class_info["class_names"])
    feature_dim = model.config.projection_dim
    head = LinearProbeHead(feature_dim, num_classes)

    head_path = model_dir / "classification_head.pt"
    if head_path.exists():
        head.load_state_dict(torch.load(head_path, map_location="cpu"))

    model.eval()
    head.eval()

    # Build a wrapper module for ONNX export
    class OnnxModel(torch.nn.Module):
        def __init__(self, clip_model: Any, head: LinearProbeHead) -> None:
            super().__init__()
            self.clip = clip_model
            self.head = head

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            features = self.clip.visual_projection(vision_out.pooler_output)
            logits = self.head(features)
            return logits

    onnx_wrapper = OnnxModel(model, head)

    # Dummy input
    image_size = processor.image_processor.size.get("shortest_edge", 224)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Size comparison
    pytorch_size = sum(
        f.stat().st_size for f in model_dir.rglob("*.safetensors")
    ) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

    quantized_path = None
    quantized_size = None

    # Quantization
    if quantization == "int8":
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = str(output_dir / "model_int8.onnx")
            quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: INT8 quantization skipped ({type(exc).__name__}: {exc})")

    elif quantization == "fp16":
        try:
            from onnxconverter_common import float16

            model_fp16 = float16.convert_float_to_float16(onnx.load(onnx_path))
            quantized_path = str(output_dir / "model_fp16.onnx")
            onnx.save(model_fp16, quantized_path)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: FP16 quantization skipped ({type(exc).__name__}: {exc})")

    return ExportResult(
        onnx_path=onnx_path,
        quantized_path=quantized_path,
        pytorch_size_mb=round(pytorch_size, 2),
        onnx_size_mb=round(onnx_size, 2),
        quantized_size_mb=round(quantized_size, 2) if quantized_size else None,
        quantization=quantization,
    )


def benchmark_onnx(
    onnx_path: str,
    num_runs: int = 100,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference latency.

    Returns dict with mean/p50/p95 latency in milliseconds.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        raise ExportError("onnxruntime is required for benchmarking")

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    h = input_shape[2] if len(input_shape) == 4 else 224
    w = input_shape[3] if len(input_shape) == 4 else 224
    dummy = np.random.randn(1, 3, h, w).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "num_runs": num_runs,
    }