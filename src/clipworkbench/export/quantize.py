"""Quantization utilities for ONNX models.

Provides FP16 and INT8 quantization wrappers around onnxruntime tools.
Imported by models/onnx_export.py for the --quantize flag.
"""

from __future__ import annotations

from pathlib import Path


def quantize_int8(onnx_path: str, output_path: str | None = None) -> str:
    """Apply dynamic INT8 quantization to an ONNX model.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        output_path: Output path. Defaults to {onnx_path}_int8.onnx.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        p = Path(onnx_path)
        output_path = str(p.parent / f"{p.stem}_int8{p.suffix}")

    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QUInt8)
    return output_path


def quantize_fp16(onnx_path: str, output_path: str | None = None) -> str:
    """Convert an ONNX model to FP16 precision.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        output_path: Output path. Defaults to {onnx_path}_fp16.onnx.

    Returns:
        Path to the FP16 model.
    """
    import onnx
    from onnxconverter_common import float16

    if output_path is None:
        p = Path(onnx_path)
        output_path = str(p.parent / f"{p.stem}_fp16{p.suffix}")

    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)
    return output_path
