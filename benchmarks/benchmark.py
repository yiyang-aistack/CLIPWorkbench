"""Benchmark: PyTorch vs ONNX Runtime inference latency comparison.

Compares end-to-end inference latency and model file size
across PyTorch, ONNX FP32, and optionally ONNX INT8.

Usage:
    python benchmarks/benchmark.py --model ./saved_models/model_linear_probe --runs 200
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from clipworkbench.core.config import get_settings


def benchmark_pytorch(model_path: str, num_runs: int = 100) -> dict:
    """Benchmark PyTorch inference latency."""
    import numpy as np
    from PIL import Image
    from clipworkbench.models.inference import load_finetuned_model

    model, processor, info = load_finetuned_model(model_path)
    device = info["device"]
    head = info["head"]
    class_names = info["class_names"]

    # Create a dummy image
    dummy = Image.new("RGB", (224, 224), (128, 128, 128))

    # Warmup
    for _ in range(10):
        from clipworkbench.models.inference import predict_finetuned
        predict_finetuned(model, processor, head, dummy, class_names, device)

    latencies = []
    for _ in range(num_runs):
        from clipworkbench.models.inference import predict_finetuned
        start = time.perf_counter()
        predict_finetuned(model, processor, head, dummy, class_names, device)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    # Model size
    total_size = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()) / (1024 * 1024)

    return {
        "engine": "PyTorch",
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "model_size_mb": round(total_size, 2),
        "num_runs": num_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime.")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs")
    parser.add_argument("--quantize", choices=["fp16", "int8"], default=None)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}\n")

    # PyTorch benchmark
    print("Benchmarking PyTorch...")
    pt_result = benchmark_pytorch(args.model, args.runs)
    print(f"  Mean: {pt_result['mean_ms']:.1f}ms | P50: {pt_result['p50_ms']:.1f}ms | P95: {pt_result['p95_ms']:.1f}ms")
    print(f"  Size: {pt_result['model_size_mb']:.1f}MB")

    # ONNX export + benchmark
    print("\nExporting to ONNX...")
    from clipworkbench.models.onnx_export import export_to_onnx, benchmark_onnx

    export_result = export_to_onnx(args.model, quantization=args.quantize)
    print(f"  ONNX: {export_result.onnx_path} ({export_result.onnx_size_mb:.1f}MB)")

    if export_result.quantized_path:
        print(f"  Quantized ({args.quantize}): {export_result.quantized_path} ({export_result.quantized_size_mb:.1f}MB)")

    print("\nBenchmarking ONNX Runtime...")
    onnx_bench = benchmark_onnx(export_result.onnx_path, args.runs)
    print(f"  Mean: {onnx_bench['mean_ms']:.1f}ms | P50: {onnx_bench['p50_ms']:.1f}ms | P95: {onnx_bench['p95_ms']:.1f}ms")

    if export_result.quantized_path:
        print("\nBenchmarking Quantized ONNX...")
        quant_bench = benchmark_onnx(export_result.quantized_path, args.runs)
        print(f"  Mean: {quant_bench['mean_ms']:.1f}ms | P50: {quant_bench['p50_ms']:.1f}ms | P95: {quant_bench['p95_ms']:.1f}ms")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Engine':<25} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'Size (MB)':>10}")
    print("-" * 60)
    print(f"{'PyTorch':<25} {pt_result['mean_ms']:>10.1f} {pt_result['p50_ms']:>10.1f} {pt_result['p95_ms']:>10.1f} {pt_result['model_size_mb']:>10.1f}")
    print(f"{'ONNX FP32':<25} {onnx_bench['mean_ms']:>10.1f} {onnx_bench['p50_ms']:>10.1f} {onnx_bench['p95_ms']:>10.1f} {export_result.onnx_size_mb:>10.1f}")
    if export_result.quantized_path and 'quant_bench' in locals():
        print(f"{'ONNX ' + str(args.quantize):<25} {quant_bench['mean_ms']:>10.1f} {quant_bench['p50_ms']:>10.1f} {quant_bench['p95_ms']:>10.1f} {export_result.quantized_size_mb:>10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()