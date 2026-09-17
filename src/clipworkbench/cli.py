"""
Copyright (c) 2023-2026 JunSu - AI
Released under the MIT License.
See LICENSE file for full license text.

Command-line interface for CLIPWorkbench.

Commands:
    train   - Fine-tune CLIP on an ImageFolder dir or a CSV annotation file
    infer   - Classify an image (zero-shot or fine-tuned model)
    export  - Export a fine-tuned model to ONNX
    serve   - Start the FastAPI server

Usage:
    python -m clipworkbench.cli train --dataset ./data --strategy linear_probe
    python -m clipworkbench.cli infer --image ./photo.jpg --model ./saved_models/m1
    python -m clipworkbench.cli infer --image ./photo.jpg --zero-shot --labels cat,dog,bird
    python -m clipworkbench.cli export --model ./saved_models/m1 --quantize int8
    python -m clipworkbench.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clipworkbench.core.config import get_settings


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune CLIP on an ImageFolder directory or a CSV annotation file."""
    from clipworkbench.data.dataset import load_dataset
    from clipworkbench.models.clip_loader import load_clip_model

    settings = get_settings()
    device = settings.resolved_device

    print(f"Device: {device}")
    print(f"Loading model: {settings.model.name}")
    model, processor = load_clip_model(device=device)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, processor)
    stats = dataset.get_stats()
    print(f"  Classes: {stats.num_classes} | Samples: {stats.num_samples}")
    for name, count in stats.class_counts.items():
        print(f"    {name}: {count}")

    save_path = args.save_path or str(Path(settings.storage.models_dir) / f"model_{args.strategy}")

    config_override = {}
    if args.epochs:
        config_override["num_epochs"] = args.epochs
    if args.batch_size:
        config_override["batch_size"] = args.batch_size
    if args.lr:
        config_override["learning_rate"] = args.lr

    if args.strategy == "lora":
        from clipworkbench.models.lora_finetune import train_lora
        result = train_lora(model, processor, dataset, save_path, device, config_override or None)
    else:
        from clipworkbench.models.linear_probe import train_linear_probe
        result = train_linear_probe(model, processor, dataset, save_path, device, config_override or None)

    print(f"\nTraining complete!")
    print(f"  Strategy: {result['strategy']}")
    print(f"  Best Val Accuracy: {result['best_val_accuracy']:.4f}")
    print(f"  Saved to: {result['save_path']}")

    # Register model
    from clipworkbench.core.schema import FineTuneStrategy
    from clipworkbench.models.registry import ModelRegistry
    reg = ModelRegistry(settings.storage.models_dir, settings.storage.db_path)
    info = reg.register(
        path=save_path,
        strategy=FineTuneStrategy(args.strategy),
        class_names=dataset.class_names,
        metrics={"val_accuracy": result["best_val_accuracy"]},
    )
    print(f"  Model ID: {info.model_id}")


def cmd_infer(args: argparse.Namespace) -> None:
    """Classify an image."""
    from PIL import Image
    from clipworkbench.core.schema import FineTuneStrategy
    from clipworkbench.models.inference import predict_image

    image = Image.open(args.image).convert("RGB")

    if args.zero_shot:
        strategy = FineTuneStrategy.zero_shot
        labels = args.labels.split(",") if args.labels else []
        model_path = None
    else:
        strategy = FineTuneStrategy.linear_probe
        labels = None
        model_path = args.model

    result = predict_image(
        image=image,
        model_path=model_path,
        candidate_labels=labels,
        strategy=strategy,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    print(f"\nPrediction (strategy: {result.strategy}):")
    print(f"  Best: {result.best_label} ({result.best_probability:.2%})")
    print(f"  Top-{len(result.top_k)}:")
    for item in result.top_k:
        bar = "█" * int(item.probability * 20)
        print(f"    {item.label:20s} {item.probability:6.2%} {bar}")
    print(f"  Latency: {result.latency_ms:.1f}ms")


def cmd_export(args: argparse.Namespace) -> None:
    """Export a model to ONNX."""
    from clipworkbench.models.onnx_export import export_to_onnx, benchmark_onnx

    result = export_to_onnx(args.model, quantization=args.quantize)
    print(f"\nExport complete!")
    print(f"  ONNX: {result.onnx_path} ({result.onnx_size_mb:.1f} MB)")
    print(f"  PyTorch: {result.pytorch_size_mb:.1f} MB")

    if result.quantized_path:
        print(f"  Quantized ({result.quantization}): {result.quantized_path} ({result.quantized_size_mb:.1f} MB)")

    # Benchmark
    print(f"\nBenchmarking ONNX Runtime...")
    bench_path = result.quantized_path or result.onnx_path
    bench = benchmark_onnx(bench_path, num_runs=args.benchmark_runs)
    print(f"  Mean: {bench['mean_ms']:.1f}ms | P50: {bench['p50_ms']:.1f}ms | P95: {bench['p95_ms']:.1f}ms")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "clipworkbench.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_gui(args: argparse.Namespace) -> None:
    """Launch the PySide6 desktop GUI."""
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Install with: pip install -e '.[gui]'")
        sys.exit(1)

    from pathlib import Path

    if args.mode == "admin":
        from clipworkbench.admin.main_window import AdminMainWindow, _resolve_icon_path
        window_cls = AdminMainWindow
    else:
        from clipworkbench.app.main_window import AppMainWindow
        from clipworkbench.admin.main_window import _resolve_icon_path
        window_cls = AppMainWindow

    app = QApplication(sys.argv)
    icon = _resolve_icon_path()
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))
    window = window_cls()
    window.show()
    sys.exit(app.exec())


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="clipworkbench",
        description="Multimodal AI desktop workbench: CLIP training, inference, and export.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Fine-tune CLIP on an ImageFolder dir or a CSV annotation file")
    p_train.add_argument("--dataset", required=True, help="Path to ImageFolder dir or .csv annotation file")
    p_train.add_argument("--strategy", choices=["linear_probe", "lora"], default="linear_probe")
    p_train.add_argument("--save-path", default="")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch-size", type=int)
    p_train.add_argument("--lr", type=float)
    p_train.set_defaults(func=cmd_train)

    # infer
    p_infer = sub.add_parser("infer", help="Classify an image")
    p_infer.add_argument("--image", required=True, help="Path to image file")
    p_infer.add_argument("--model", default="", help="Path to fine-tuned model")
    p_infer.add_argument("--zero-shot", action="store_true", help="Use zero-shot classification")
    p_infer.add_argument("--labels", default="", help="Comma-separated candidate labels (zero-shot)")
    p_infer.add_argument("--top-k", type=int, default=5)
    p_infer.add_argument("--temperature", type=float, default=1.0)
    p_infer.set_defaults(func=cmd_infer)

    # export
    p_export = sub.add_parser("export", help="Export model to ONNX")
    p_export.add_argument("--model", required=True, help="Path to fine-tuned model")
    p_export.add_argument("--quantize", choices=["fp16", "int8"], default=None)
    p_export.add_argument("--benchmark-runs", type=int, default=100)
    p_export.set_defaults(func=cmd_export)

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    # gui
    p_gui = sub.add_parser("gui", help="Launch PySide6 desktop GUI")
    p_gui.add_argument(
        "--mode",
        choices=["admin", "app"],
        default="admin",
        help="admin = training/management window, app = inference window (default: admin)",
    )
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()