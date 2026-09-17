# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""Admin GUI main window: dataset import, training, model registry, ONNX export.

Heavy core imports (torch, transformers, peft) are performed lazily inside
methods so the window can open even when optional dependencies are missing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _torch_available() -> bool:
    """Return True if torch can be imported."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class TrainingThread(QThread):
    """Background training worker.

    Signals:
        progress(epoch, train_loss, val_accuracy)
        finished_ok(result_dict)
        error(message)
    """

    progress = Signal(int, float, float)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        dataset_path: str,
        strategy: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        save_path: str,
        use_early_stopping: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.strategy = strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.use_early_stopping = use_early_stopping

    def run(self) -> None:
        try:
            from clipworkbench.core.schema import FineTuneStrategy
            from clipworkbench.data.dataset import load_dataset
            from clipworkbench.models.clip_loader import load_clip_model
            from clipworkbench.models.lora_finetune import train_lora
            from clipworkbench.models.linear_probe import train_linear_probe

            strategy_enum = FineTuneStrategy(self.strategy)
            model, processor = load_clip_model()
            dataset = load_dataset(self.dataset_path, processor=processor)
            config_override = {
                "num_epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "use_early_stopping": self.use_early_stopping,
            }

            def callback(epoch: int, total: int, metrics: dict) -> None:
                self.progress.emit(
                    epoch,
                    float(metrics.get("train_loss", 0.0)),
                    float(metrics.get("val_accuracy", 0.0)),
                )

            train_fn = (
                train_lora
                if strategy_enum == FineTuneStrategy.lora
                else train_linear_probe
            )
            result = train_fn(
                model,
                processor,
                dataset,
                self.save_path,
                config_override=config_override,
                status_callback=callback,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")


class OnnxExportThread(QThread):
    """QThread that exports a trained model to ONNX (runs in background).

    Emits:
        finished_ok(dict): success payload with onnx paths and sizes.
        error(str):         human-readable error message.
    """

    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(self, model_path: str, quantization: str | None) -> None:
        super().__init__()
        self.model_path = model_path
        self.quantization = quantization

    def run(self) -> None:
        try:
            from clipworkbench.models.onnx_export import export_to_onnx

            result = export_to_onnx(
                model_path=self.model_path,
                quantization=self.quantization,
            )
            self.finished_ok.emit(
                {
                    "onnx_path": result.onnx_path,
                    "quantized_path": result.quantized_path,
                    "onnx_size_mb": result.onnx_size_mb,
                    "quantized_size_mb": result.quantized_size_mb,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")


def _resolve_icon_path() -> Path | None:
    """Locate the app icon under assets/ (development or bundled)."""
    candidates = [
        Path("assets/junsuAi.jpg"),                      # cwd / bundled exe
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.jpg",  # repo root
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.png",
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.ico",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


class AdminMainWindow(QMainWindow):
    """Admin-side main window: dataset, training, and model management."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CLIPWorkbench — Admin")
        self.resize(1200, 760)

        icon = _resolve_icon_path()
        if icon is not None:
            self.setWindowIcon(QIcon(str(icon)))

        self._dataset_path: str = ""
        self._models: list[dict] = []
        self._train_thread: TrainingThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        self._refresh_status()

    # ---- Panel construction ----------------------------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        v.addWidget(QLabel("<b>Dataset</b>"))
        self.import_btn = QPushButton("Import Dataset")
        self.import_btn.clicked.connect(self._on_import_dataset)
        v.addWidget(self.import_btn)
        self.dataset_path_label = QLabel("No dataset selected")
        self.dataset_path_label.setWordWrap(True)
        v.addWidget(self.dataset_path_label)

        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["Class", "Samples"])
        v.addWidget(self.class_table)

        v.addSpacing(10)
        v.addWidget(QLabel("<b>Zero-Shot Test</b>"))
        self.zs_labels_edit = QLineEdit()
        self.zs_labels_edit.setPlaceholderText("cat, dog, bird, ...")
        v.addWidget(self.zs_labels_edit)
        self.zs_run_btn = QPushButton("Run Zero-Shot")
        self.zs_run_btn.clicked.connect(self._on_run_zero_shot)
        v.addWidget(self.zs_run_btn)
        self.zs_result = QTextEdit()
        self.zs_result.setReadOnly(True)
        self.zs_result.setMaximumHeight(120)
        v.addWidget(self.zs_result)

        v.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(QLabel("<b>Training Configuration</b>"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        # label column fixed width, value columns stretch equally
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Row 0: Strategy (left) + Epochs (right)
        grid.addWidget(QLabel("Strategy:"), 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["linear_probe", "lora"])
        grid.addWidget(self.strategy_combo, 0, 1)

        grid.addWidget(QLabel("Epochs:"), 0, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(10)
        grid.addWidget(self.epochs_spin, 0, 3)

        # Row 1: Batch (left) + Learning Rate (right)
        grid.addWidget(QLabel("Batch:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        grid.addWidget(self.batch_spin, 1, 1)

        grid.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-4)
        grid.addWidget(self.lr_spin, 1, 3)

        # Row 2: Early Stop (left column span, right column empty)
        grid.addWidget(QLabel("Early Stop:"), 2, 0)
        self.early_stop_cb = QCheckBox("Enable")
        self.early_stop_cb.setChecked(True)
        grid.addWidget(self.early_stop_cb, 2, 1, 1, 3)

        v.addLayout(grid)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "    stop:0 #43e97b, stop:1 #2fa866);"
            "  color: white;"
            "  border: none;"
            "  border-radius: 6px;"
            "  padding: 10px 20px;"
            "  font-size: 14px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "    stop:0 #5eefa0, stop:1 #38b874);"
            "}"
            "QPushButton:pressed {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "    stop:0 #2fa866, stop:1 #248a53);"
            "}"
            "QPushButton:disabled {"
            "  background: #9c9c9c;"
            "  color: #e0e0e0;"
            "}"
        )
        self.train_btn.clicked.connect(self._on_start_training)
        v.addWidget(self.train_btn)

        self.progress_bar = QProgressBar()
        v.addWidget(self.progress_bar)
        self.loss_label = QLabel("Idle")
        v.addWidget(self.loss_label)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        v.addWidget(self.train_log)

        if not _torch_available():
            warn = QLabel("PyTorch is not installed. Training is unavailable.")
            warn.setStyleSheet("color: #b00; font-weight: bold;")
            v.addWidget(warn)
            self.train_btn.setEnabled(False)

        v.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(QLabel("<b>Model Registry</b>"))
        self.model_list = QListWidget()
        v.addWidget(self.model_list)
        self.model_list.currentItemChanged.connect(self._update_export_button_state)
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        v.addWidget(self.refresh_models_btn)

        v.addSpacing(10)
        v.addWidget(QLabel("<b>ONNX Export</b>"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Quantization:"))
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["none", "fp16", "int8"])
        row.addWidget(self.quant_combo, 1)
        v.addLayout(row)
        self.export_btn = QPushButton("Export ONNX")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_onnx)
        v.addWidget(self.export_btn)

        v.addStretch(1)
        return panel

    # ---- Status / models -------------------------------------------------

    def _refresh_status(self) -> None:
        try:
            from clipworkbench.core.config import get_settings

            device = get_settings().resolved_device
        except Exception:
            device = "unknown"
        self.statusBar().showMessage(f"Device: {device} | Ready")
        self._refresh_models()

    def _refresh_models(self) -> None:
        self._models = []
        self.model_list.clear()
        try:
            from clipworkbench.models.registry import ModelRegistry

            models = ModelRegistry().list_models()
        except Exception as exc:  # noqa: BLE001
            self.model_list.addItem(f"(error: {exc})")
            return
        self._models = list(models)
        for m in self._models:
            label = (
                f"{m.get('model_id', '?')} - {m.get('strategy', '?')} "
                f"({m.get('num_classes', '?')} classes)"
            )
            self.model_list.addItem(label)
        self._update_export_button_state()

    # ---- Dataset ---------------------------------------------------------

    def _on_import_dataset(self) -> None:
        choice, _ = QInputDialog.getItem(
            self, "Import Dataset",
            "Select dataset source type:",
            ["ImageFolder (directory)", "CSV annotation file"],
            0, False,
        )
        if choice == "ImageFolder (directory)":
            path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
            if not path:
                return
            self._dataset_path = path
            self.dataset_path_label.setText(path)
            counts = self._scan_image_folder(path)

        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select CSV Annotation File",
                "", "CSV Files (*.csv)",
            )
            if not path:
                return
            self._dataset_path = path
            self.dataset_path_label.setText(path)
            try:
                counts = self._scan_csv_file(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self, "CSV parse error",
                    f"Failed to parse CSV:\n{exc}",
                )
                return

        self.class_table.setRowCount(0)
        for name, count in counts.items():
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)
            self.class_table.setItem(row, 0, QTableWidgetItem(name))
            self.class_table.setItem(row, 1, QTableWidgetItem(str(count)))
        total = sum(counts.values())
        self.statusBar().showMessage(
            f"Dataset: {len(counts)} classes, {total} samples"
        )

    @staticmethod
    def _scan_image_folder(root: str) -> "dict[str, int]":
        counts: dict[str, int] = {}
        for entry in sorted(Path(root).iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            n = sum(
                1
                for f in entry.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            )
            if n > 0:
                counts[entry.name] = n
        return counts

    @staticmethod
    def _scan_csv_file(csv_path: str) -> "dict[str, int]":
        """Parse CSV and return label → sample count."""
        import csv as _csv
        counts: dict[str, int] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(
                    f"CSV missing 'label' column. Found: {reader.fieldnames}"
                )
            for row in reader:
                lbl = (row.get("label") or "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
        if not counts:
            raise ValueError("CSV has no data rows")
        return counts

    # ---- Zero-shot -------------------------------------------------------

    def _on_run_zero_shot(self) -> None:
        if not _torch_available():
            QMessageBox.warning(self, "Unavailable", "PyTorch is not installed.")
            return
        text = self.zs_labels_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "No labels", "Enter candidate labels.")
            return
        labels = [s.strip() for s in text.split(",") if s.strip()]
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image to Test",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not img_path:
            return
        self.zs_result.setPlainText("Running zero-shot...")
        QApplication.processEvents()
        try:
            from PIL import Image

            from clipworkbench.core.schema import FineTuneStrategy
            from clipworkbench.models.inference import predict_image

            image = Image.open(img_path).convert("RGB")
            result = predict_image(
                image=image,
                candidate_labels=labels,
                strategy=FineTuneStrategy.zero_shot,
                top_k=5,
            )
            lines = [f"Best: {result.best_label} ({result.best_probability:.2%})"]
            for item in result.top_k:
                lines.append(f"  {item.label}: {item.probability:.2%}")
            lines.append(f"Latency: {result.latency_ms:.1f} ms")
            self.zs_result.setPlainText("\n".join(lines))
            self.statusBar().showMessage(f"Zero-shot: {result.best_label}")
        except Exception as exc:  # noqa: BLE001
            self.zs_result.setPlainText(f"Error: {exc}")
            self.statusBar().showMessage("Zero-shot failed")

    # ---- Training --------------------------------------------------------

    def _on_start_training(self) -> None:
        if not _torch_available():
            QMessageBox.warning(self, "Unavailable", "PyTorch is not installed.")
            return
        if not self._dataset_path:
            QMessageBox.warning(self, "No dataset", "Import a dataset first.")
            return
        strategy = self.strategy_combo.currentText()
        save_path = os.path.join("./saved_models", f"{strategy}_{int(time.time())}")
        self.progress_bar.setValue(0)
        self.loss_label.setText("Starting...")
        self.train_log.clear()
        self.train_btn.setEnabled(False)
        self._train_thread = TrainingThread(
            self._dataset_path,
            strategy,
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.lr_spin.value(),
            save_path,
            use_early_stopping=self.early_stop_cb.isChecked(),
        )
        self._train_thread.progress.connect(self._on_train_progress)
        self._train_thread.finished_ok.connect(self._on_train_finished)
        self._train_thread.error.connect(self._on_train_error)
        self._train_thread.start()
        self.statusBar().showMessage("Training...")

    def _on_train_progress(self, epoch: int, loss: float, val_acc: float) -> None:
        total = self.epochs_spin.value()
        self.progress_bar.setValue(int(epoch / total * 100))
        self.loss_label.setText(
            f"Epoch {epoch}/{total} | loss={loss:.4f} | val_acc={val_acc:.2%}"
        )
        self.train_log.append(f"Epoch {epoch}: loss={loss:.4f}, val_acc={val_acc:.2%}")

    def _on_train_finished(self, result: dict) -> None:
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        save_path = result.get("save_path", "")
        acc = result.get("best_val_accuracy", 0.0)
        self.train_log.append(
            f"\nDone. Saved to {save_path} | best val acc={acc:.2%}"
        )
        self.statusBar().showMessage("Training complete")

        # Register the new model in the SQLite-backed registry
        try:
            from clipworkbench.core.schema import FineTuneStrategy
            from clipworkbench.models.registry import ModelRegistry

            strategy_val = result.get("strategy", "linear_probe")
            class_names = result.get("dataset_stats", {}).get("class_names", [])
            ModelRegistry().register(
                path=save_path,
                strategy=FineTuneStrategy(strategy_val),
                class_names=class_names,
                metrics={"best_val_accuracy": acc},
            )
        except Exception as exc:  # noqa: BLE001
            self.train_log.append(f"(Registry warning: {exc})")

        self._refresh_models()

    def _on_train_error(self, msg: str) -> None:
        self.train_btn.setEnabled(True)
        self.train_log.append(f"ERROR: {msg}")
        self.statusBar().showMessage("Training failed")
        QMessageBox.critical(self, "Training error", msg)

    # ---- ONNX export -----------------------------------------------------

    def _update_export_button_state(self) -> None:
        """Enable export_btn only when a model row is selected."""
        row = self.model_list.currentRow()
        self.export_btn.setEnabled(0 <= row < len(self._models))

    def _on_export_onnx(self) -> None:
        row = self.model_list.currentRow()
        if row < 0 or row >= len(self._models):
            QMessageBox.warning(self, "No model", "Select a model to export.")
            return
        model = self._models[row]
        model_path = model.get("path")
        if not model_path:
            QMessageBox.warning(self, "Bad path", "Model has no path.")
            return

        quant = self.quant_combo.currentText()
        quant_arg = None if quant == "none" else quant

        self.export_btn.setEnabled(False)
        self.statusBar().showMessage(f"Exporting ONNX ({quant or 'none'})...")

        self._export_thread = OnnxExportThread(model_path, quant_arg)
        self._export_thread.finished_ok.connect(self._on_export_finished)
        self._export_thread.error.connect(self._on_export_error)
        self._export_thread.start()

    def _on_export_finished(self, result: dict) -> None:
        msg = f"ONNX: {result['onnx_path']}\nSize: {result['onnx_size_mb']} MB"
        if result.get("quantized_path"):
            msg += f"\nQuantized: {result['quantized_size_mb']} MB"
        QMessageBox.information(self, "Export complete", msg)
        self.statusBar().showMessage("ONNX export complete")
        self._update_export_button_state()

    def _on_export_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Export failed", msg)
        self.statusBar().showMessage("Export failed")
        self._update_export_button_state()


def main() -> None:
    """Entry point for running the Admin window standalone."""
    import sys

    app = QApplication(sys.argv)
    icon = _resolve_icon_path()
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))
    window = AdminMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()