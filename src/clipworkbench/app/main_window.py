# Copyright (c) 2023-2026 JunSu - AI
# Released under the MIT License.
# See LICENSE file for full license text.

"""App GUI main window: image selection, inference, Top-K, export.

This module is part of the GUI "shell" layer for end-users. Heavy core
imports (torch, transformers, PIL) are performed lazily inside worker threads
so that the window can still be opened even when optional dependencies are
missing.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
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


class InferenceThread(QThread):
    """Single-image inference worker.

    Signals:
        result_ready(result_dict)
        error(message)
    """

    result_ready = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        image_path: str,
        model_path: str | None,
        candidate_labels: list[str] | None,
        strategy: str,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.candidate_labels = candidate_labels
        self.strategy = strategy
        self.top_k = top_k

    def run(self) -> None:
        try:
            from PIL import Image

            from clipworkbench.core.schema import FineTuneStrategy
            from clipworkbench.models.inference import predict_image

            image = Image.open(self.image_path).convert("RGB")
            result = predict_image(
                image=image,
                model_path=self.model_path,
                candidate_labels=self.candidate_labels,
                strategy=FineTuneStrategy(self.strategy),
                top_k=self.top_k,
            )
            self.result_ready.emit(result.model_dump())
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")


class BatchInferenceThread(QThread):
    """Batch inference over a folder of images.

    Signals:
        row_ready(image_name, label, probability)
        finished_ok(done, total)
        error(message)
    """

    row_ready = Signal(str, str, float)
    finished_ok = Signal(int, int)
    error = Signal(str)

    def __init__(
        self,
        folder: str,
        model_path: str | None,
        candidate_labels: list[str] | None,
        strategy: str,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.model_path = model_path
        self.candidate_labels = candidate_labels
        self.strategy = strategy
        self.top_k = top_k

    def run(self) -> None:
        try:
            from PIL import Image

            from clipworkbench.core.schema import FineTuneStrategy
            from clipworkbench.models.inference import predict_image

            strategy_enum = FineTuneStrategy(self.strategy)
            files = sorted(
                f
                for f in Path(self.folder).iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            )
            total = len(files)
            done = 0
            for f in files:
                image = Image.open(f).convert("RGB")
                result = predict_image(
                    image=image,
                    model_path=self.model_path,
                    candidate_labels=self.candidate_labels,
                    strategy=strategy_enum,
                    top_k=1,
                )
                self.row_ready.emit(f.name, result.best_label, result.best_probability)
                done += 1
            self.finished_ok.emit(done, total)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")


def _resolve_icon_path() -> Path | None:
    """Locate the app icon under assets/ (development or bundled)."""
    candidates = [
        Path("assets/junsuAi.jpg"),
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.jpg",
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.png",
        Path(__file__).resolve().parents[3] / "assets" / "junsuAi.ico",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


class AppMainWindow(QMainWindow):
    """Client-side inference window for end-users."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CLIPWorkbench — Inference")
        self.resize(900, 720)

        icon = _resolve_icon_path()
        if icon is not None:
            self.setWindowIcon(QIcon(str(icon)))

        self._image_path: str = ""
        self._infer_thread: InferenceThread | None = None
        self._batch_thread: BatchInferenceThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # Top: image selection + preview
        row = QHBoxLayout()
        self.select_btn = QPushButton("Select Image")
        self.select_btn.clicked.connect(self._on_select_image)
        row.addWidget(self.select_btn)
        self.image_path_label = QLabel("No image selected")
        row.addWidget(self.image_path_label, 1)
        v.addLayout(row)

        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        self.preview.setStyleSheet("border: 1px solid #ccc;")
        v.addWidget(self.preview)

        # Mode radio buttons (auto-exclusive within the same parent)
        row = QHBoxLayout()
        self.rb_finetuned = QRadioButton("Fine-tuned")
        self.rb_zeroshot = QRadioButton("Zero-shot")
        self.rb_finetuned.setChecked(True)
        row.addWidget(self.rb_finetuned)
        row.addWidget(self.rb_zeroshot)
        row.addStretch(1)
        v.addLayout(row)

        # Fine-tuned model selector (populated from ModelRegistry)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fine-tuned CLIP model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        row.addWidget(self.model_combo, 1)
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.clicked.connect(self._refresh_models_combo)
        row.addWidget(self.refresh_models_btn)
        self.browse_model_btn = QPushButton("Browse…")
        self.browse_model_btn.setToolTip("Manually pick a model directory not in the registry")
        self.browse_model_btn.clicked.connect(self._on_browse_model)
        row.addWidget(self.browse_model_btn)
        v.addLayout(row)

        # Zero-shot labels
        row = QHBoxLayout()
        row.addWidget(QLabel("Zero-shot labels:"))
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("cat, dog, bird")
        self.labels_edit.setEnabled(False)
        row.addWidget(self.labels_edit, 1)
        v.addLayout(row)

        self.rb_finetuned.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()

        # Action buttons
        row = QHBoxLayout()
        self.classify_btn = QPushButton("Classify")
        self.classify_btn.clicked.connect(self._on_classify)
        row.addWidget(self.classify_btn)
        self.batch_btn = QPushButton("Batch Inference")
        self.batch_btn.clicked.connect(self._on_batch)
        row.addWidget(self.batch_btn)
        v.addLayout(row)

        # Results
        v.addWidget(QLabel("<b>Results</b>"))
        self.results_label = QLabel("Best: —")
        v.addWidget(self.results_label)
        self.bars_layout = QVBoxLayout()
        v.addLayout(self.bars_layout)
        self.latency_label = QLabel("Latency: —")
        v.addWidget(self.latency_label)

        # Batch summary table
        v.addWidget(QLabel("<b>Batch Summary</b>"))
        self.batch_table = QTableWidget(0, 3)
        self.batch_table.setHorizontalHeaderLabels(["Image", "Label", "Prob"])
        v.addWidget(self.batch_table)

        if not _torch_available():
            warn = QLabel("PyTorch is not installed. Inference is unavailable.")
            warn.setStyleSheet("color: #b00; font-weight: bold;")
            v.addWidget(warn)
            self.classify_btn.setEnabled(False)
            self.batch_btn.setEnabled(False)

        self._refresh_models_combo()
        # Use a dedicated QLabel for status messages so we can render rich HTML
        # (e.g. highlighted class names).  stretch=1 makes it fill the bar.
        self._status_msg = QLabel("Ready")
        self.statusBar().addWidget(self._status_msg, 1)

    # ---- Model registry helpers ------------------------------------------

    def _refresh_models_combo(self) -> None:
        """Populate model_combo from the local ModelRegistry (SQLite)."""
        self.model_combo.clear()
        try:
            from clipworkbench.models.registry import ModelRegistry

            models = ModelRegistry().list_models()
        except Exception:
            models = []

        if not models:
            self.model_combo.addItem("(no models trained yet — click Admin to train)")
            self.model_combo.setEnabled(False)
            return

        self.model_combo.setEnabled(True)
        for m in models:
            label = (
                f"{m.get('model_id', '?')} — {m.get('strategy', '?')} "
                f"({m.get('num_classes', '?')} cls)"
            )
            self.model_combo.addItem(label, m.get("path", ""))

    # ---- Slots ------------------------------------------------------------

    def _on_select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self._image_path = path
        self.image_path_label.setText(path)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.preview.setPixmap(
                pixmap.scaledToHeight(220, Qt.SmoothTransformation)
            )

    def _on_browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if not path:
            return
        # If this path already exists in combo, just select it.
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == path:
                self.model_combo.setCurrentIndex(i)
                return
        # Otherwise, insert it as a manual entry at the top.
        self.model_combo.insertItem(0, f"(manual) {path}", path)
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setEnabled(True)

    def _on_mode_changed(self) -> None:
        finetuned = self.rb_finetuned.isChecked()
        self.model_combo.setEnabled(finetuned and self.model_combo.count() > 0)
        self.refresh_models_btn.setEnabled(finetuned)
        self.browse_model_btn.setEnabled(finetuned)
        self.labels_edit.setEnabled(not finetuned)

    def _resolve_mode(self) -> tuple[str, str | None, list[str] | None]:
        """Return (strategy, model_path, candidate_labels) for current mode."""
        if self.rb_finetuned.isChecked():
            path = self.model_combo.currentData()
            # currentData() returns None for the placeholder item
            return "linear_probe", path or None, None
        labels = [s.strip() for s in self.labels_edit.text().split(",") if s.strip()]
        return "zero_shot", None, labels

    def _on_classify(self) -> None:
        if not _torch_available():
            QMessageBox.warning(self, "Unavailable", "PyTorch is not installed.")
            return
        if not self._image_path:
            QMessageBox.warning(self, "No image", "Select an image first.")
            return
        strategy, model_path, labels = self._resolve_mode()
        if strategy == "zero_shot" and not labels:
            QMessageBox.warning(self, "No labels", "Enter zero-shot labels.")
            return
        if strategy != "zero_shot" and not model_path:
            QMessageBox.warning(self, "No model", "Enter a model path.")
            return

        self.classify_btn.setEnabled(False)
        self._status_msg.setText("Running inference...")
        self._infer_thread = InferenceThread(
            self._image_path, model_path, labels, strategy
        )
        self._infer_thread.result_ready.connect(self._on_infer_result)
        self._infer_thread.error.connect(self._on_infer_error)
        self._infer_thread.start()

    def _on_infer_result(self, result: dict) -> None:
        top_k = result.get("top_k", [])
        best = result.get("best_label", "—")
        prob = float(result.get("best_probability", 0.0))
        latency = float(result.get("latency_ms", 0.0))
        self.results_label.setText(
            f"Best: <span style='color:#f5d142;font-weight:600'>{best}</span> ({prob:.2%})"
        )
        self.latency_label.setText(f"Latency: {latency:.1f} ms")

        # Clear previous bars
        while self.bars_layout.count():
            item = self.bars_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for i, entry in enumerate(top_k):
            label = entry.get("label", "")
            p = float(entry.get("probability", 0.0))
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setFormat(f"{label} — {p:.2%}")
            bar.setValue(int(p * 10000))
            chunk_color = "#2a7" if i == 0 else "#48f"
            bar.setStyleSheet(
                "QProgressBar {"
                "  border: 1px solid #888;"
                "  border-radius: 3px;"
                "  text-align: left;"
                "  padding: 1px;"
                "  height: 20px;"
                "}"
                f"QProgressBar::chunk {{ background: {chunk_color}; border-radius: 2px; }}"
            )
            self.bars_layout.addWidget(bar)

        self.classify_btn.setEnabled(True)
        self._status_msg.setText(
            f"Inference Result: "
            f"<span style='color:#f5d142;font-weight:400'> {best} "
            f"is recognized by your Fine-tuned CLIP model.</span>"
        )

    def _on_infer_error(self, msg: str) -> None:
        self.classify_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self._status_msg.setText("Inference failed")
        QMessageBox.critical(self, "Inference error", msg)

    def _on_batch(self) -> None:
        if not _torch_available():
            QMessageBox.warning(self, "Unavailable", "PyTorch is not installed.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        strategy, model_path, labels = self._resolve_mode()
        if strategy == "zero_shot" and not labels:
            QMessageBox.warning(self, "No labels", "Enter zero-shot labels.")
            return
        if strategy != "zero_shot" and not model_path:
            QMessageBox.warning(self, "No model", "Enter a model path.")
            return

        self.batch_table.setRowCount(0)
        self.batch_btn.setEnabled(False)
        self._status_msg.setText("Running batch inference...")
        self._batch_thread = BatchInferenceThread(
            folder, model_path, labels, strategy
        )
        self._batch_thread.row_ready.connect(self._on_batch_row)
        self._batch_thread.finished_ok.connect(self._on_batch_done)
        self._batch_thread.error.connect(self._on_infer_error)
        self._batch_thread.start()

    def _on_batch_row(self, name: str, label: str, prob: float) -> None:
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(name))
        self.batch_table.setItem(row, 1, QTableWidgetItem(label))
        self.batch_table.setItem(row, 2, QTableWidgetItem(f"{prob:.4f}"))

    def _on_batch_done(self, done: int, total: int) -> None:
        self.batch_btn.setEnabled(True)
        self._status_msg.setText(f"Batch complete: {done}/{total} images")


def main() -> None:
    """Entry point for running the Inference window standalone."""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    icon = _resolve_icon_path()
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))
    window = AppMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()