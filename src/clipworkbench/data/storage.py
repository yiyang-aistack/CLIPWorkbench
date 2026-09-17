"""SQLite storage for model metadata and inference history.

Uses stdlib sqlite3 (no external dependencies).
Stores: model versions, prediction logs, human corrections.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class Storage:
    """Thread-safe SQLite storage for model metadata and history."""

    def __init__(self, db_path: str | os.PathLike = "clipworkbench.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id    TEXT PRIMARY KEY,
                    path        TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    num_classes INTEGER NOT NULL,
                    class_names TEXT NOT NULL,  -- JSON array
                    created_at  TEXT NOT NULL,
                    metrics     TEXT             -- JSON dict
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path  TEXT,
                    model_id    TEXT,
                    top_label   TEXT,
                    probability REAL,
                    full_result TEXT,  -- JSON
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path  TEXT,
                    model_id    TEXT,
                    original_label TEXT,
                    corrected_label TEXT,
                    reason      TEXT,
                    created_at  TEXT NOT NULL
                );
            """)
            conn.commit()
            conn.close()

    # --- Model operations ---

    def save_model_info(
        self,
        model_id: str,
        path: str,
        strategy: str,
        num_classes: int,
        class_names: list[str],
        metrics: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO models
                   (model_id, path, strategy, num_classes, class_names, created_at, metrics)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_id,
                    path,
                    strategy,
                    num_classes,
                    json.dumps(class_names, ensure_ascii=False),
                    datetime.now().isoformat(),
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()
            conn.close()

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            conn.close()
            if row is None:
                return None
            return {
                "model_id": row["model_id"],
                "path": row["path"],
                "strategy": row["strategy"],
                "num_classes": row["num_classes"],
                "class_names": json.loads(row["class_names"]),
                "created_at": row["created_at"],
                "metrics": json.loads(row["metrics"] or "{}"),
            }

    def delete_model_info(self, model_id: str) -> bool:
        """Delete a model record from storage. Returns True if deleted."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            return [
                {
                    "model_id": r["model_id"],
                    "path": r["path"],
                    "strategy": r["strategy"],
                    "num_classes": r["num_classes"],
                    "class_names": json.loads(r["class_names"]),
                    "created_at": r["created_at"],
                    "metrics": json.loads(r["metrics"] or "{}"),
                }
                for r in rows
            ]

    # --- Prediction logging ---

    def log_prediction(
        self,
        image_path: str,
        model_id: str,
        top_label: str,
        probability: float,
        full_result: dict[str, Any],
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO predictions
                   (image_path, model_id, top_label, probability, full_result, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    image_path,
                    model_id,
                    top_label,
                    probability,
                    json.dumps(full_result, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()

    # --- Correction tracking (for closed-loop learning) ---

    def add_correction(
        self,
        image_path: str,
        model_id: str,
        original_label: str,
        corrected_label: str,
        reason: str = "",
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO corrections
                   (image_path, model_id, original_label, corrected_label, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    image_path,
                    model_id,
                    original_label,
                    corrected_label,
                    reason,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()

    def get_correction_count(self) -> int:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) as n FROM corrections").fetchone()
            conn.close()
            return row["n"] if row else 0
