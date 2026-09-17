"""Custom exceptions for CLIPWorkbench."""

from __future__ import annotations


class ClipWorkbenchError(Exception):
    """Base exception for all CLIPWorkbench errors."""


class ModelNotFoundError(ClipWorkbenchError):
    """Raised when a model path does not exist or is not loadable."""


class DatasetError(ClipWorkbenchError):
    """Raised when the dataset is empty, malformed, or unreadable."""


class TrainingError(ClipWorkbenchError):
    """Raised when training fails (forward/backward/optimizer issues)."""


class ExportError(ClipWorkbenchError):
    """Raised when ONNX export or quantization fails."""


class ConfigError(ClipWorkbenchError):
    """Raised when configuration is invalid or missing."""
