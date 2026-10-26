"""Explicit device and precision selection for local transformer inference."""

from typing import Any

from settings import RAGSettings


def runtime(settings: RAGSettings, precision: str | None = None) -> tuple[str, Any]:
    """Device and dtype; `precision` overrides model_precision for one model (CPU is fp32)."""
    import torch

    device = settings.model_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU retry is forbidden")
    if not device.startswith("cuda"):
        return device, torch.float32
    precision = precision or settings.model_precision
    if precision == "auto":
        with torch.cuda.device(device):
            precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return device, getattr(torch, precision)
