"""Explicit device selection.

An unavailable requested device is an error; it is never silently replaced by
CPU.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but CUDA is not available. "
                "Use --device cpu or install a CUDA-capable PyTorch build."
            )
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid CUDA device {device!r}") from exc
        if resolved.type != "cuda":
            raise ValueError(f"Invalid CUDA device {device!r}")
        if resolved.index is not None:
            available = torch.cuda.device_count()
            if resolved.index < 0 or resolved.index >= available:
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is out of range; "
                    f"{available} CUDA device(s) available."
                )
        return resolved
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {device!r}; use auto, cpu, or cuda")


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    """Move labels/boxes in a DETR target list to a device."""
    return [
        {
            "labels": target["labels"].to(device=device, dtype=torch.long),
            "boxes": target["boxes"].to(device=device, dtype=torch.float32),
        }
        for target in targets
    ]
