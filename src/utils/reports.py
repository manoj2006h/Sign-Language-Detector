"""Report writing helpers: atomic JSON, timestamps, and environment capture."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def write_json(path: str | Path, payload: dict) -> Path:
    """Atomically write a JSON document, creating parent directories."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=resolved.parent,
        prefix=resolved.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, resolved)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return resolved


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_text(path: str | Path, text: str) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def environment_info(device: str | torch.device | None = None) -> dict:
    """Capture software/hardware provenance for a report."""
    info = {
        "timestamp": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device) if device is not None else None,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover - optional dependency
        info["torchvision"] = None
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def markdown_table(rows: list[tuple[str, object]]) -> str:
    """Render a two-column key/value Markdown table."""
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)
