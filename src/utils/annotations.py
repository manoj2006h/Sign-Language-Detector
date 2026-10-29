"""YOLO annotation parsing and validation.

Each non-blank line must contain exactly five fields::

    class_id center_x center_y width height

Coordinates are normalized to [0, 1]. Boxes are validated before they reach
Albumentations or the DETR loss so malformed labels fail with a file and line
number instead of an obscure tensor error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
_BOUND_EPS = 1e-6


class AnnotationError(ValueError):
    """Raised when an annotation file or line is invalid."""


def _source_name(source: str | Path) -> str:
    return str(source)


def parse_yolo_annotations(
    text: str, num_classes: int, source: str | Path = "<annotations>"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO text into ``(labels, boxes)`` tensors.

    Blank lines are ignored. Empty text is valid and yields zero targets.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    source_name = _source_name(source)
    labels: list[int] = []
    boxes: list[list[float]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise AnnotationError(
                f"{source_name}: line {line_number} has {len(fields)} fields; "
                "expected 'class_id center_x center_y width height'"
            )

        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise AnnotationError(
                f"{source_name}: line {line_number} class id is not an integer: "
                f"{fields[0]!r}"
            ) from exc
        if class_id < 0 or class_id >= num_classes:
            raise AnnotationError(
                f"{source_name}: line {line_number} references class {class_id}, "
                f"but the configured classes are 0..{num_classes - 1}"
            )

        try:
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise AnnotationError(
                f"{source_name}: line {line_number} box values must be numbers"
            ) from exc

        if not all(
            math.isfinite(value)
            for value in (center_x, center_y, width, height)
        ):
            raise AnnotationError(
                f"{source_name}: line {line_number} box contains NaN or infinity"
            )
        if width <= 0 or height <= 0:
            raise AnnotationError(
                f"{source_name}: line {line_number} width and height must be positive; "
                f"got width={width}, height={height}"
            )

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        if (
            center_x < -_BOUND_EPS
            or center_x > 1 + _BOUND_EPS
            or center_y < -_BOUND_EPS
            or center_y > 1 + _BOUND_EPS
            or x1 < -_BOUND_EPS
            or y1 < -_BOUND_EPS
            or x2 > 1 + _BOUND_EPS
            or y2 > 1 + _BOUND_EPS
        ):
            raise AnnotationError(
                f"{source_name}: line {line_number} box is outside normalized "
                f"image bounds: center=({center_x}, {center_y}), "
                f"size=({width}, {height})"
            )

        labels.append(class_id)
        boxes.append([center_x, center_y, width, height])

    label_tensor = torch.tensor(labels, dtype=torch.long)
    box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return label_tensor, box_tensor


def parse_yolo_file(
    path: str | Path, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and parse a YOLO label file."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AnnotationError(
            f"Could not read annotation file {resolved}: {exc}"
        ) from exc
    return parse_yolo_annotations(text, num_classes, source=resolved)


def find_image_for_stem(images_dir: str | Path, stem: str) -> Path | None:
    """Find an image by stem, accepting common case-insensitive extensions."""
    root = Path(images_dir)
    if not root.is_dir():
        return None
    candidates: Iterable[Path] = root.iterdir()
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stem == stem
            and candidate.suffix.lower() in IMAGE_EXTENSIONS
        ):
            return candidate
    return None
