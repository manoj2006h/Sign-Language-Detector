"""Shared test helpers. Adds the source directory to sys.path for flat imports."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_dataset(
    root: Path,
    split: str,
    samples: list[tuple[str, str, tuple[int, int], str]],
) -> Path:
    """Create a tiny YOLO-style dataset.

    Each sample is ``(stem, color, (width, height), label_text)``.
    Returns the split directory (``root / split``).
    """
    split_dir = root / split
    images = split_dir / "images"
    labels = split_dir / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for stem, color, size, label_text in samples:
        Image.new("RGB", size, color).save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    return split_dir
