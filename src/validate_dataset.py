"""Non-destructive dataset validation.

Scans image/label pairs, parses every annotation, and reports class counts and
problems. It never modifies, moves, or rewrites the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import DatasetError, DETRData
from utils.setup import ConfigError, get_classes


def validate_dataset(path: str | Path, classes: list[str]) -> dict:
    dataset = DETRData(path, train=False, classes=classes, verbose=False)
    image_errors = dataset.validate_images()
    if image_errors:
        preview = "\n  - ".join(image_errors[:20])
        more = (
            ""
            if len(image_errors) <= 20
            else f"\n  ... and {len(image_errors) - 20} more"
        )
        raise DatasetError(
            f"Image validation failed for {dataset.path}:\n  - {preview}{more}"
        )
    return dataset.summary()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="Dataset split directory (repeatable; default: data/train and data/test)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dirs = args.data_dirs or [Path("data/train"), Path("data/test")]
    try:
        classes = get_classes()
    except ConfigError as exc:
        print(f"INVALID configuration\n{exc}")
        return 1

    failed = False
    for path in data_dirs:
        try:
            summary = validate_dataset(path, classes)
        except (DatasetError, ConfigError) as exc:
            failed = True
            print(f"INVALID {path}\n{exc}")
            continue
        print(
            f"OK {path}: {summary['samples']} samples, {summary['boxes']} boxes, "
            f"classes={summary['class_counts']}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
