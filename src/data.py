"""Dataset and preprocessing pipeline.

Training/evaluation and webcam inference share the same resize, normalization,
and RGB ordering. See ``build_train_transform`` and ``build_eval_transform``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from utils.annotations import (
    AnnotationError,
    IMAGE_EXTENSIONS,
    parse_yolo_file,
)
from utils.setup import get_classes

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224


class DatasetError(RuntimeError):
    """Raised when a dataset path, image, or annotation cannot be used."""


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(format="yolo", label_fields=["class_labels"])


def _image_ops(image_size: int, train: bool) -> list:
    ops: list = [A.Resize(500, 500)]
    if train:
        ops.append(A.RandomCrop(width=224, height=224, p=0.33))
    ops.append(A.Resize(image_size, image_size))
    if train:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5,
                    p=0.5,
                ),
            ]
        )
    ops.extend([A.Normalize(mean=MEAN, std=STD), A.ToTensorV2()])
    return ops


def build_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE, *, with_bboxes: bool = True
) -> A.Compose:
    """Build the training transform (random crop/flip/color jitter)."""
    kwargs = {"bbox_params": _bbox_params()} if with_bboxes else {}
    return A.Compose(_image_ops(image_size, train=True), **kwargs)


def build_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE, *, with_bboxes: bool = True
) -> A.Compose:
    """Build the deterministic evaluation/inference transform."""
    kwargs = {"bbox_params": _bbox_params()} if with_bboxes else {}
    return A.Compose(_image_ops(image_size, train=False), **kwargs)


class DETRData(Dataset):
    """YOLO-format dataset with fail-fast annotation validation.

    Images and labels must share a stem, e.g.
    ``images/foo.jpg`` and ``labels/foo.txt``. An empty label file is a valid
    image with zero objects.
    """

    def __init__(
        self,
        path: str | Path,
        train: bool = True,
        *,
        classes: Sequence[str] | None = None,
        indices: Sequence[int] | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        transform: A.Compose | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.labels_path = self.path / "labels"
        self.images_path = self.path / "images"
        self.train = bool(train)
        self.image_size = int(image_size)
        self.classes = list(classes) if classes is not None else get_classes()
        if not self.classes:
            raise DatasetError("The class list is empty")
        self.num_classes = len(self.classes)

        if not self.images_path.is_dir():
            raise DatasetError(f"Images directory not found: {self.images_path}")
        if not self.labels_path.is_dir():
            raise DatasetError(f"Labels directory not found: {self.labels_path}")

        self.logger = None
        if verbose:
            from utils.logger import get_logger
            from utils.rich_handlers import DataLoaderHandler

            self.logger = get_logger("data_loader")
            DataLoaderHandler().log_dataset_stats(
                {
                    "Dataset Path": str(self.path),
                    "Mode": "Training" if self.train else "Evaluation",
                    "Images Path": str(self.images_path),
                    "Labels Path": str(self.labels_path),
                }
            )

        label_paths = sorted(
            path for path in self.labels_path.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
        )
        all_label_stems = {path.stem for path in label_paths}
        total_before_subset = len(label_paths)
        if indices is not None:
            selected: list[Path] = []
            for index in indices:
                if not isinstance(index, (int, np.integer)):
                    raise TypeError(f"Dataset indices must be integers, got {index!r}")
                if index < 0 or index >= len(label_paths):
                    raise IndexError(
                        f"Dataset index {index} is out of range for {len(label_paths)} samples"
                    )
                selected.append(label_paths[int(index)])
            label_paths = selected

        image_map: dict[str, Path] = {}
        for candidate in sorted(self.images_path.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if candidate.stem in image_map:
                raise DatasetError(
                    f"Duplicate image stems for {candidate.stem!r}: "
                    f"{image_map[candidate.stem]} and {candidate}"
                )
            image_map[candidate.stem] = candidate
        self.samples: list[tuple[Path, Path, torch.Tensor, torch.Tensor]] = []
        errors: list[str] = []
        for label_path in label_paths:
            image_path = image_map.get(label_path.stem)
            if image_path is None:
                errors.append(
                    f"{label_path}: no matching image with stem {label_path.stem!r} in {self.images_path}"
                )
                continue
            try:
                labels, boxes = parse_yolo_file(label_path, self.num_classes)
            except AnnotationError as exc:
                errors.append(str(exc))
                continue
            self.samples.append((image_path, label_path, labels, boxes))

        for stem, image_path in sorted(image_map.items()):
            if stem not in all_label_stems:
                errors.append(
                    f"{image_path}: no matching annotation file "
                    f"{self.labels_path / (stem + '.txt')}. Create an empty "
                    "label file for an image with no objects."
                )

        if errors:
            preview = "\n  - ".join(errors[:20])
            more = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
            raise DatasetError(
                f"Dataset validation failed for {self.path}:\n  - {preview}{more}"
            )

        self.transform = transform or (
            build_train_transform(self.image_size)
            if self.train
            else build_eval_transform(self.image_size)
        )
        # A deterministic fallback is used if random augmentation repeatedly
        # drops every box. It keeps the sample usable without returning stale
        # unnormalized boxes.
        self.fallback_transform = (
            build_eval_transform(self.image_size)
            if self.train
            else self.transform
        )
        self.image_transform = (
            build_train_transform(self.image_size, with_bboxes=False)
            if self.train
            else build_eval_transform(self.image_size, with_bboxes=False)
        )

        if self.logger is not None:
            self.logger.data(
                f"Loaded {len(self.samples)} of {total_before_subset} samples from {self.path}"
            )
            if not self.samples:
                self.logger.warning(
                    f"No annotation files found in {self.labels_path}; "
                    "the dataset is empty."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                return np.asarray(rgb)
        except (UnidentifiedImageError, OSError) as exc:
            raise DatasetError(f"Could not read image {image_path}: {exc}") from exc

    def safe_transform(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        labels: list[int],
        *,
        max_attempts: int = 10,
    ) -> dict:
        """Apply the transform, retrying only when augmentation drops all boxes.

        Random crops can legitimately remove every box. Any other transform
        failure is raised immediately instead of being silently replaced by the
        deterministic fallback.
        """
        for _ in range(max_attempts):
            try:
                transformed = self.transform(
                    image=image, bboxes=bboxes, class_labels=labels
                )
            except Exception as exc:
                raise DatasetError(
                    f"Configured transform failed: {exc}"
                ) from exc
            if not isinstance(transformed, dict) or "image" not in transformed:
                raise DatasetError(
                    "Configured transform returned an invalid result; expected a "
                    "dictionary with an 'image' entry"
                )
            if "bboxes" not in transformed:
                raise DatasetError(
                    "Configured transform returned an invalid result; expected a "
                    "'bboxes' entry"
                )
            if len(bboxes) == 0 or len(transformed["bboxes"]) > 0:
                return transformed

        try:
            transformed = self.fallback_transform(
                image=image, bboxes=bboxes, class_labels=labels
            )
        except Exception as exc:
            raise DatasetError(
                "Could not transform sample with random augmentation or the "
                f"deterministic fallback after {max_attempts} attempts: {exc}"
            ) from exc

        if len(transformed["bboxes"]) == 0:
            raise DatasetError(
                "Deterministic fallback unexpectedly removed all boxes from "
                "the sample"
            )
        if self.logger is not None:
            self.logger.warning(
                "Random augmentation dropped all boxes; used the deterministic "
                "resize for one sample."
            )
        return transformed

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Dataset index must be an integer, got {idx!r}")
        index = int(idx)
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Dataset index {idx} is out of range")

        image_path, label_path, labels, boxes = self.samples[index]
        image = self._read_image(image_path)

        if len(labels) == 0:
            try:
                augmented_image = self.image_transform(image=image)["image"]
            except Exception as exc:
                raise DatasetError(
                    f"Could not transform empty annotation sample {label_path}: {exc}"
                ) from exc
            target = {
                "labels": torch.empty(0, dtype=torch.long),
                "boxes": torch.empty(0, 4, dtype=torch.float32),
            }
            return augmented_image, target

        try:
            augmented = self.safe_transform(
                image=image,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
        except DatasetError as exc:
            raise DatasetError(f"{label_path}: {exc}") from exc

        augmented_image = augmented["image"]
        if not isinstance(augmented_image, torch.Tensor):
            raise DatasetError(
                f"Transform for {image_path} returned {type(augmented_image).__name__}, "
                "expected a tensor"
            )
        augmented_boxes = np.asarray(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
        augmented_labels = np.asarray(augmented["class_labels"], dtype=np.int64).reshape(-1)
        target = {
            "labels": torch.from_numpy(augmented_labels.copy()).long(),
            "boxes": torch.from_numpy(augmented_boxes.copy()).float(),
        }
        return augmented_image, target

    def summary(self) -> dict:
        """Return a small, non-destructive dataset summary."""
        class_counts = [0] * self.num_classes
        box_count = 0
        for _, _, labels, boxes in self.samples:
            box_count += len(boxes)
            for label in labels.tolist():
                class_counts[int(label)] += 1
        return {
            "path": str(self.path),
            "samples": len(self.samples),
            "boxes": box_count,
            "class_counts": {
                name: count for name, count in zip(self.classes, class_counts)
            },
        }

    def validate_images(self) -> list[str]:
        """Return actionable errors for images the loader cannot decode.

        This is intentionally separate from ``__init__`` so training does not
        open every image just to build an index. The dataset validator calls it
        explicitly.
        """
        errors: list[str] = []
        for image_path, _, _, _ in self.samples:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    if width <= 0 or height <= 0:
                        raise OSError(
                            f"image has invalid dimensions {width}x{height}"
                        )
                    image.verify()
            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                errors.append(f"Could not read image {image_path}: {exc}")
        return errors


if __name__ == "__main__":
    train_dataset = DETRData("data/train", train=True)
    eval_dataset = DETRData("data/test", train=False)
    print(train_dataset.summary())
    print(eval_dataset.summary())
