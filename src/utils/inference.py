"""Shared inference postprocessing and BGR frame preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np
import torch

from utils.boxes import clip_boxes, rescale_bboxes


def average_or_none(total: float, count: int) -> float | None:
    """Avoid division by zero for empty loaders/datasets."""
    return total / count if count > 0 else None


def count_detections(
    output: dict[str, torch.Tensor], threshold: float = 0.8
) -> int:
    """Count queries whose best non-background probability exceeds a threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    logits = output.get("pred_logits")
    if logits is None or logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("Expected pred_logits of shape (B, Q, C+1)")
    probabilities = logits.softmax(-1)[..., :-1]
    return int((probabilities.max(-1).values > threshold).sum().item())


def compute_loss_total(losses: dict, weight_dict: dict) -> torch.Tensor:
    """Combine DETR loss components with the configured weights."""
    return (
        losses["labels"]["loss_ce"] * weight_dict["class_weighting"]
        + losses["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"]
        + losses["boxes"]["loss_giou"] * weight_dict["giou_weighting"]
    )


def preprocess_frame(frame: np.ndarray, transform: A.Compose) -> torch.Tensor:
    """Convert a BGR OpenCV frame to the shared RGB eval pipeline.

    Returns a ``(1, 3, H, W)`` float tensor ready for ``model(...).to(device)``.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Camera frame is missing or is not a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Camera frame must be HxWx3 BGR, got shape {getattr(frame, 'shape', None)}"
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(image=rgb)["image"]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Inference transform must return a torch.Tensor")
    return tensor.unsqueeze(0)


def postprocess_detections(
    output: dict[str, torch.Tensor],
    image_sizes: Sequence[tuple[int, int]],
    class_names: Sequence[str],
    threshold: float = 0.8,
) -> list[list[dict]]:
    """Convert model output to per-image detections in original pixel space.

    ``image_sizes`` entries are ``(width, height)`` for the actual input frames.
    Boxes are returned in clipped xyxy pixel coordinates.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if "pred_logits" not in output or "pred_boxes" not in output:
        raise ValueError("Model output must contain 'pred_logits' and 'pred_boxes'")

    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    if logits.ndim != 3 or boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(
            "Expected pred_logits (B, Q, C+1) and pred_boxes (B, Q, 4)"
        )
    num_classes = logits.shape[-1] - 1
    if num_classes != len(class_names):
        raise ValueError(
            f"Model predicts {num_classes} classes, but {len(class_names)} class "
            "names were supplied"
        )
    if len(image_sizes) != logits.shape[0]:
        raise ValueError(
            f"Got {len(image_sizes)} image sizes for batch size {logits.shape[0]}"
        )

    probabilities = logits.softmax(-1)[..., :-1]
    max_probs, max_classes = probabilities.max(-1)
    results: list[list[dict]] = []
    for batch_index, size in enumerate(image_sizes):
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Image size must be positive, got {size!r}")
        keep = max_probs[batch_index] > threshold
        query_indices = torch.nonzero(keep, as_tuple=False).flatten()
        if query_indices.numel() == 0:
            results.append([])
            continue

        selected_boxes = boxes[batch_index, query_indices]
        scaled = clip_boxes(rescale_bboxes(selected_boxes, (width, height)), (width, height))
        selected_classes = max_classes[batch_index, query_indices]
        selected_probs = max_probs[batch_index, query_indices]

        image_detections = []
        for box, class_index, probability in zip(scaled, selected_classes, selected_probs):
            class_id = int(class_index.item())
            if class_id < 0 or class_id >= len(class_names):
                raise ValueError(
                    f"Model produced class index {class_id}, outside configured "
                    f"classes 0..{len(class_names) - 1}"
                )
            image_detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": float(probability.item()),
                    "bbox": [float(value) for value in box.tolist()],
                }
            )
        results.append(image_detections)
    return results
