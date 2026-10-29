"""Tensor space.

Boxes are (cx, cy, w, h) normalized to [0, 1] in dataset/checkpoint space, and
(x1, y1, x2, y2) in pixel space after :func:`rescale_bboxes`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
    )


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    x1, y1, x2, y2 = x.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
    )


def _validate_size(size: Sequence[float]) -> tuple[float, float]:
    if len(size) != 2:
        raise ValueError(f"Expected image size as (width, height), got {size!r}")
    width, height = float(size[0]), float(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {size!r}")
    return width, height


def rescale_bboxes(out_bbox: torch.Tensor, size: Sequence[float]) -> torch.Tensor:
    """Convert normalized cxcywh boxes to pixel-space xyxy boxes.

    ``size`` is ``(width, height)`` in pixels. The returned tensor keeps the
    input dtype/device.
    """
    width, height = _validate_size(size)
    scale = torch.tensor(
        [width, height, width, height], dtype=out_bbox.dtype, device=out_bbox.device
    )
    return box_cxcywh_to_xyxy(out_bbox) * scale


def clip_boxes(boxes: torch.Tensor, size: Sequence[float]) -> torch.Tensor:
    """Clamp pixel-space xyxy boxes to the image rectangle."""
    width, height = _validate_size(size)
    return torch.stack(
        (
            boxes[..., 0].clamp(min=0.0, max=width),
            boxes[..., 1].clamp(min=0.0, max=height),
            boxes[..., 2].clamp(min=0.0, max=width),
            boxes[..., 3].clamp(min=0.0, max=height),
        ),
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute the area of pixel-space xyxy boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU and union area between two sets of xyxy boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    eps = 1e-7
    return inter / (union + eps), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU for two sets of pixel-space xyxy boxes."""
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    c_area = wh[:, :, 0] * wh[:, :, 1]
    eps = 1e-7
    return iou - (c_area - union) / (c_area + eps)


def stacker(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """Collate images while keeping variable-length targets as a list."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images, dim=0), targets
