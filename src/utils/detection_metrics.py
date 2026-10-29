"""Object-detection metrics with explicit threshold conventions.

Implements per-class average precision with greedy confidence-ordered matching
and 101-point interpolated precision, plus micro precision/recall and counts.
The conventions follow COCO-style detection evaluation:

* Boxes are normalized ``(x1, y1, x2, y2)`` in the same coordinate space.
* A prediction matches a ground-truth box only when the classes are equal and
  IoU >= the threshold. Each ground-truth box can be matched once; a second
  overlapping prediction is a duplicate and therefore a false positive.
* AP at one IoU threshold uses all predictions (no confidence cutoff), sorted
  by descending confidence. Duplicate/false predictions lower precision at
  the recall levels where they occur.
* Classes with no ground-truth boxes have *undefined* AP (``None``), not zero.
  They are excluded from the mean AP and listed explicitly.
* mAP@0.5:0.95 is the mean of AP across IoU thresholds 0.50, 0.55, ..., 0.95,
  computed per class and then averaged over classes, never over batches.

Confidence-thresholded precision/recall is reported separately at
``match_iou_threshold``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_BOUND_EPS = 1e-6


class DetectionMetricsError(ValueError):
    """Raised for malformed metric inputs."""


def _validate_box(box, source: str) -> np.ndarray:
    try:
        array = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DetectionMetricsError(f"{source} bbox is not numeric: {box!r}") from exc
    if array.shape != (4,):
        raise DetectionMetricsError(
            f"{source} bbox must have 4 values, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DetectionMetricsError(f"{source} bbox contains NaN or infinity")
    x1, y1, x2, y2 = array
    if x2 < x1 - _BOUND_EPS or y2 < y1 - _BOUND_EPS:
        raise DetectionMetricsError(
            f"{source} bbox has negative width/height: {array.tolist()}"
        )
    if (
        x1 < -_BOUND_EPS
        or y1 < -_BOUND_EPS
        or x2 > 1 + _BOUND_EPS
        or y2 > 1 + _BOUND_EPS
    ):
        raise DetectionMetricsError(
            f"{source} bbox is outside normalized [0, 1] bounds: {array.tolist()}"
        )
    return np.clip(array, 0.0, 1.0)


def validate_inputs(
    ground_truth: Sequence[Sequence[Mapping]],
    predictions: Sequence[Sequence[Mapping]],
    class_names: Sequence[str],
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Validate and normalize metric inputs into internal dicts."""
    if not class_names:
        raise DetectionMetricsError("class_names must not be empty")
    if len(ground_truth) != len(predictions):
        raise DetectionMetricsError(
            f"ground_truth has {len(ground_truth)} images but predictions has "
            f"{len(predictions)}"
        )
    num_classes = len(class_names)
    normalized_gt: list[list[dict]] = []
    normalized_pred: list[list[dict]] = []
    for image_index, (image_gt, image_pred) in enumerate(zip(ground_truth, predictions)):
        gt_items = []
        for object_index, item in enumerate(image_gt):
            label = item.get("class_id")
            if not isinstance(label, (int, np.integer)) or isinstance(label, bool):
                raise DetectionMetricsError(
                    f"image {image_index} GT {object_index} class_id must be an integer"
                )
            label = int(label)
            if label < 0 or label >= num_classes:
                raise DetectionMetricsError(
                    f"image {image_index} GT {object_index} class_id {label} is "
                    f"outside 0..{num_classes - 1}"
                )
            gt_items.append(
                {
                    "class_id": label,
                    "bbox": _validate_box(
                        item.get("bbox"), f"image {image_index} GT {object_index}"
                    ),
                }
            )

        pred_items = []
        for object_index, item in enumerate(image_pred):
            label = item.get("class_id")
            if not isinstance(label, (int, np.integer)) or isinstance(label, bool):
                raise DetectionMetricsError(
                    f"image {image_index} prediction {object_index} class_id must "
                    "be an integer"
                )
            label = int(label)
            if label < 0 or label >= num_classes:
                raise DetectionMetricsError(
                    f"image {image_index} prediction {object_index} class_id "
                    f"{label} is outside 0..{num_classes - 1}"
                )
            try:
                score = float(item.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise DetectionMetricsError(
                    f"image {image_index} prediction {object_index} confidence "
                    "must be numeric"
                ) from exc
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise DetectionMetricsError(
                    f"image {image_index} prediction {object_index} confidence "
                    f"must be in [0, 1], got {score}"
                )
            pred_items.append(
                {
                    "class_id": label,
                    "confidence": score,
                    "bbox": _validate_box(
                        item.get("bbox"),
                        f"image {image_index} prediction {object_index}",
                    ),
                }
            )
        normalized_gt.append(gt_items)
        normalized_pred.append(pred_items)
    return normalized_gt, normalized_pred


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for two ``(N, 4)`` arrays of xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    width = np.clip(x2 - x1, 0.0, None)
    height = np.clip(y2 - y1, 0.0, None)
    intersection = width * height
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _match_image(
    gt_items: list[dict], pred_items: list[dict], iou_threshold: float
) -> list[tuple[float, bool]]:
    """Match predictions to GT for one image, returning (score, is_tp) rows."""
    if not pred_items:
        return []
    if not gt_items:
        return [(item["confidence"], False) for item in pred_items]

    gt_boxes = np.stack([item["bbox"] for item in gt_items])
    pred_boxes = np.stack([item["bbox"] for item in pred_items])
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = sorted(
        range(len(pred_items)),
        key=lambda index: (-pred_items[index]["confidence"], index),
    )
    used: set[int] = set()
    rows: list[tuple[float, bool]] = []
    for pred_index in order:
        best_gt = -1
        best_iou = -1.0
        for gt_index in range(len(gt_items)):
            if gt_index in used:
                continue
            value = float(ious[pred_index, gt_index])
            if value > best_iou:
                best_iou = value
                best_gt = gt_index
        if best_gt >= 0 and best_iou >= iou_threshold:
            used.add(best_gt)
            rows.append((pred_items[pred_index]["confidence"], True))
        else:
            rows.append((pred_items[pred_index]["confidence"], False))
    return rows


def average_precision(rows: list[tuple[float, bool]], num_gt: int) -> float | None:
    """101-point interpolated AP from (score, is_tp) rows."""
    if num_gt <= 0:
        return None
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda row: -row[0])
    true_positive = 0
    false_positive = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, is_tp in ordered:
        if is_tp:
            true_positive += 1
        else:
            false_positive += 1
        recalls.append(true_positive / num_gt)
        precisions.append(true_positive / (true_positive + false_positive))
    recall_array = np.asarray(recalls)
    precision_array = np.asarray(precisions)
    envelope = np.maximum.accumulate(precision_array[::-1])[::-1]
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = []
    for recall_point in recall_points:
        mask = recall_array >= recall_point
        interpolated.append(float(envelope[mask].max()) if mask.any() else 0.0)
    return float(np.mean(interpolated))


def _counts_at_confidence(
    gt: list[list[dict]],
    pred: list[list[dict]],
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict:
    gt_count = 0
    tp = fp = 0
    for image_gt, image_pred in zip(gt, pred):
        class_gt = [item for item in image_gt if item["class_id"] == class_id]
        class_pred = [
            item
            for item in image_pred
            if item["class_id"] == class_id
            and item["confidence"] >= confidence_threshold
        ]
        gt_count += len(class_gt)
        rows = _match_image(class_gt, class_pred, iou_threshold)
        tp += sum(1 for _, is_tp in rows if is_tp)
        fp += sum(1 for _, is_tp in rows if not is_tp)
    fn = gt_count - tp
    return {
        "ground_truth": gt_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def match_image_detections(
    ground_truth: Sequence[Mapping],
    predictions: Sequence[Mapping],
    class_names: Sequence[str],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> dict:
    """Return per-box matching detail for error analysis.

    Only predictions at or above ``confidence_threshold`` participate. Matching
    is class-aware, greedy by confidence, and each GT box can be used once.
    """
    gt, pred = validate_inputs([ground_truth], [predictions], class_names)
    image_gt = gt[0]
    selected = [
        item
        for item in pred[0]
        if item["confidence"] >= confidence_threshold
    ]
    matches: list[dict] = []
    false_positive_indices: list[int] = []
    matched_gt: set[int] = set()
    if selected and image_gt:
        gt_boxes = np.stack([item["bbox"] for item in image_gt])
        pred_boxes = np.stack([item["bbox"] for item in selected])
        ious = iou_matrix(pred_boxes, gt_boxes)
        order = sorted(
            range(len(selected)),
            key=lambda index: (-selected[index]["confidence"], index),
        )
        for pred_index in order:
            candidate_gt = -1
            candidate_iou = -1.0
            for gt_index in range(len(image_gt)):
                if gt_index in matched_gt:
                    continue
                if image_gt[gt_index]["class_id"] != selected[pred_index]["class_id"]:
                    continue
                value = float(ious[pred_index, gt_index])
                if value > candidate_iou:
                    candidate_iou = value
                    candidate_gt = gt_index
            if candidate_gt >= 0 and candidate_iou >= iou_threshold:
                matched_gt.add(candidate_gt)
                matches.append(
                    {
                        "gt_index": candidate_gt,
                        "pred_index": pred_index,
                        "iou": candidate_iou,
                        "class_id": selected[pred_index]["class_id"],
                    }
                )
            else:
                false_positive_indices.append(pred_index)
    else:
        false_positive_indices = list(range(len(selected)))

    return {
        "matches": matches,
        "false_positive_indices": sorted(false_positive_indices),
        "unmatched_ground_truth": [
            index for index in range(len(image_gt)) if index not in matched_gt
        ],
        "ground_truth": image_gt,
        "predictions": selected,
    }


def _confusion(
    gt: list[list[dict]], pred: list[list[dict]], class_names: Sequence[str], iou_threshold: float
) -> dict:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for image_gt, image_pred in zip(gt, pred):
        if not image_gt or not image_pred:
            continue
        gt_boxes = np.stack([item["bbox"] for item in image_gt])
        pred_boxes = np.stack([item["bbox"] for item in image_pred])
        ious = iou_matrix(pred_boxes, gt_boxes)
        for pred_index, pred_item in enumerate(image_pred):
            gt_index = int(np.argmax(ious[pred_index]))
            if float(ious[pred_index, gt_index]) < iou_threshold:
                continue
            gt_class = image_gt[gt_index]["class_id"]
            pred_class = pred_item["class_id"]
            if gt_class != pred_class:
                matrix[class_names[gt_class]][class_names[pred_class]] += 1
    return {gt_name: dict(preds) for gt_name, preds in matrix.items()}


def evaluate_detection(
    ground_truth: Sequence[Sequence[Mapping]],
    predictions: Sequence[Sequence[Mapping]],
    class_names: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    match_iou_threshold: float = 0.5,
) -> dict:
    """Aggregate detection metrics over the whole dataset.

    ``ground_truth[i]`` and ``predictions[i]`` belong to the same image.
    Ground-truth items need ``class_id`` and normalized xyxy ``bbox``.
    Predictions add a ``confidence`` in ``[0, 1]``.
    """
    names = list(class_names)
    gt, pred = validate_inputs(ground_truth, predictions, names)
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds:
        raise DetectionMetricsError("At least one IoU threshold is required")
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise DetectionMetricsError("IoU thresholds must be in (0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise DetectionMetricsError("confidence_threshold must be in [0, 1]")
    if not 0.0 < match_iou_threshold <= 1.0:
        raise DetectionMetricsError("match_iou_threshold must be in (0, 1]")

    num_classes = len(names)
    gt_counts = [0] * num_classes
    for image_gt in gt:
        for item in image_gt:
            gt_counts[item["class_id"]] += 1

    ap_by_threshold: dict[float, dict[str, float | None]] = {}
    per_class_ap: dict[str, list[float]] = defaultdict(list)
    for threshold in thresholds:
        rows_by_class: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for image_gt, image_pred in zip(gt, pred):
            for class_id in range(num_classes):
                class_gt = [item for item in image_gt if item["class_id"] == class_id]
                class_pred = [
                    item for item in image_pred if item["class_id"] == class_id
                ]
                rows_by_class[class_id].extend(
                    _match_image(class_gt, class_pred, threshold)
                )
        threshold_ap: dict[str, float | None] = {}
        for class_id, class_name in enumerate(names):
            value = average_precision(
                rows_by_class[class_id], gt_counts[class_id]
            )
            threshold_ap[class_name] = value
            if value is not None:
                per_class_ap[class_name].append(value)
        ap_by_threshold[threshold] = threshold_ap

    per_class: dict[str, dict] = {}
    for class_id, class_name in enumerate(names):
        counts = _counts_at_confidence(
            gt, pred, class_id, confidence_threshold, match_iou_threshold
        )
        values = per_class_ap.get(class_name, [])
        per_class[class_name] = {
            "class_id": class_id,
            "ap50": ap_by_threshold.get(0.5, {}).get(class_name),
            "ap75": ap_by_threshold.get(0.75, {}).get(class_name),
            "ap": float(np.mean(values)) if values else None,
            "ground_truth": counts["ground_truth"],
            "predictions": sum(
                1
                for image_pred in pred
                for item in image_pred
                if item["class_id"] == class_id
                and item["confidence"] >= confidence_threshold
            ),
            "true_positives": counts["true_positives"],
            "false_positives": counts["false_positives"],
            "false_negatives": counts["false_negatives"],
            "precision": counts["precision"],
            "recall": counts["recall"],
        }

    defined = [name for name, values in per_class_ap.items() if values]
    map50_values = [
        ap_by_threshold[0.5][name]
        for name in names
        if ap_by_threshold.get(0.5, {}).get(name) is not None
    ]
    map_values = [
        float(np.mean(per_class_ap[name])) for name in defined
    ]
    total_gt = sum(gt_counts)
    micro_tp = sum(item["true_positives"] for item in per_class.values())
    micro_fp = sum(item["false_positives"] for item in per_class.values())
    micro_fn = sum(item["false_negatives"] for item in per_class.values())

    return {
        "schema_version": 1,
        "conventions": {
            "box_format": "normalized xyxy",
            "iou_thresholds": list(thresholds),
            "confidence_threshold": confidence_threshold,
            "match_iou_threshold": match_iou_threshold,
            "ap_interpolation": "101-point COCO-style precision envelope",
            "aggregation": "whole dataset, matched per image then accumulated",
        },
        "class_names": names,
        "images": len(gt),
        "ground_truth_objects": total_gt,
        "ground_truth_per_class": {
            name: gt_counts[index] for index, name in enumerate(names)
        },
        "predictions_total": sum(len(items) for items in pred),
        "predictions_at_confidence": sum(
            1
            for items in pred
            for item in items
            if item["confidence"] >= confidence_threshold
        ),
        "mAP@0.5": float(np.mean(map50_values)) if map50_values else None,
        "mAP@0.5:0.95": float(np.mean(map_values)) if map_values else None,
        "ap_by_iou": {
            f"{threshold:.2f}": value
            for threshold, value in ap_by_threshold.items()
        },
        "per_class": per_class,
        "micro": {
            "true_positives": micro_tp,
            "false_positives": micro_fp,
            "false_negatives": micro_fn,
            "precision": micro_tp / (micro_tp + micro_fp)
            if micro_tp + micro_fp
            else None,
            "recall": micro_tp / (micro_tp + micro_fn)
            if micro_tp + micro_fn
            else None,
        },
        "defined_classes": defined,
        "undefined_classes": [name for name in names if name not in defined],
        "class_confusion": _confusion(gt, pred, names, match_iou_threshold),
    }
