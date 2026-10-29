"""Evaluate a DETR checkpoint with real detection metrics.

This replaces loss-only reporting with COCO-style mAP, per-class AP, micro
precision/recall, false-positive/false-negative counts, confidence
distributions, and visual error analysis. Aggregate metrics are computed over
the full dataset, never averaged per batch.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data import DatasetError, DETRData
from model import DETR
from utils.boxes import box_cxcywh_to_xyxy, stacker
from utils.checkpoint import DEFAULT_CHECKPOINT_PATH, CheckpointError
from utils.detection_metrics import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_detection,
    match_image_detections,
)
from utils.device import resolve_device
from utils.reports import (
    environment_info,
    markdown_table,
    utc_now,
    write_json,
    write_text,
)
from utils.setup import ConfigError, get_classes

VISUAL_COLORS = {
    "ground_truth": (40, 200, 80),
    "true_positive": (45, 130, 255),
    "false_positive": (230, 60, 60),
    "false_negative": (255, 165, 0),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(dataset: DETRData) -> str:
    """Hash paths, sizes, and raw labels so reports identify exact data."""
    digest = hashlib.sha256()
    for image_path, label_path, _, _ in dataset.samples:
        digest.update(image_path.name.encode("utf-8"))
        digest.update(str(image_path.stat().st_size).encode("ascii"))
        digest.update(label_path.read_bytes())
    return digest.hexdigest()


def raw_ground_truth(dataset: DETRData) -> list[list[dict]]:
    """Ground truth from raw annotations (eval resize preserves normalization)."""
    images: list[list[dict]] = []
    for _, _, labels, boxes in dataset.samples:
        xyxy = box_cxcywh_to_xyxy(boxes)
        images.append(
            [
                {
                    "class_id": int(label),
                    "bbox": [float(value) for value in box],
                }
                for label, box in zip(labels.tolist(), xyxy.tolist())
            ]
        )
    return images


def predictions_from_output(
    output: dict, score_threshold: float
) -> list[list[dict]]:
    """Convert raw model output to normalized xyxy detections per image."""
    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    probabilities = logits.softmax(-1)[..., :-1]
    scores, classes = probabilities.max(-1)
    xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
    results: list[list[dict]] = []
    for image_index in range(scores.shape[0]):
        detections = []
        for query_index in range(scores.shape[1]):
            score = float(scores[image_index, query_index].item())
            if score < score_threshold:
                continue
            detections.append(
                {
                    "class_id": int(classes[image_index, query_index].item()),
                    "confidence": score,
                    "bbox": [
                        float(value)
                        for value in xyxy[image_index, query_index].tolist()
                    ],
                    "query_index": query_index,
                }
            )
        results.append(detections)
    return results


def _draw_box(
    draw: ImageDraw.ImageDraw, box, color, label: str, size: tuple[int, int]
) -> None:
    width, height = size
    x1, y1, x2, y2 = box
    draw.rectangle(
        [x1 * width, y1 * height, x2 * width, y2 * height],
        outline=color,
        width=3,
    )
    text_position = (max(0, x1 * width), max(0, y1 * height - 14))
    draw.text(text_position, label, fill=color)


def render_visualization(
    image_path: Path,
    destination: Path,
    gt_items: list[dict],
    pred_items: list[dict],
    matching: dict,
    class_names: list[str],
) -> dict:
    """Draw GT, TP, FP, and FN boxes for one image."""
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for gt_index, item in enumerate(gt_items):
        status = (
            "FN"
            if gt_index in matching["unmatched_ground_truth"]
            else "GT/TP"
        )
        color = (
            VISUAL_COLORS["false_negative"]
            if status == "FN"
            else VISUAL_COLORS["ground_truth"]
        )
        _draw_box(
            draw,
            item["bbox"],
            color,
            f"{status} {class_names[item['class_id']]}",
            canvas.size,
        )
    false_positive = set(matching["false_positive_indices"])
    for pred_index, item in enumerate(pred_items):
        status = "FP" if pred_index in false_positive else "TP"
        color = (
            VISUAL_COLORS["false_positive"]
            if status == "FP"
            else VISUAL_COLORS["true_positive"]
        )
        _draw_box(
            draw,
            item["bbox"],
            color,
            f"{status} {class_names[item['class_id']]} {item['confidence']:.2f}",
            canvas.size,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=85)
    return {"path": str(destination), "width": canvas.width, "height": canvas.height}


def confidence_histogram(predictions: list[list[dict]]) -> dict:
    bins = [index / 10 for index in range(11)]
    counts = [0] * 10
    for image_pred in predictions:
        for item in image_pred:
            index = min(9, int(item["confidence"] * 10))
            counts[index] += 1
    return {"bin_edges": bins, "counts": counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.01,
        help="Predictions below this score are excluded from AP calculation",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for reported precision/recall and errors",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for reported precision/recall and errors",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="Experiment name used for output filenames",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports")
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="Maximum number of annotated error images to write",
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Trusted legacy checkpoints only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if not 0.0 <= args.score_threshold <= 1.0:
            raise ValueError("--score-threshold must be in [0, 1]")
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError("--confidence must be in [0, 1]")
        if not 0.0 < args.iou <= 1.0:
            raise ValueError("--iou must be in (0, 1]")
        if args.max_visualizations < 0:
            raise ValueError("--max-visualizations must be non-negative")
        if not args.name.strip() or "/" in args.name:
            raise ValueError("--name must be a non-empty filename component")

        classes = get_classes()
        device = resolve_device(args.device)
        dataset = DETRData(
            args.test_dir, train=False, classes=classes, verbose=False
        )
        if len(dataset) == 0:
            raise DatasetError(f"Test dataset is empty: {args.test_dir}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=stacker,
            drop_last=False,
            num_workers=args.num_workers,
        )

        model = DETR(num_classes=len(classes), backbone_weights=None, verbose=False)
        model = model.to(device)
        model.eval()
        metadata = model.load_pretrained(
            str(args.checkpoint),
            classes=classes,
            map_location=device,
            allow_unsafe_load=args.allow_unsafe_load,
        )

        ground_truth = raw_ground_truth(dataset)
        predictions: list[list[dict]] = []
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device)
                output = model(images)
                batch_predictions = predictions_from_output(
                    output, args.score_threshold
                )
                predictions.extend(batch_predictions)

        if len(predictions) != len(ground_truth):
            raise RuntimeError(
                f"Collected {len(predictions)} prediction sets for "
                f"{len(ground_truth)} images"
            )

        metrics = evaluate_detection(
            ground_truth,
            predictions,
            classes,
            iou_thresholds=DEFAULT_IOU_THRESHOLDS,
            confidence_threshold=args.confidence,
            match_iou_threshold=args.iou,
        )

        # Error analysis at the reported confidence/IoU thresholds.
        image_records = []
        visual_paths = []
        for image_index, (gt_items, pred_items) in enumerate(
            zip(ground_truth, predictions)
        ):
            selected = [
                item for item in pred_items if item["confidence"] >= args.confidence
            ]
            matching = match_image_detections(
                gt_items,
                pred_items,
                classes,
                iou_threshold=args.iou,
                confidence_threshold=args.confidence,
            )
            true_positive = len(matching["matches"])
            false_positive = len(matching["false_positive_indices"])
            false_negative = len(matching["unmatched_ground_truth"])
            image_path = dataset.samples[image_index][0]
            record = {
                "image": image_path.name,
                "path": str(image_path),
                "ground_truth_objects": len(gt_items),
                "predictions": len(selected),
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "matches": matching["matches"],
                "false_positive_indices": matching["false_positive_indices"],
                "unmatched_ground_truth": matching["unmatched_ground_truth"],
                "ground_truth_boxes": gt_items,
                "predicted_boxes": selected,
            }
            if len(visual_paths) < args.max_visualizations:
                destination = (
                    args.reports_dir
                    / "error_analysis"
                    / f"{args.name}_{image_path.stem}.jpg"
                )
                visual = render_visualization(
                    image_path,
                    destination,
                    gt_items,
                    selected,
                    matching,
                    classes,
                )
                record["visualization"] = visual["path"]
                visual_paths.append(visual["path"])
            image_records.append(record)

        checkpoint_hash = sha256_file(args.checkpoint)
        split_integrity = None
        integrity_path = args.reports_dir / "split_integrity.json"
        if integrity_path.is_file():
            import json

            split_integrity = json.loads(integrity_path.read_text(encoding="utf-8"))

        warnings = []
        if split_integrity and split_integrity.get("leakage", {}).get("status") in {
            "confirmed",
            "potential",
        }:
            warnings.append(
                "Split integrity report flags leakage between train and test; "
                "these metrics are not person-independent generalization."
            )
        if not args.test_dir.resolve() == Path("data/test").resolve():
            warnings.append(
                f"Evaluation used a non-default test directory: {args.test_dir}"
            )

        summary = dataset.summary()
        report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "experiment": args.name,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": checkpoint_hash,
                "epoch": metadata.get("epoch"),
                "best_val_loss": metadata.get("best_val_loss"),
                "classes": metadata.get("classes", classes),
                "format_version": metadata.get("format_version"),
            },
            "dataset": {
                "path": str(args.test_dir),
                "identity_sha256": dataset_identity(dataset),
                "samples": summary["samples"],
                "boxes": summary["boxes"],
                "class_counts": summary["class_counts"],
            },
            "settings": {
                "device": str(device),
                "batch_size": args.batch_size,
                "score_threshold": args.score_threshold,
                "confidence_threshold": args.confidence,
                "match_iou_threshold": args.iou,
                "iou_thresholds": list(DEFAULT_IOU_THRESHOLDS),
            },
            "metrics": metrics,
            "environment": environment_info(device),
            "warnings": warnings,
        }
        metrics_path = args.reports_dir / f"{args.name}_metrics.json"
        write_json(metrics_path, report)

        error_report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "experiment": args.name,
            "checkpoint_sha256": checkpoint_hash,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "summary": {
                "images": len(image_records),
                "ground_truth_objects": sum(
                    record["ground_truth_objects"] for record in image_records
                ),
                "predictions": sum(record["predictions"] for record in image_records),
                "true_positives": sum(
                    record["true_positives"] for record in image_records
                ),
                "false_positives": sum(
                    record["false_positives"] for record in image_records
                ),
                "false_negatives": sum(
                    record["false_negatives"] for record in image_records
                ),
                "images_with_errors": sum(
                    1
                    for record in image_records
                    if record["false_positives"] or record["false_negatives"]
                ),
            },
            "confidence_distribution": confidence_histogram(predictions),
            "images": image_records,
            "visualizations": visual_paths,
            "notes": [
                "Box overlays: green = ground truth, blue = true positive, "
                "red = false positive, orange = false negative.",
                "Visualizations are limited to avoid large generated artifacts.",
            ],
        }
        error_path = args.reports_dir / "error_analysis.json"
        write_json(error_path, error_report)

        md_rows = [
            ("Experiment", args.name),
            ("Checkpoint", f"`{args.checkpoint}`"),
            ("Checkpoint SHA-256", f"`{checkpoint_hash}`"),
            ("Test directory", f"`{args.test_dir}`"),
            ("Images", str(summary["samples"])),
            ("Ground-truth objects", str(metrics["ground_truth_objects"])),
            ("mAP@0.5", _fmt(metrics["mAP@0.5"])),
            ("mAP@0.5:0.95", _fmt(metrics["mAP@0.5:0.95"])),
            ("Micro precision", _fmt(metrics["micro"]["precision"])),
            ("Micro recall", _fmt(metrics["micro"]["recall"])),
            ("False positives", str(metrics["micro"]["false_positives"])),
            ("False negatives", str(metrics["micro"]["false_negatives"])),
        ]
        per_class_lines = [
            "| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | GT | TP | FP | FN |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, values in metrics["per_class"].items():
            per_class_lines.append(
                f"| {name} | {_fmt(values['ap50'])} | {_fmt(values['ap'])} | "
                f"{_fmt(values['precision'])} | {_fmt(values['recall'])} | "
                f"{values['ground_truth']} | {values['true_positives']} | "
                f"{values['false_positives']} | {values['false_negatives']} |"
            )
        markdown = "\n".join(
            [
                f"# Detection evaluation — {args.name}",
                "",
                markdown_table(md_rows),
                "",
                f"Thresholds: confidence {args.confidence}, IoU {args.iou}; "
                f"AP input score threshold {args.score_threshold}.",
                "",
                "## Per-class metrics",
                "",
                *per_class_lines,
                "",
                "## Files",
                "",
                f"- Metrics: `{metrics_path}`",
                f"- Error analysis: `{error_path}`",
                "",
                *(["## Warnings", ""] + [f"- {item}" for item in warnings] if warnings else []),
                "",
            ]
        )
        write_text(args.reports_dir / f"{args.name}_summary.md", markdown)

        print(f"mAP@0.5: {_fmt(metrics['mAP@0.5'])}")
        print(f"mAP@0.5:0.95: {_fmt(metrics['mAP@0.5:0.95'])}")
        print(f"Wrote {metrics_path}")
        print(f"Wrote {error_path}")
        return 0
    except (CheckpointError, DatasetError, ConfigError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2


def _fmt(value) -> str:
    return "undefined" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
