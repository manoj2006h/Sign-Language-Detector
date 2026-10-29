"""Evaluate a trained checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DatasetError, DETRData
from loss import DETRLoss, HungarianMatcher
from model import DETR
from utils.boxes import stacker
from utils.checkpoint import DEFAULT_CHECKPOINT_PATH, CheckpointError
from utils.device import move_targets_to_device, resolve_device
from utils.inference import average_or_none, compute_loss_total, count_detections
from utils.logger import get_logger
from utils.rich_handlers import TestHandler
from utils.setup import get_classes

WEIGHTS = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Trained task checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Detection confidence threshold"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow loading a trusted legacy checkpoint (unsafe for untrusted files)",
    )
    return parser.parse_args(argv)


def evaluate(
    model: DETR,
    loader: DataLoader,
    criterion: DETRLoss,
    device: torch.device,
    threshold: float,
) -> dict:
    """Evaluate every batch without dropping incomplete batches."""
    model.eval()
    total_loss = 0.0
    batches = 0
    samples = 0
    detection_count = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            output = model(images)
            losses = criterion(output, targets)
            total_loss += float(
                compute_loss_total(losses, criterion.weight_dict).item()
            )
            batches += 1
            samples += images.shape[0]
            detection_count += count_detections(output, threshold)

    return {
        "loss": average_or_none(total_loss, batches),
        "batches": batches,
        "samples": samples,
        "detections": detection_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger("test")
    handler = TestHandler()
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be between 0 and 1")
        classes = get_classes()
        device = resolve_device(args.device)
        dataset = DETRData(
            args.test_dir, train=False, classes=classes, verbose=True
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

        model = DETR(
            num_classes=len(classes), backbone_weights=None, verbose=True
        ).to(device)
        model.eval()
        model.load_pretrained(
            str(args.checkpoint),
            classes=classes,
            map_location=device,
            allow_unsafe_load=args.allow_unsafe_load,
        )
        criterion = DETRLoss(
            num_classes=len(classes),
            matcher=HungarianMatcher(WEIGHTS),
            weight_dict=WEIGHTS,
            eos_coef=0.1,
        ).to(device)

        result = evaluate(model, loader, criterion, device, args.threshold)
        handler.log_test_results(
            {
                "Test loss": result["loss"],
                "Batches": result["batches"],
                "Samples": result["samples"],
                "Detections above threshold": result["detections"],
                "Threshold": args.threshold,
                "Checkpoint": str(args.checkpoint),
            }
        )
        # Loss is not an accuracy metric; no accuracy/FPS claim is made here.
        return 0
    except (CheckpointError, DatasetError, RuntimeError, ValueError, OSError) as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
