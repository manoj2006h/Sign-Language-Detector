"""Train the DETR sign detector with a leakage-free validation split."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import DatasetError, DETRData
from loss import DETRLoss, HungarianMatcher
from model import DETR
from utils.boxes import stacker
from utils.checkpoint import (
    CheckpointError,
    load_model_checkpoint,
    save_training_checkpoint,
)
from utils.device import move_targets_to_device, resolve_device
from utils.inference import average_or_none, compute_loss_total
from utils.logger import get_logger
from utils.setup import get_classes

WEIGHTS = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}


def split_train_validation_indices(
    num_samples: int, val_fraction: float = 0.1, *, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Deterministically split training samples without touching the test set."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    indices = list(range(num_samples))
    if num_samples == 0:
        return [], []
    if val_fraction == 0.0:
        return indices, []
    if num_samples < 2:
        raise ValueError(
            "At least 2 training samples are required for a validation split; "
            "pass --val-fraction 0 for a single-sample smoke run."
        )

    num_val = max(1, int(round(num_samples * val_fraction)))
    num_val = min(num_val, num_samples - 1)
    rng = random.Random(seed)
    val = sorted(rng.sample(indices, num_val))
    val_set = set(val)
    train = [index for index in indices if index not in val_set]
    return train, val


def check_resume_split(
    metadata: dict, train_indices: list[int], val_indices: list[int]
) -> bool:
    """Reject a resume whose validation split no longer matches the checkpoint.

    Returns ``False`` when the checkpoint predates split metadata, so old
    checkpoints can still be resumed with a warning.
    """
    saved_train = metadata.get("train_indices")
    saved_val = metadata.get("val_indices")
    if saved_train is None and saved_val is None:
        return False
    if list(saved_train or []) != list(train_indices) or list(saved_val or []) != list(
        val_indices
    ):
        raise CheckpointError(
            "Checkpoint was created with a different train/validation split. "
            "The data/train files changed since this checkpoint was saved. "
            "Restore the original dataset or start a new training run."
        )
    return True


def _build_loaders(args, classes: list[str], device: torch.device):
    full_train = DETRData(
        args.train_dir, train=True, classes=classes, verbose=True
    )
    if len(full_train) == 0:
        raise DatasetError(f"No training samples found in {args.train_dir}")
    train_indices, val_indices = split_train_validation_indices(
        len(full_train), args.val_fraction, seed=args.seed
    )
    if not train_indices:
        raise DatasetError("The training split is empty")

    train_dataset = DETRData(
        args.train_dir,
        train=True,
        classes=classes,
        indices=train_indices,
        verbose=False,
    )
    val_dataset = (
        DETRData(
            args.train_dir,
            train=False,
            classes=classes,
            indices=val_indices,
            verbose=False,
        )
        if val_indices
        else None
    )
    if val_dataset is not None and full_train.logger is not None:
        missing_classes = [
            name
            for name, count in val_dataset.summary()["class_counts"].items()
            if count == 0
        ]
        if missing_classes:
            full_train.logger.warning(
                "Validation split contains no boxes for class(es): "
                + ", ".join(missing_classes)
                + ". Validation loss will not reflect them."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=stacker,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=stacker,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader, train_indices, val_indices


def _run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        losses = criterion(output, targets)
        loss = compute_loss_total(losses, criterion.weight_dict)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        batches += 1
    average = average_or_none(total, batches)
    if average is None:
        raise RuntimeError("Training loader produced no batches")
    return average


def _run_validation(model, loader, criterion, device):
    model.eval()
    total = 0.0
    batches = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            output = model(images)
            losses = criterion(output, targets)
            total += float(compute_loss_total(losses, criterion.weight_dict).item())
            batches += 1
    average = average_or_none(total, batches)
    if average is None:
        raise RuntimeError("Validation loader produced no batches")
    return average


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional task checkpoint for initialization (not used with --resume)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume optimizer/scheduler/epoch from a checkpoint",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Do not download/load ImageNet ResNet-50 initialization",
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow loading a trusted legacy checkpoint (unsafe for untrusted files)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = get_logger("training")
    logger.print_banner()
    classes = get_classes()
    num_classes = len(classes)
    device = resolve_device(args.device)
    logger.training(f"Using device: {device}")

    train_loader, val_loader, train_indices, val_indices = _build_loaders(
        args, classes, device
    )

    # A full task checkpoint (resume or init) supplies the backbone, so avoid
    # an unnecessary ImageNet download in those paths.
    needs_imagenet = not args.no_pretrained_backbone and not (
        args.resume is not None or args.init_checkpoint is not None
    )
    model = DETR(
        num_classes=num_classes,
        backbone_weights="imagenet" if needs_imagenet else "none",
        verbose=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, max(1, len(train_loader) * 30), T_mult=2
    )
    criterion = DETRLoss(
        num_classes=num_classes,
        matcher=HungarianMatcher(WEIGHTS),
        weight_dict=WEIGHTS,
        eos_coef=0.1,
    ).to(device)

    start_epoch = 0
    best_val_loss: float | None = None
    history: list[dict] = []
    if args.resume is not None:
        metadata = load_model_checkpoint(
            model,
            args.resume,
            num_classes=num_classes,
            classes=classes,
            expected_model_config=model.architecture_config(),
            map_location=device,
            restore_training=True,
            optimizer=optimizer,
            scheduler=scheduler,
            allow_unsafe_load=args.allow_unsafe_load,
        )
        start_epoch = int(metadata.get("epoch", 0)) + 1
        best_val_loss = metadata.get("best_val_loss")
        history = list(metadata.get("history") or [])
        if not check_resume_split(metadata, train_indices, val_indices):
            logger.warning(
                "Checkpoint has no train/validation split metadata; "
                "cannot verify that the split is unchanged."
            )
        logger.training(
            f"Resumed from {args.resume} at epoch {start_epoch} "
            f"(best validation loss: {best_val_loss})"
        )
    elif args.init_checkpoint is not None:
        model.load_pretrained(
            str(args.init_checkpoint),
            classes=classes,
            map_location=device,
            allow_unsafe_load=args.allow_unsafe_load,
        )
        logger.training(f"Initialized model from {args.init_checkpoint}")

    model.log_model_info()
    logger.print_table(
        "Training Configuration",
        ["Parameter", "Value"],
        [
            ["Epochs", str(args.epochs)],
            ["Batch Size", str(args.batch_size)],
            ["Train Batches", str(len(train_loader))],
            ["Validation Batches", str(len(val_loader) if val_loader else 0)],
            ["Learning Rate", str(args.lr)],
            ["Device", str(device)],
            ["Validation Fraction", str(args.val_fraction)],
        ],
    )

    best_path = args.checkpoint_dir / "best.pt"
    last_path = args.checkpoint_dir / "last.pt"

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = (
            _run_validation(model, val_loader, criterion, device)
            if val_loader is not None
            else None
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        logger.training(
            f"Epoch {epoch + 1}/{args.epochs}: train={train_loss:.5f}"
            + (f", val={val_loss:.5f}" if val_loss is not None else "")
            + f", lr={record['lr']:.2e}, {time.time() - started:.1f}s"
        )

        improved = val_loss is not None and (
            best_val_loss is None or val_loss < best_val_loss
        )
        if improved:
            best_val_loss = float(val_loss)

        save_training_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            classes=classes,
            model_config=model.architecture_config(),
            val_loss=val_loss,
            history=history,
            train_indices=train_indices,
            val_indices=val_indices,
        )
        if improved:
            save_training_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                classes=classes,
                model_config=model.architecture_config(),
                val_loss=val_loss,
                history=history,
                train_indices=train_indices,
                val_indices=val_indices,
            )
            logger.success(f"Saved new best checkpoint to {best_path}")

    if val_loader is None:
        logger.warning(
            "Validation was disabled (--val-fraction 0); no best.pt was selected. "
            f"The latest checkpoint is {last_path}."
        )
    if start_epoch >= args.epochs:
        logger.warning("No epochs were run; the requested epoch count is already reached.")

    return {
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if val_loader is not None else None,
        "best_val_loss": best_val_loss,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except (CheckpointError, DatasetError, RuntimeError, ValueError, OSError) as exc:
        get_logger("training").error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
