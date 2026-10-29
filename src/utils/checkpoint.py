"""Strict, explicit checkpoint loading and saving.

A task checkpoint is distinct from the ImageNet backbone initialization. Task
checkpoints produced here contain the model state plus metadata and, for
resumable training, optimizer/scheduler state. Loading is strict and never
falls back to random weights.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best.pt")
LAST_CHECKPOINT_PATH = Path("checkpoints/last.pt")
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


def _read_prefix(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_LFS_PREFIX) + 32)
    except OSError as exc:
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_load: bool = False,
) -> dict:
    """Safely load a checkpoint payload.

    ``weights_only=True`` is used by default. Legacy checkpoints can only be
    loaded with explicit ``allow_unsafe_load=True`` (documented as trusted-only).
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {resolved}. "
            "Train a model or pass an explicit --checkpoint path."
        )
    if resolved.stat().st_size == 0:
        raise CheckpointError(f"Checkpoint is empty: {resolved}")

    if _read_prefix(resolved).startswith(_LFS_PREFIX):
        raise CheckpointError(
            f"{resolved} is a Git LFS pointer, not a checkpoint. "
            "Install Git LFS and run `git lfs pull` to fetch the real weights, "
            "or pass the path to a local trained checkpoint."
        )

    try:
        payload = torch.load(
            resolved, map_location=map_location, weights_only=True
        )
    except Exception as first_error:
        if not allow_unsafe_load:
            raise CheckpointError(
                f"Could not safely load {resolved}. The file may be corrupt, or it "
                "may be a legacy checkpoint that requires the trusted-only "
                "`--allow-unsafe-load` option. Original error: {first_error}"
            ) from first_error
        try:
            payload = torch.load(
                resolved, map_location=map_location, weights_only=False
            )
        except Exception as second_error:
            raise CheckpointError(
                f"Could not load checkpoint {resolved}: {second_error}"
            ) from second_error

    if not isinstance(payload, dict):
        raise CheckpointError(
            f"Checkpoint {resolved} must contain a dictionary, got "
            f"{type(payload).__name__}"
        )
    return payload


def extract_state_dict(payload: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """Extract a model state dict from a new or legacy checkpoint payload."""
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = payload
    elif "state_dict" in payload:
        state = payload["state_dict"]
        metadata = payload
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
        metadata = {}
    else:
        raise CheckpointError(
            "Checkpoint does not contain a model state dict. Expected "
            "'model_state_dict', 'state_dict', or a raw tensor state dict."
        )

    if not isinstance(state, dict) or not state:
        raise CheckpointError("Checkpoint model state dict is empty or invalid")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise CheckpointError("Checkpoint model state dict contains non-tensor values")

    version = metadata.get("format_version")
    if version is not None:
        try:
            version_number = int(version)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                f"Checkpoint format version must be an integer, got {version!r}"
            ) from exc
        if version_number > CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"Checkpoint format version {version} is newer than supported "
                f"version {CHECKPOINT_FORMAT_VERSION}"
            )
    return state, metadata


def validate_checkpoint_compatibility(
    state: dict[str, torch.Tensor],
    metadata: dict,
    *,
    num_classes: int,
    classes: list[str] | None = None,
    expected_model_config: dict | None = None,
) -> None:
    """Check class head size, class names, and architecture metadata."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    class_weight = state.get("linear_class.weight")
    if class_weight is None:
        raise CheckpointError(
            "Checkpoint is incompatible with this DETR model: "
            "'linear_class.weight' is missing."
        )
    if class_weight.ndim != 2:
        raise CheckpointError(
            "Checkpoint is malformed: 'linear_class.weight' must have shape "
            f"(num_classes + 1, hidden_dim), got {tuple(class_weight.shape)}."
        )
    model_classes = int(class_weight.shape[0]) - 1
    if model_classes != num_classes:
        raise CheckpointError(
            f"Checkpoint class head has {model_classes} classes, but the config "
            f"defines {num_classes}. Update src/config.json or use a compatible "
            "checkpoint."
        )

    meta_classes = metadata.get("classes")
    if meta_classes is not None:
        if not isinstance(meta_classes, (list, tuple)) or not all(
            isinstance(name, str) for name in meta_classes
        ):
            raise CheckpointError(
                "Checkpoint metadata is malformed: 'classes' must be a list of "
                f"class-name strings, got {meta_classes!r}."
            )
        if classes is not None and list(meta_classes) != list(classes):
            raise CheckpointError(
                "Checkpoint class names do not match src/config.json. "
                f"Checkpoint: {list(meta_classes)}; config: {list(classes)}."
            )

    meta_num_classes = metadata.get("num_classes")
    if meta_num_classes is not None:
        if isinstance(meta_num_classes, bool):
            raise CheckpointError(
                "Checkpoint metadata is malformed: 'num_classes' must be an "
                f"integer, got {meta_num_classes!r}."
            )
        try:
            parsed_num_classes = int(meta_num_classes)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "Checkpoint metadata is malformed: 'num_classes' must be an "
                f"integer, got {meta_num_classes!r}."
            ) from exc
        if parsed_num_classes != num_classes:
            raise CheckpointError(
                f"Checkpoint metadata says num_classes={meta_num_classes}, "
                f"but the config defines {num_classes}."
            )

    if expected_model_config:
        saved_config = metadata.get("model_config")
        if saved_config:
            mismatches = {
                key: (saved_config.get(key), value)
                for key, value in expected_model_config.items()
                if key in saved_config and saved_config.get(key) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}: checkpoint={saved!r}, model={current!r}"
                    for key, (saved, current) in mismatches.items()
                )
                raise CheckpointError(
                    f"Checkpoint architecture does not match the current model ({details})"
                )


def load_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    num_classes: int,
    classes: list[str] | None = None,
    expected_model_config: dict | None = None,
    map_location: str | torch.device = "cpu",
    restore_training: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    allow_unsafe_load: bool = False,
) -> dict:
    """Load a model state dict strictly and optionally restore training state."""
    payload = load_checkpoint(
        path, map_location=map_location, allow_unsafe_load=allow_unsafe_load
    )
    state, metadata = extract_state_dict(payload)
    validate_checkpoint_compatibility(
        state,
        metadata,
        num_classes=num_classes,
        classes=classes,
        expected_model_config=expected_model_config,
    )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint {path} is incompatible with the model architecture: {exc}"
        ) from exc

    if restore_training:
        if optimizer is None:
            raise CheckpointError(
                "Cannot restore training without an optimizer instance"
            )
        optimizer_state = metadata.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict) or not optimizer_state:
            raise CheckpointError(
                f"Checkpoint {path} does not contain optimizer state; "
                "it cannot be used with --resume."
            )
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError) as exc:
            raise CheckpointError(
                f"Could not restore optimizer state from {path}: {exc}"
            ) from exc
        if scheduler is not None and metadata.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(metadata["scheduler_state_dict"])
            except (ValueError, RuntimeError) as exc:
                raise CheckpointError(
                    f"Could not restore scheduler state from {path}: {exc}"
                ) from exc

    return metadata


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    *,
    epoch: int,
    best_val_loss: float | None,
    classes: list[str],
    model_config: dict | None = None,
    val_loss: float | None = None,
    history: list[dict] | None = None,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
) -> Path:
    """Atomically save a resumable task checkpoint."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if model_config is None:
        model_config = getattr(model, "architecture_config", lambda: None)()

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "classes": list(classes),
        "num_classes": len(classes),
        "model_config": model_config,
        "history": history or [],
        "train_indices": (
            None if train_indices is None else [int(i) for i in train_indices]
        ),
        "val_indices": None if val_indices is None else [int(i) for i in val_indices],
    }

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, resolved)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return resolved
