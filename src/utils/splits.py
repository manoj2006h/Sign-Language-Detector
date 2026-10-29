"""Dataset split integrity: grouping, duplicates, and leakage detection.

The project dataset does not ship signer metadata. Filenames do carry a UUID
whose timestamp/clock/node fields are shared by frames captured in the same
recording batch. That is a heuristic grouping key, not a verified signer id.
It is strong enough to detect neighbouring frames split across train/test.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

UUID_RE = re.compile(
    r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})",
    re.IGNORECASE,
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class SplitIntegrityError(ValueError):
    """Raised when split integrity inputs are malformed."""


def recording_group(stem: str) -> str:
    """Infer a recording-batch key from a filename stem.

    UUID time-low is per sample; the remaining UUID fields encode capture time
    and node. Files without a UUID fall back to the whole stem so they never
    share a group accidentally.
    """
    match = UUID_RE.search(stem)
    if not match:
        return stem
    return "-".join(match.groups()[1:])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: str | Path, hash_size: int = 8) -> int:
    """Return a 64-bit difference hash for near-duplicate detection."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _class_counts(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    if not labels_dir.is_dir():
        return counts
    for path in sorted(labels_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields:
                try:
                    counts[int(fields[0])] += 1
                except ValueError:
                    counts["invalid"] += 1
    return counts


def _index_split(split_dir: Path, split: str, class_names: Sequence[str]) -> list[dict]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        raise SplitIntegrityError(f"Images directory not found: {images_dir}")
    records: list[dict] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        class_ids: list[int] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields:
                    class_ids.append(int(fields[0]))
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            {
                "split": split,
                "stem": image_path.stem,
                "path": str(image_path),
                "group": recording_group(image_path.stem),
                "sha256": sha256_file(image_path),
                "dhash": dhash(image_path),
                "width": width,
                "height": height,
                "class_ids": class_ids,
                "class_names": [
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else f"unknown:{class_id}"
                    for class_id in class_ids
                ],
            }
        )
    return records


def grouped_split_indices(
    group_keys: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split indices so every recording group lands wholly on one side."""
    if not 0.0 < val_fraction < 1.0:
        raise SplitIntegrityError(
            f"val_fraction must be in (0, 1), got {val_fraction}"
        )
    if not group_keys:
        raise SplitIntegrityError("No samples supplied")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        by_group[str(key)].append(index)
    groups = sorted(by_group)
    if len(groups) < 2:
        raise SplitIntegrityError(
            "At least two recording groups are required for a grouped split"
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    target = max(1, int(round(len(group_keys) * val_fraction)))
    val_indices: list[int] = []
    for group in groups:
        if len(val_indices) >= target:
            break
        val_indices.extend(by_group[group])
    if not val_indices:
        val_indices = list(by_group[groups[0]])
    val_set = set(val_indices)
    train_indices = [index for index in range(len(group_keys)) if index not in val_set]
    return sorted(train_indices), sorted(val_indices)


def audit_split_integrity(
    data_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    class_names: Sequence[str] | None = None,
    near_duplicate_distance: int = 6,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Audit exact/near duplicates and recording-group overlap across splits."""
    root = Path(data_root)
    names = list(class_names) if class_names else []
    if near_duplicate_distance < 0 or near_duplicate_distance > 64:
        raise SplitIntegrityError(
            "near_duplicate_distance must be between 0 and 64"
        )

    records: list[dict] = []
    per_split: dict[str, dict] = {}
    for split in splits:
        split_dir = root / split
        split_records = _index_split(split_dir, split, names)
        records.extend(split_records)
        per_split[split] = {
            "samples": len(split_records),
            "class_counts": dict(
                sorted(_class_counts(split_dir / "labels").items())
            ),
            "objects": sum(len(record["class_ids"]) for record in split_records),
        }

    # Exact duplicates anywhere.
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    exact_pairs = [
        {
            "sha256": digest,
            "splits": sorted({item["split"] for item in items}),
            "paths": [item["path"] for item in items],
        }
        for digest, items in by_hash.items()
        if len(items) > 1
    ]
    exact_cross_split = [pair for pair in exact_pairs if len(pair["splits"]) > 1]

    # Near duplicates (perceptual dHash) across different files.
    near_pairs: list[dict] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left, right = records[left_index], records[right_index]
            if left["sha256"] == right["sha256"]:
                continue
            distance = hamming_distance(left["dhash"], right["dhash"])
            if distance <= near_duplicate_distance:
                near_pairs.append(
                    {
                        "distance": distance,
                        "splits": sorted({left["split"], right["split"]}),
                        "paths": [left["path"], right["path"]],
                    }
                )
    near_cross_split = [pair for pair in near_pairs if len(pair["splits"]) > 1]

    # Recording-group overlap.
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["group"]].add(record["split"])
    shared_groups = {
        group: sorted(values)
        for group, values in group_splits.items()
        if len(values) > 1
    }

    reasons: list[str] = []
    if exact_cross_split:
        reasons.append(
            f"{len(exact_cross_split)} byte-identical image(s) appear in more "
            "than one split."
        )
    if near_cross_split:
        reasons.append(
            f"{len(near_cross_split)} near-duplicate pair(s) cross splits "
            f"(dHash distance <= {near_duplicate_distance})."
        )
    if shared_groups:
        reasons.append(
            f"{len(shared_groups)} recording-group key(s) occur in more than "
            "one split; neighbouring frames are not isolated."
        )
    status = "confirmed" if reasons else "none"
    if not reasons and near_pairs:
        status = "potential"

    # Proposed leakage-resistant validation split over the training data only.
    train_records = [record for record in records if record["split"] == splits[0]]
    proposed = None
    if len(train_records) >= 2 and len({r["group"] for r in train_records}) >= 2:
        train_indices, val_indices = grouped_split_indices(
            [record["group"] for record in train_records],
            val_fraction=val_fraction,
            seed=seed,
        )
        proposed = {
            "description": (
                "Grouped validation split over the training split only. The "
                "test split is not touched. Not applied automatically."
            ),
            "val_fraction": val_fraction,
            "seed": seed,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "val_groups": sorted(
                {train_records[index]["group"] for index in val_indices}
            ),
        }

    return {
        "schema_version": 1,
        "data_root": str(root),
        "splits": list(splits),
        "per_split": per_split,
        "total_samples": len(records),
        "grouping": {
            "method": (
                "Unaltered UUID fields are used as an inferred recording-batch "
                "key. This is not a verified signer identity."
            ),
            "group_count": len(group_splits),
            "shared_groups": shared_groups,
        },
        "exact_duplicates": exact_pairs,
        "exact_cross_split": exact_cross_split,
        "near_duplicates": near_pairs,
        "near_duplicates_cross_split": near_cross_split,
        "near_duplicate_threshold": near_duplicate_distance,
        "leakage": {
            "status": status,
            "reasons": reasons,
        },
        "signer_metadata_available": False,
        "person_independent_generalization": "not established",
        "limitations": [
            "No signer or session metadata is shipped with data/train or "
            "data/test; group keys are inferred from filename UUIDs.",
            "The existing data/test split is not signer-independent and must "
            "not be described as external generalization.",
            "Exact duplicates cannot be detected after lossy re-encoding; use "
            "the near-duplicate report as a lower bound.",
        ],
        "proposed_grouped_validation_split": proposed,
        "samples": [
            {key: record[key] for key in (
                "split", "stem", "group", "sha256", "dhash", "class_ids",
                "class_names", "width", "height",
            )}
            for record in records
        ],
    }
