"""Audit the internal dataset, the Amrita archive, and split integrity.

Writes ``reports/dataset_audit.json``, ``reports/dataset_compatibility.json``,
``reports/split_integrity.json``, and a human-readable Markdown summary. The
archive is inspected, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data import DatasetError, DETRData
from utils.archive import ArchiveError, inspect_archive
from utils.compat import (
    CompatibilityError,
    compare_vocabularies,
    discover_external_classes,
)
from utils.reports import environment_info, markdown_table, utc_now, write_json, write_text
from utils.setup import ConfigError, get_classes
from utils.splits import SplitIntegrityError, audit_split_integrity

try:
    from prepare_amrita import DEFAULT_OUTPUT, MANIFEST_NAME, find_default_archive
except ImportError:  # pragma: no cover - module path fallback
    DEFAULT_OUTPUT = Path("dataset/external/amrita_slr")
    MANIFEST_NAME = "amrita_manifest.json"

    def find_default_archive() -> Path:
        raise ArchiveError("Amrita archive helper is unavailable")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def dataset_identity(dataset: DETRData) -> str:
    digest = hashlib.sha256()
    for image_path, label_path, _, _ in dataset.samples:
        digest.update(image_path.name.encode("utf-8"))
        digest.update(str(image_path.stat().st_size).encode("ascii"))
        digest.update(label_path.read_bytes())
    return digest.hexdigest()


def split_summary(dataset: DETRData) -> dict:
    dims: Counter = Counter()
    for image_path, _, _, _ in dataset.samples:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                dims[f"{image.size[0]}x{image.size[1]}"] += 1
        except OSError:
            dims["unreadable"] += 1
    errors = dataset.validate_images()
    return {
        **dataset.summary(),
        "identity_sha256": dataset_identity(dataset),
        "image_dimensions": dict(dims.most_common()),
        "invalid_images": errors,
    }


def load_evidence(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence file must contain a JSON object: {path}")
    return payload


def audit(
    *,
    data_root: Path,
    splits: tuple[str, ...],
    archive: Path | None,
    external_root: Path,
    reports_dir: Path,
    evidence: dict,
    class_names: list[str],
) -> dict:
    internal: dict[str, dict] = {}
    for split in splits:
        split_dir = data_root / split
        dataset = DETRData(split_dir, train=(split == "train"), classes=class_names, verbose=False)
        internal[split] = split_summary(dataset)

    integrity = audit_split_integrity(
        data_root, splits=splits, class_names=class_names
    )

    archive_info = None
    archive_error = None
    try:
        resolved_archive = archive or find_default_archive()
        archive_info = inspect_archive(resolved_archive, check_crc=True)
        archive_info["path"] = str(resolved_archive)
    except (ArchiveError, OSError) as exc:
        archive_error = str(exc)

    manifest = None
    manifest_path = external_root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    external_summary = {
        "archive": archive_info,
        "archive_error": archive_error,
        "extracted_root": str(external_root),
        "extracted": external_root.is_dir(),
        "manifest_path": str(manifest_path) if manifest else None,
    }
    if manifest:
        external_summary.update(
            {
                "annotation_status": manifest.get("annotation_status"),
                "bounding_boxes_available": manifest.get("bounding_boxes_available"),
                "temporal_annotations_available": manifest.get(
                    "temporal_annotations_available"
                ),
                "totals": manifest.get("totals"),
                "classes": manifest.get("classes"),
                "signer_metadata": manifest.get("signer_metadata"),
                "license": manifest.get("license"),
                "documentation_files": manifest.get("documentation_files"),
                "duplicates": manifest.get("duplicates"),
                "detr_training_ready": manifest.get("detr_training_ready"),
                "blockers": manifest.get("blockers"),
            }
        )

    # Compatibility: prefer the extracted manifest, else archive folder names.
    external_classes = []
    if manifest:
        external_classes = discover_external_classes(
            [item["folder"] for item in manifest.get("classes", [])],
            {item["folder"]: item.get("videos", 0) for item in manifest.get("classes", [])},
        )
    elif external_root.is_dir():
        folders = sorted(
            path.name
            for path in external_root.rglob("*")
            if path.is_dir()
            and path.parent.name == "sign project"
        )
        external_classes = discover_external_classes(folders)
    compatibility = compare_vocabularies(
        class_names, external_classes, evidence=evidence
    )

    dataset_audit = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "internal": {
            "data_root": str(data_root),
            "splits": internal,
            "exact_duplicate_groups": len(integrity["exact_duplicates"]),
            "near_duplicate_pairs": len(integrity["near_duplicates"]),
        },
        "external": external_summary,
        "environment": environment_info(),
        "limitations": [
            "Internal test labels are object detection boxes for three classes; "
            "the dataset is small.",
            "Amrita provides no bounding boxes, so it cannot be scored with "
            "detection mAP.",
        ],
    }

    compatibility_report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "project_classes": class_names,
        "external_dataset": "Amrita SLR - Real Time ISL Dataset",
        "external_annotation_level": (
            manifest.get("annotation_status") if manifest else "unknown"
        ),
        "evidence_supplied": bool(evidence),
        "compatibility": compatibility.to_dict(),
        "can_convert_to_detr": False,
        "conversion_blockers": [
            "No bounding-box annotations exist for any Amrita video.",
            "No temporal segmentation exists to isolate a sign window.",
            "No documentation verifies that external hello/thankyou definitions "
            "match the project's signs.",
            "No license/citation file is present in the archive.",
        ],
        "recommended_use": (
            "Video-level qualitative inspection or a separate video/landmark "
            "classification experiment after obtaining authorization and "
            "verified annotations. Not DETR training data in Phase 2."
        ),
    }

    write_json(reports_dir / "dataset_audit.json", dataset_audit)
    write_json(reports_dir / "dataset_compatibility.json", compatibility_report)
    write_json(reports_dir / "split_integrity.json", integrity)

    md = _markdown(
        internal=internal,
        integrity=integrity,
        external=external_summary,
        compatibility=compatibility.to_dict(),
        archive_error=archive_error,
    )
    write_text(reports_dir / "PHASE2_DATASET_AUDIT.md", md)
    return {
        "dataset_audit": dataset_audit,
        "compatibility": compatibility_report,
        "integrity": integrity,
    }


def _markdown(
    *,
    internal: dict,
    integrity: dict,
    external: dict,
    compatibility: dict,
    archive_error: str | None,
) -> str:
    lines = [
        "# Phase 2 dataset audit",
        "",
        "Generated by `src/audit_datasets.py`. All values are measured from the",
        "local files; nothing is inferred beyond the stated heuristics.",
        "",
        "## Internal dataset",
        "",
        "| Split | Images | Objects | Classes | Invalid images |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split, summary in internal.items():
        lines.append(
            f"| {split} | {summary['samples']} | {summary['boxes']} | "
            f"{summary['class_counts']} | {len(summary['invalid_images'])} |"
        )
    lines.extend(
        [
            "",
            "## Split integrity",
            "",
            f"- Leakage status: **{integrity['leakage']['status']}**",
            f"- Inferred recording groups: {integrity['grouping']['group_count']}",
            f"- Shared groups across splits: {len(integrity['grouping']['shared_groups'])}",
            f"- Exact duplicate groups: {len(integrity['exact_duplicates'])}",
            f"- Near-duplicate pairs: {len(integrity['near_duplicates'])}",
            "- Signer metadata available: "
            f"{integrity['signer_metadata_available']}",
            "- Person-independent generalization: "
            f"{integrity['person_independent_generalization']}",
            "",
        ]
    )
    for reason in integrity["leakage"]["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## Amrita SLR external dataset", ""])
    if archive_error:
        lines.append(f"Archive error: `{archive_error}`")
    else:
        archive = external.get("archive") or {}
        lines.extend(
            [
                markdown_table(
                    [
                        ("Archive", f"`{archive.get('path')}`"),
                        ("ZIP SHA-256", f"`{archive.get('sha256')}`"),
                        ("ZIP size", f"{archive.get('size_bytes')} bytes"),
                        ("Entries", str(archive.get("entry_count"))),
                        ("Videos", str((external.get("totals") or {}).get("videos"))),
                        ("Classes", str((external.get("totals") or {}).get("classes"))),
                        ("Annotation status", str(external.get("annotation_status"))),
                        ("Bounding boxes", str(external.get("bounding_boxes_available"))),
                        ("License", str(external.get("license"))),
                        ("DETR training ready", str(external.get("detr_training_ready"))),
                    ]
                ),
                "",
            ]
        )
    lines.extend(["## Vocabulary compatibility", ""])
    lines.append("| External label | Project class | Status | Reason |")
    lines.append("| --- | --- | --- | --- |")
    for match in compatibility["matches"]:
        lines.append(
            f"| {match['external_label']} | {match['project_class']} | "
            f"{match['status']} | {match['reason']} |"
        )
    lines.extend(
        [
            "",
            f"- Unmatched external classes: {len(compatibility['unmatched_external'])}",
            f"- Project classes without verified mapping: "
            f"{compatibility['missing_project_classes']}",
            f"- Training mapping: {compatibility['training_mapping'] or 'none'}",
            "",
            "No class is mapped from English gloss alone. Without verified sign",
            "definitions and bounding boxes, Amrita is excluded from DETR",
            "training and detection evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="JSON evidence for verified sign-definition mappings",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        classes = get_classes()
        evidence = load_evidence(args.evidence)
        audit(
            data_root=args.data_root,
            splits=tuple(args.splits),
            archive=args.archive,
            external_root=args.external_root,
            reports_dir=args.reports_dir,
            evidence=evidence,
            class_names=classes,
        )
        print(f"Wrote {args.reports_dir / 'dataset_audit.json'}")
        print(f"Wrote {args.reports_dir / 'dataset_compatibility.json'}")
        print(f"Wrote {args.reports_dir / 'split_integrity.json'}")
        return 0
    except (
        ArchiveError,
        CompatibilityError,
        ConfigError,
        DatasetError,
        SplitIntegrityError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
