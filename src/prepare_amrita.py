"""Safely extract and audit the Amrita SLR video corpus.

The archive is a ZIP containing a ZIP. It holds 978 videos organised as
``sign project/<order>.<label>/*.mp4|*.mov``. There are **no annotations**:
labels are video-level folder names only. This script therefore prepares a
reproducible manifest and frame samples; it never invents bounding boxes and
never marks the corpus as DETR-training-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from utils.archive import (
    ArchiveError,
    inspect_archive,
    safe_extract,
)
from utils.compat import (
    CompatibilityError,
    compare_vocabularies,
    discover_external_classes,
)
from utils.reports import environment_info, utc_now, write_json
from utils.setup import ConfigError, get_classes

DEFAULT_ARCHIVES = (
    Path("dataset/Amrita SLR- Real Time ISL Dataset.zip"),
    Path("dataset_zip/Amrita SLR- Real Time ISL Dataset.zip"),
)
DEFAULT_OUTPUT = Path("dataset/external/amrita_slr")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
SIGNER_NAMES = ("krishnaraj", "manideep", "nachiketh")
REPORT_PATH = Path("reports/amrita_preparation.json")
MANIFEST_NAME = "amrita_manifest.json"


def find_default_archive() -> Path:
    """Return the first existing default archive path."""
    for candidate in DEFAULT_ARCHIVES:
        if candidate.is_file():
            return candidate
    joined = " or ".join(str(path) for path in DEFAULT_ARCHIVES)
    raise ArchiveError(f"Amrita archive not found at {joined}")


def infer_signer(filename: str) -> tuple[str, str]:
    """Infer a signer label from the filename, clearly marked as heuristic."""
    lowered = filename.lower()
    for name in SIGNER_NAMES:
        if name in lowered:
            return name, "filename heuristic"
    if lowered.startswith("0403"):
        return "unknown_camera_batch_0403", "filename heuristic"
    return "unknown", "unavailable"


def deterministic_sample_id(relative_path: str) -> str:
    """Stable sample id derived from the archive-relative path."""
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    return f"amrita-slr-{digest[:16]}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path, *, enabled: bool = True) -> dict:
    """Return ffprobe metadata or an explicit unavailable record."""
    empty = {
        "probed": False,
        "duration_seconds": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "codec": None,
        "error": None,
    }
    executable = shutil.which("ffprobe") if enabled else None
    if not executable:
        empty["error"] = "ffprobe not available or probing disabled"
        return empty
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        empty["error"] = f"ffprobe failed: {exc}"
        return empty
    if completed.returncode != 0:
        empty["error"] = (completed.stderr or "ffprobe failed").strip()[:300]
        return empty
    try:
        payload = json.loads(completed.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        duration = stream.get("duration") or (payload.get("format") or {}).get(
            "duration"
        )
        frame_count = stream.get("nb_frames")
        return {
            "probed": True,
            "duration_seconds": float(duration) if duration is not None else None,
            "width": int(stream["width"]) if stream.get("width") else None,
            "height": int(stream["height"]) if stream.get("height") else None,
            "fps": fps,
            "frame_count": int(frame_count) if frame_count else None,
            "codec": stream.get("codec_name"),
            "error": None,
        }
    except (ValueError, TypeError, KeyError) as exc:
        empty["error"] = f"Could not parse ffprobe output: {exc}"
        return empty


def _parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def build_manifest(
    root: str | Path,
    *,
    archive_info: dict | None = None,
    classes: list[str] | None = None,
    evidence: dict | None = None,
    probe: bool = True,
    project_classes: list[str] | None = None,
) -> dict:
    """Build a manifest for an extracted Amrita tree without modifying it."""
    base = Path(root)
    if not base.is_dir():
        raise ArchiveError(f"Extracted dataset directory not found: {base}")
    video_files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        raise ArchiveError(f"No video files found under {base}")

    samples: list[dict] = []
    folder_counts: Counter = Counter()
    signer_counts: Counter = Counter()
    for path in video_files:
        relative = path.relative_to(base).as_posix()
        parts = relative.split("/")
        class_folder = parts[-2] if len(parts) >= 2 else "unknown"
        folder_counts[class_folder] += 1
        signer, signer_source = infer_signer(path.name)
        signer_counts[signer] += 1
        metadata = probe_video(path, enabled=probe)
        samples.append(
            {
                "id": deterministic_sample_id(relative),
                "relative_path": relative,
                "class_folder": class_folder,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "signer": signer,
                "signer_source": signer_source,
                "annotation": {
                    "level": "video_label_only",
                    "has_bbox": False,
                    "has_temporal_segments": False,
                },
                **metadata,
            }
        )

    by_hash: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_hash[sample["sha256"]].append(sample["id"])
    duplicates = [
        {"sha256": digest, "sample_ids": ids}
        for digest, ids in by_hash.items()
        if len(ids) > 1
    ]

    external = discover_external_classes(sorted(folder_counts), folder_counts)
    comparison = compare_vocabularies(
        project_classes if project_classes is not None else (classes or []),
        external,
        evidence=evidence,
    )

    total_bytes = sum(sample["size_bytes"] for sample in samples)
    durations = [
        sample["duration_seconds"]
        for sample in samples
        if sample.get("duration_seconds")
    ]
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "Amrita SLR - Real Time ISL Dataset",
        "extracted_root": str(base),
        "archive": archive_info,
        "annotation_status": "video_level_only",
        "bounding_boxes_available": False,
        "temporal_annotations_available": False,
        "predefined_splits": False,
        "documentation_files": [],
        "license": "not stated in archive",
        "signer_metadata": {
            "authoritative": False,
            "method": "filename heuristic (krishnaraj/manideep/nachiketh)",
            "counts": dict(sorted(signer_counts.items())),
        },
        "totals": {
            "videos": len(samples),
            "classes": len(folder_counts),
            "bytes": total_bytes,
            "duration_seconds": round(sum(durations), 3) if durations else None,
            "videos_with_duration": len(durations),
            "duplicate_video_groups": len(duplicates),
        },
        "classes": [
            {"folder": item.folder, "label": item.label, "order": item.order,
             "videos": item.sample_count}
            for item in external
        ],
        "duplicates": duplicates,
        "compatibility": comparison.to_dict(),
        "detr_training_ready": False,
        "blockers": [
            "No bounding-box or segmentation annotations: only video-level "
            "class labels are provided.",
            "No temporal boundaries, so sign windows cannot be localized.",
            "No verified sign-definition documentation for candidate gloss "
            "matches (hello, thankyou).",
            "No train/test split and no authoritative signer/session metadata.",
            "No license or citation file is included in the archive.",
        ],
        "samples": samples,
    }


def extract_archive(archive: Path, output: Path, *, overwrite: bool = False) -> dict:
    """Extract a (possibly nested) archive into ``output``."""
    info = inspect_archive(archive, check_crc=True)
    nested = info["nested_archives"]
    single_nested = info["file_count"] == 1 and len(nested) == 1

    if not single_nested:
        return safe_extract(archive, output, overwrite=overwrite)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="amrita-outer-", dir=str(output.parent))
    )
    try:
        safe_extract(archive, staging)
        inner_path = staging / nested[0]
        inner_info = inspect_archive(inner_path, check_crc=True)
        report = safe_extract(inner_path, output, overwrite=overwrite)
        report["outer_archive"] = info
        report["inner_archive"] = inner_info
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract_frame(video: Path, destination: Path, position: float = 0.5) -> dict:
    """Extract one frame at a fractional position; never invents annotations."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is present
        raise ArchiveError("OpenCV is required for frame extraction") from exc
    if not 0.0 <= position < 1.0:
        raise ValueError("frame position must be in [0, 1)")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return {"ok": False, "error": "could not open video"}
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = int(count * position) if count > 0 else 0
        if target > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            return {"ok": False, "error": "could not decode frame"}
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), frame):
            return {"ok": False, "error": "could not write frame"}
        height, width = frame.shape[:2]
        return {"ok": True, "width": width, "height": height, "frame_index": target}
    finally:
        capture.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to the Amrita ZIP (default: dataset/ or dataset_zip/)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Extraction directory"
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect the archive and write the report without extracting",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Reuse an existing extraction and rebuild the manifest only",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty extraction directory",
    )
    parser.add_argument(
        "--no-probe", action="store_true", help="Skip ffprobe metadata collection"
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="JSON file with verified sign-definition evidence",
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_PATH, help="Output JSON report"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optionally extract one frame per video into this directory",
    )
    parser.add_argument(
        "--frame-classes",
        nargs="*",
        default=None,
        help="Only extract frames for these class labels",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit manifest/frame work (useful for smoke tests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        archive = args.archive or find_default_archive()
        if args.max_videos is not None and args.max_videos < 1:
            raise ValueError("--max-videos must be at least 1")
        archive_info = inspect_archive(archive, check_crc=True)
        project_classes = get_classes()
        evidence = {}
        if args.evidence:
            evidence = json.loads(args.evidence.read_text(encoding="utf-8"))

        if args.inspect_only:
            payload = {
                "schema_version": 1,
                "generated_at": utc_now(),
                "mode": "inspect_only",
                "archive": archive_info,
                "environment": environment_info(),
                "detr_training_ready": False,
                "blockers": [
                    "Archive inspected only; not extracted.",
                    "No annotations or documentation are present in the archive.",
                ],
            }
            write_json(args.report, payload)
            print(f"Wrote {args.report}")
            return 0

        if args.manifest_only:
            if not args.output.is_dir():
                raise ArchiveError(
                    f"--manifest-only requested but {args.output} does not exist"
                )
            extraction = {
                "mode": "reused_existing_extraction",
                "extracted_file_count": None,
                "existing_file_count": sum(
                    1 for path in args.output.rglob("*") if path.is_file()
                ),
                "extracted_bytes": sum(
                    path.stat().st_size
                    for path in args.output.rglob("*")
                    if path.is_file()
                ),
            }
        else:
            extraction = extract_archive(
                archive, args.output, overwrite=args.overwrite
            )
        manifest = build_manifest(
            args.output,
            archive_info=archive_info,
            project_classes=project_classes,
            evidence=evidence,
            probe=not args.no_probe,
        )
        if args.max_videos:
            manifest["samples"] = manifest["samples"][: args.max_videos]
            manifest["totals"]["videos_manifested"] = len(manifest["samples"])

        frame_results = []
        if args.frames_dir:
            selected = manifest["samples"]
            if args.frame_classes:
                wanted = set(args.frame_classes)
                selected = [
                    sample
                    for sample in selected
                    if sample["class_folder"].split(".", 1)[-1] in wanted
                ]
            for sample in selected:
                source = args.output / sample["relative_path"]
                destination = (
                    args.frames_dir
                    / sample["class_folder"].split(".", 1)[-1]
                    / f"{sample['id']}.jpg"
                )
                result = extract_frame(source, destination, position=0.5)
                frame_results.append(
                    {
                        "sample_id": sample["id"],
                        "frame": str(destination) if result["ok"] else None,
                        "ok": result["ok"],
                        "error": result.get("error"),
                        "width": result.get("width"),
                        "height": result.get("height"),
                    }
                )
            manifest["frame_extraction"] = {
                "requested": len(selected),
                "succeeded": sum(1 for item in frame_results if item["ok"]),
                "failed": sum(1 for item in frame_results if not item["ok"]),
                "frames_dir": str(args.frames_dir),
                "note": (
                    "Frames are for inspection only. They have no bounding "
                    "boxes and are not DETR training data."
                ),
                "results": frame_results,
            }

        manifest_path = args.output / MANIFEST_NAME
        write_json(manifest_path, manifest)
        summary = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "archive": archive_info,
            "extraction": {
                "destination": str(args.output),
                "mode": extraction.get("mode", "extracted"),
                "extracted_file_count": extraction.get("extracted_file_count"),
                "existing_file_count": extraction.get("existing_file_count"),
                "extracted_bytes": extraction["extracted_bytes"],
            },
            "manifest": str(manifest_path),
            "totals": manifest["totals"],
            "annotation_status": manifest["annotation_status"],
            "compatibility": manifest["compatibility"],
            "frame_extraction": (
                {
                    key: manifest["frame_extraction"][key]
                    for key in (
                        "requested",
                        "succeeded",
                        "failed",
                        "frames_dir",
                    )
                }
                if manifest.get("frame_extraction")
                else None
            ),
            "detr_training_ready": False,
            "blockers": manifest["blockers"],
            "environment": environment_info(),
        }
        write_json(args.report, summary)
        if extraction.get("extracted_file_count") is not None:
            print(
                f"Extracted {extraction['extracted_file_count']} files to "
                f"{args.output}"
            )
        else:
            print(
                f"Reused {extraction['existing_file_count']} existing files in "
                f"{args.output}"
            )
        print(f"Wrote {manifest_path}")
        print(f"Wrote {args.report}")
        print("DETR training ready: no (video-level labels only, no boxes)")
        return 0
    except (
        ArchiveError,
        CompatibilityError,
        ConfigError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
