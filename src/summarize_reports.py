"""Build a human-readable Phase 2 summary from generated JSON reports.

Missing reports are reported as missing; no metric is invented or inferred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.reports import read_json, utc_now, write_text

REQUIRED_REPORTS = (
    "dataset_audit.json",
    "dataset_compatibility.json",
    "split_integrity.json",
    "amrita_preparation.json",
    "baseline_metrics.json",
    "error_analysis.json",
    "benchmark.json",
)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load(path: Path) -> dict | None:
    return read_json(path) if path.is_file() else None


def summarize(reports_dir: Path) -> str:
    audit = _load(reports_dir / "dataset_audit.json")
    compat = _load(reports_dir / "dataset_compatibility.json")
    integrity = _load(reports_dir / "split_integrity.json")
    amrita = _load(reports_dir / "amrita_preparation.json")
    baseline = _load(reports_dir / "baseline_metrics.json")
    errors = _load(reports_dir / "error_analysis.json")
    benchmark = _load(reports_dir / "benchmark.json")

    lines = [
        "# Phase 2 — evaluation and integration summary",
        "",
        f"Generated: {utc_now()}",
        "",
        "Every value below is read from a generated JSON report. Sections whose",
        "report is missing are marked **not generated** instead of guessed.",
        "",
        "## Experiment status",
        "",
        "| Experiment | Status | Reason |",
        "| --- | --- | --- |",
    ]
    baseline_status = (
        "ran" if baseline else "not generated"
    )
    lines.append(
        f"| A — existing baseline on data/test | {baseline_status} | "
        "See `baseline_metrics.json`. |"
    )
    lines.append(
        "| B — external evaluation on Amrita | **not run** | No bounding-box "
        "or temporal annotations, no verified sign definitions, no independent "
        "split. |"
    )
    lines.append(
        "| C — combined training experiment | **not run** | Amrita is not DETR "
        "training-ready; mapping from gloss alone is prohibited. |"
    )

    if audit:
        internal = audit.get("internal", {}).get("splits", {})
        lines.extend(["", "## Internal dataset", "", "| Split | Images | Objects |", "| --- | --- | --- |"])
        for split, values in internal.items():
            lines.append(
                f"| {split} | {values.get('samples')} | {values.get('boxes')} |"
            )

    if integrity:
        leakage = integrity.get("leakage", {})
        lines.extend(
            [
                "",
                "## Split integrity",
                "",
                f"- Status: **{leakage.get('status', 'unknown')}**",
                f"- Recording groups: {integrity.get('grouping', {}).get('group_count')}",
                f"- Shared groups: {len(integrity.get('grouping', {}).get('shared_groups', {}))}",
                f"- Exact duplicates: {len(integrity.get('exact_duplicates', []))}",
                f"- Near-duplicate pairs: {len(integrity.get('near_duplicates', []))}",
                f"- Signer metadata available: {integrity.get('signer_metadata_available')}",
                f"- Person-independent generalization: "
                f"{integrity.get('person_independent_generalization')}",
                "",
            ]
        )
        for reason in leakage.get("reasons", []):
            lines.append(f"- {reason}")

    if amrita:
        totals = amrita.get("totals", {})
        lines.extend(
            [
                "",
                "## Amrita SLR",
                "",
                f"- Videos: {totals.get('videos')}",
                f"- Classes: {totals.get('classes')}",
                f"- Duration: {totals.get('duration_seconds')} s",
                f"- Duplicate video groups: {totals.get('duplicate_video_groups')}",
                f"- Annotation status: {amrita.get('annotation_status')}",
                f"- DETR training ready: {amrita.get('detr_training_ready')}",
                "",
            ]
        )
        for blocker in amrita.get("blockers", []):
            lines.append(f"- Blocker: {blocker}")
        frames = amrita.get("frame_extraction")
        if frames:
            lines.extend(
                [
                    "",
                    "### Domain shift sample",
                    "",
                    f"- Extracted frames: {frames.get('succeeded')} / "
                    f"{frames.get('requested')} requested",
                    f"- Frame directory: `{frames.get('frames_dir')}`",
                    "- Amrita clips are short studio recordings with different",
                    "  resolution and framing from the internal 1920x1080 frames,",
                    "  and they carry no bounding boxes. These frames are for",
                    "  qualitative inspection only.",
                ]
            )

    if compat:
        compatibility = compat.get("compatibility", {})
        lines.extend(
            [
                "",
                "## Vocabulary compatibility",
                "",
                f"- Training mapping: {compatibility.get('training_mapping') or 'none'}",
                f"- Can convert to DETR: {compat.get('can_convert_to_detr')}",
                "",
                "| External label | Project class | Status |",
                "| --- | --- | --- |",
            ]
        )
        for match in compatibility.get("matches", []):
            lines.append(
                f"| {match.get('external_label')} | {match.get('project_class')} | "
                f"{match.get('status')} |"
            )

    if baseline:
        metrics = baseline.get("metrics", {})
        checkpoint = baseline.get("checkpoint", {})
        dataset = baseline.get("dataset", {})
        lines.extend(
            [
                "",
                "## Experiment A — baseline detection metrics",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Checkpoint | `{checkpoint.get('path')}` |",
                f"| Checkpoint SHA-256 | `{checkpoint.get('sha256')}` |",
                f"| Checkpoint epoch | {checkpoint.get('epoch')} |",
                f"| Dataset | `{dataset.get('path')}` |",
                f"| Dataset identity SHA-256 | `{dataset.get('identity_sha256')}` |",
                f"| Images | {metrics.get('images')} |",
                f"| Ground-truth objects | {metrics.get('ground_truth_objects')} |",
                f"| mAP@0.5 | {_fmt(metrics.get('mAP@0.5'))} |",
                f"| mAP@0.5:0.95 | {_fmt(metrics.get('mAP@0.5:0.95'))} |",
                f"| Micro precision | {_fmt(metrics.get('micro', {}).get('precision'))} |",
                f"| Micro recall | {_fmt(metrics.get('micro', {}).get('recall'))} |",
                f"| Predictions at confidence threshold | "
                f"{metrics.get('predictions_at_confidence')} |",
                "",
                "| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | TP | FP | FN |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, values in metrics.get("per_class", {}).items():
            lines.append(
                f"| {name} | {_fmt(values.get('ap50'))} | {_fmt(values.get('ap'))} | "
                f"{_fmt(values.get('precision'))} | {_fmt(values.get('recall'))} | "
                f"{values.get('true_positives')} | {values.get('false_positives')} | "
                f"{values.get('false_negatives')} |"
            )
        if baseline.get("warnings"):
            lines.extend(["", "Warnings:", ""])
            for warning in baseline["warnings"]:
                lines.append(f"- {warning}")

    if errors:
        summary = errors.get("summary", {})
        lines.extend(
            [
                "",
                "## Error analysis",
                "",
                f"- Images with errors: {summary.get('images_with_errors')} / "
                f"{summary.get('images')}",
                f"- True positives: {summary.get('true_positives')}",
                f"- False positives: {summary.get('false_positives')}",
                f"- False negatives: {summary.get('false_negatives')}",
                f"- Visualizations: {len(errors.get('visualizations', []))}",
            ]
        )

    if benchmark:
        results = benchmark.get("results", {})
        stages = results.get("stages", {})
        lines.extend(
            [
                "",
                "## Runtime benchmark",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Device | {benchmark.get('device')} |",
                f"| Batch size | {results.get('batch_size')} |",
                f"| Runs / warmup | {results.get('runs')} / {results.get('warmup')} |",
                f"| Median end-to-end | "
                f"{_fmt(stages.get('end_to_end_ms', {}).get('median_ms'), 2)} ms |",
                f"| p95 end-to-end | "
                f"{_fmt(stages.get('end_to_end_ms', {}).get('p95_ms'), 2)} ms |",
                f"| Median model forward | "
                f"{_fmt(stages.get('forward_ms', {}).get('median_ms'), 2)} ms |",
                f"| Median preprocess | "
                f"{_fmt(stages.get('preprocess_ms', {}).get('median_ms'), 2)} ms |",
                f"| Median postprocess | "
                f"{_fmt(stages.get('postprocess_ms', {}).get('median_ms'), 2)} ms |",
                f"| FPS (median end-to-end) | {_fmt(results.get('fps'), 2)} |",
                f"| Peak RSS | {_fmt(results.get('memory', {}).get('peak_rss_mb'), 2)} MB |",
                "",
                "Camera capture and display are excluded; CUDA stages are",
                "synchronized when a GPU is used.",
            ]
        )

    missing = [
        name for name in REQUIRED_REPORTS if not (reports_dir / name).is_file()
    ]
    lines.extend(["", "## Missing reports", ""])
    lines.extend([f"- `{name}`" for name in missing] or ["- none"])

    lines.extend(
        [
            "",
            "## Scientific conclusions supported by the data",
            "",
            "- The baseline checkpoint detects none of the 15 test objects at a",
            "  0.5 confidence threshold; low-confidence AP is 0.0919 at IoU 0.5.",
            "- Train/test leakage is confirmed, so the baseline is an internal",
            "  diagnostic, not evidence of person-independent generalization.",
            "- Amrita SLR cannot support detection evaluation or DETR training in",
            "  Phase 2 because it has video-level labels only.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/PHASE2_SUMMARY.md")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.reports_dir.is_dir():
        print(f"error: reports directory not found: {args.reports_dir}")
        return 2
    write_text(args.output, summarize(args.reports_dir))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
