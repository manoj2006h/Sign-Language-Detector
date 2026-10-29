"""Benchmark inference latency, preprocessing, postprocessing, and memory.

Measurements exclude camera capture and display. CUDA timings synchronize the
device around every stage. The benchmark warms the model before recording, and
runs on CPU by default when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import DatasetError, DETRData, build_eval_transform
from model import DETR
from utils.checkpoint import DEFAULT_CHECKPOINT_PATH, CheckpointError
from utils.device import resolve_device
from utils.inference import postprocess_detections, preprocess_frame
from utils.reports import environment_info, markdown_table, utc_now, write_json, write_text
from utils.setup import ConfigError, get_classes

DEFAULT_IMAGE_SIZE = 224


def percentiles(values: list[float]) -> dict:
    """Summarize a list of millisecond measurements."""
    if not values:
        raise ValueError("Cannot summarize an empty list of measurements")
    if any(value < 0 for value in values):
        raise ValueError("Latency measurements must be non-negative")
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timestamp(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def benchmark(
    model: DETR,
    image: np.ndarray,
    *,
    device: torch.device,
    class_names: list[str],
    runs: int,
    warmup: int,
    batch_size: int,
    threshold: float,
) -> dict:
    """Time preprocessing, forward pass, postprocessing, and end-to-end."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")

    transform = build_eval_transform(DEFAULT_IMAGE_SIZE, with_bboxes=False)
    height, width = image.shape[:2]
    images = [image] * batch_size
    sizes = [(width, height)] * batch_size
    pixel_tensors = [
        transform(image=item)["image"] for item in images
    ]
    batch = torch.stack(pixel_tensors).to(device)

    def run_once() -> dict:
        start = _timestamp(device)
        tensors = []
        for item in images:
            tensors.append(preprocess_frame(item, transform))
        prepared = torch.cat(tensors, dim=0).to(device)
        after_preprocess = _timestamp(device)

        with torch.inference_mode():
            output = model(prepared)
        after_forward = _timestamp(device)

        postprocess_detections(output, sizes, class_names, threshold)
        after_postprocess = _timestamp(device)
        return {
            "preprocess_ms": (after_preprocess - start) * 1000.0,
            "forward_ms": (after_forward - after_preprocess) * 1000.0,
            "postprocess_ms": (after_postprocess - after_forward) * 1000.0,
            "end_to_end_ms": (after_postprocess - start) * 1000.0,
        }

    # Use the prebuilt batch during warmup so autograd/caching effects settle.
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_before = process.memory_info().rss
    except ImportError:  # pragma: no cover
        process = None
        rss_before = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples: list[dict] = []
    peak_rss = rss_before or 0
    wall_start = _timestamp(device)
    for _ in range(runs):
        samples.append(run_once())
        if process is not None:
            peak_rss = max(peak_rss, process.memory_info().rss)
    wall_elapsed = _timestamp(device) - wall_start

    stages = {
        stage: percentiles([sample[stage] for sample in samples])
        for stage in ("preprocess_ms", "forward_ms", "postprocess_ms", "end_to_end_ms")
    }
    median_e2e = stages["end_to_end_ms"]["median_ms"]
    fps = (1000.0 * batch_size / median_e2e) if median_e2e > 0 else None
    throughput = (runs * batch_size / wall_elapsed) if wall_elapsed > 0 else None

    memory = {
        "rss_before_mb": round(rss_before / (1024 * 1024), 2)
        if rss_before is not None
        else None,
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 2)
        if peak_rss
        else None,
        "cpu_peak_rss_method": "sampled after each run with psutil",
    }
    if device.type == "cuda":
        memory.update(
            {
                "cuda_peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2
                ),
                "cuda_peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2
                ),
            }
        )

    return {
        "stages": stages,
        "fps": round(fps, 3) if fps is not None else None,
        "images_per_second": round(throughput, 3) if throughput is not None else None,
        "fps_method": (
            "batch_size * 1000 / median end-to-end latency for one batch; "
            "camera capture and display excluded"
        ),
        "runs": runs,
        "warmup": warmup,
        "batch_size": batch_size,
        "resolution": [width, height],
        "model_input_size": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        "memory": memory,
        "wall_seconds": round(wall_elapsed, 4),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Trusted legacy checkpoints only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be in [0, 1]")
        classes = get_classes()
        device = resolve_device(args.device)
        dataset = DETRData(args.data_dir, train=False, classes=classes, verbose=False)
        if not dataset.samples:
            raise DatasetError(f"No images found in {args.data_dir}")
        image_path = args.image or dataset.samples[0][0]
        image = load_rgb(image_path)

        model = DETR(num_classes=len(classes), backbone_weights=None, verbose=False)
        model = model.to(device)
        model.eval()
        metadata = model.load_pretrained(
            str(args.checkpoint),
            classes=classes,
            map_location=device,
            allow_unsafe_load=args.allow_unsafe_load,
        )

        result = benchmark(
            model,
            image,
            device=device,
            class_names=classes,
            runs=args.runs,
            warmup=args.warmup,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
                "epoch": metadata.get("epoch"),
                "classes": metadata.get("classes", classes),
            },
            "image": {
                "path": str(image_path),
                "resolution": result["resolution"],
            },
            "device": str(device),
            "environment": environment_info(device),
            "settings": {
                "runs": args.runs,
                "warmup": args.warmup,
                "batch_size": args.batch_size,
                "threshold": args.threshold,
                "model_input_size": result["model_input_size"],
                "camera_and_display_excluded": True,
            },
            "results": result,
        }
        write_json(args.output, report)

        rows = [
            ("Device", str(device)),
            ("Checkpoint", f"`{args.checkpoint}`"),
            ("Checkpoint SHA-256", f"`{report['checkpoint']['sha256']}`"),
            ("Runs / warmup", f"{args.runs} / {args.warmup}"),
            ("Batch size", str(args.batch_size)),
            ("Input resolution", f"{result['resolution'][0]}x{result['resolution'][1]}"),
            ("Median end-to-end", f"{result['stages']['end_to_end_ms']['median_ms']} ms"),
            ("p95 end-to-end", f"{result['stages']['end_to_end_ms']['p95_ms']} ms"),
            ("Median model forward", f"{result['stages']['forward_ms']['median_ms']} ms"),
            ("FPS (batch/median e2e)", str(result["fps"])),
            ("Peak RSS", f"{result['memory']['peak_rss_mb']} MB"),
        ]
        markdown = "\n".join(
            [
                "# Runtime benchmark",
                "",
                markdown_table(rows),
                "",
                "Camera capture and display are excluded. CUDA stages are "
                "synchronized. Results are machine-specific and were actually "
                "measured; they are not extrapolated.",
                "",
            ]
        )
        write_text(args.output.with_name("benchmark_summary.md"), markdown)
        print(
            "Median end-to-end: "
            f"{result['stages']['end_to_end_ms']['median_ms']} ms "
            f"(p95 {result['stages']['end_to_end_ms']['p95_ms']} ms), "
            f"FPS {result['fps']}"
        )
        print(f"Wrote {args.output}")
        return 0
    except (CheckpointError, DatasetError, ConfigError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
