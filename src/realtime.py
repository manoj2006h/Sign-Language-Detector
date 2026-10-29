"""Webcam inference with the same RGB preprocessing used for evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from data import build_eval_transform
from model import DETR
from utils.checkpoint import DEFAULT_CHECKPOINT_PATH, CheckpointError
from utils.device import resolve_device
from utils.inference import postprocess_detections, preprocess_frame
from utils.logger import get_logger
from utils.setup import get_classes, get_colors


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or returns an invalid frame."""


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            f"Could not open camera index {index}. Check the camera connection, "
            "permissions, or pass --camera N."
        )
    return capture


def _draw_detections(
    frame, detections: list[dict], colors: list[list[int]]
) -> None:
    height, width = frame.shape[:2]
    thickness = max(2, int(height / 300))
    font_scale = max(0.6, height / 1200)
    font_thickness = max(1, int(height / 500))
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
        class_id = int(detection["class_id"])
        rgb = colors[class_id % len(colors)]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)
        text = f"{detection['class_name']} - {detection['confidence']:.3f}"
        label_height = max(24, int(40 * font_scale))
        cv2.rectangle(frame, (x1, max(0, y1 - label_height)), (x2, y1), bgr, -1)
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )


def run_capture(
    capture: Any,
    model: Any,
    transform: Any,
    class_names: list[str],
    colors: list[list[int]],
    device: torch.device,
    *,
    threshold: float = 0.8,
    max_frames: int | None = None,
    display: bool = True,
    draw: bool = True,
    logger: Any = None,
) -> int:
    """Run the capture loop; returns the number of processed frames."""
    frame_count = 0
    fps_start = time.time()
    try:
        while max_frames is None or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                if frame_count == 0:
                    raise CameraError(
                        "Camera opened but could not read a valid frame. "
                        "Check camera permissions or try another --camera index."
                    )
                if logger is not None:
                    logger.error("Failed to read a frame; stopping capture.")
                break

            started = time.time()
            try:
                tensor = preprocess_frame(frame, transform).to(device)
            except ValueError as exc:
                raise CameraError(f"Invalid camera frame: {exc}") from exc
            with torch.inference_mode():
                output = model(tensor)
            inference_ms = (time.time() - started) * 1000

            height, width = frame.shape[:2]
            detections = postprocess_detections(
                output, [(width, height)], class_names, threshold
            )[0]
            if draw:
                _draw_detections(frame, detections, colors)

            frame_count += 1
            if frame_count % 30 == 0 and logger is not None:
                elapsed = time.time() - fps_start
                fps = 30 / elapsed if elapsed > 0 else 0.0
                logger.detection(
                    f"frame {frame_count}: {len(detections)} detection(s), "
                    f"{inference_ms:.1f} ms, {fps:.1f} FPS"
                )
                fps_start = time.time()

            if display:
                cv2.imshow("Sign Language Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if logger is not None:
                        logger.realtime("Stopping real-time detection...")
                    break
    finally:
        capture.release()
    return frame_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Trained task checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames (smoke test)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Do not open an OpenCV window"
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow loading a trusted legacy checkpoint (unsafe for untrusted files)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger("realtime")
    logger.print_banner()
    try:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be between 0 and 1")
        if args.max_frames is not None and args.max_frames < 0:
            raise ValueError("--max-frames must be non-negative")
        classes = get_classes()
        colors = get_colors()
        device = resolve_device(args.device)
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
        transform = build_eval_transform(with_bboxes=False)
        capture = open_camera(args.camera)
        logger.realtime(f"Camera {args.camera} opened; press q to quit.")
        count = run_capture(
            capture,
            model,
            transform,
            classes,
            colors,
            device,
            threshold=args.threshold,
            max_frames=args.max_frames,
            display=not args.no_display,
            logger=logger,
        )
        logger.realtime(f"Processed {count} frame(s).")
        return 0
    except (CameraError, CheckpointError, RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 2
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
