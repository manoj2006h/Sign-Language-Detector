"""Capture unlabeled camera images for a chosen class.

Captured images are written to an image directory (default:
``data/train/images``). Labels must still be created in the matching
``data/train/labels`` directory using the documented YOLO format; this tool
does not invent annotations.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid

import cv2

from utils.logger import logger
from utils.setup import get_classes


class CaptureImages:
    def __init__(self, path: str, classes: list[str], camera_id: int) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        logger.print_banner()
        logger.capture("Image capture system initialized")
        if not self.cap.isOpened():
            self.cap.release()
            logger.capture_error("Camera", f"Could not open camera {camera_id}")
            raise RuntimeError(f"Could not open camera {camera_id}")
        logger.success(f"Camera {camera_id} connected successfully")
        os.makedirs(self.path, exist_ok=True)
        logger.info(f"Output directory: {self.path}")

    def capture(self, class_name: str) -> bool:
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.capture_error(class_name, "Failed to read a valid frame from camera")
            return False

        raw_frame = frame.copy()
        image = cv2.putText(
            frame,
            f"Capturing {class_name}",
            (0, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Image Capture", image)

        filename = f"{class_name}-{uuid.uuid1()}.jpg"
        filepath = os.path.join(self.path, filename)
        if not cv2.imwrite(filepath, raw_frame):
            logger.capture_error(class_name, f"Could not write {filepath}")
            return False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.warning("Quit key pressed - stopping capture")
            return False
        return True

    def run(self, sleep_time: float = 1, num_images: int = 10) -> int:
        logger.capture_session_start(self.classes, num_images, sleep_time)
        total_captured = 0
        try:
            for img_class in self.classes:
                logger.capture_class_start(img_class, num_images)
                with logger.create_capture_progress(num_images, img_class) as progress:
                    task = progress.add_task(f"Capturing {img_class}", total=num_images)
                    for idx in range(num_images):
                        if self.capture(img_class):
                            total_captured += 1
                            logger.capture_success(img_class, idx + 1)
                        else:
                            logger.capture_error(img_class, f"Image {idx + 1}")
                        progress.update(task, advance=1)
                        time.sleep(sleep_time)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released and windows closed")
        logger.capture_session_complete(total_captured, len(self.classes))
        return total_captured


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/train/images",
        help="Directory for captured images (default: data/train/images)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        classes = get_classes()
        capture = CaptureImages(args.output, classes, args.camera)
        capture.run(sleep_time=args.sleep, num_images=args.num_images)
        print(
            "Capture complete. Add matching .txt annotations in the corresponding "
            "labels directory before training."
        )
        return 0
    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
