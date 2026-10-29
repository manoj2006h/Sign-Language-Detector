import unittest
from unittest import mock

import numpy as np
import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from data import build_eval_transform
from realtime import CameraError, open_camera, run_capture


class FakeCameraHandle:
    def __init__(self, opened):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class ConstantModel:
    def __call__(self, tensor):
        return {
            "pred_logits": torch.tensor([[[10.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        }


class RealtimeFailureTests(unittest.TestCase):
    def test_open_camera_reports_the_requested_index(self):
        handle = FakeCameraHandle(opened=True)
        with mock.patch("realtime.cv2.VideoCapture", return_value=handle) as factory:
            capture = open_camera(3)
        self.assertIs(capture, handle)
        factory.assert_called_once_with(3)

    def test_open_camera_releases_a_failed_handle(self):
        handle = FakeCameraHandle(opened=False)
        with mock.patch("realtime.cv2.VideoCapture", return_value=handle):
            with self.assertRaises(CameraError) as ctx:
                open_camera(7)
        self.assertTrue(handle.released)
        self.assertIn("7", str(ctx.exception))

    def test_first_frame_failure_is_a_clear_camera_error(self):
        with self.assertRaises(CameraError) as ctx:
            run_capture(
                FakeCapture([]),
                model=None,
                transform=None,
                class_names=["a"],
                colors=[[255, 0, 0]],
                device=torch.device("cpu"),
                display=False,
            )
        self.assertIn("frame", str(ctx.exception).lower())

    def test_invalid_frame_is_rejected(self):
        with self.assertRaises(CameraError):
            run_capture(
                FakeCapture([None]),
                model=None,
                transform=None,
                class_names=["a"],
                colors=[[255, 0, 0]],
                device=torch.device("cpu"),
                display=False,
            )

    def test_invalid_frame_after_first_frame_stops_and_releases(self):
        transform = build_eval_transform(image_size=224, with_bboxes=False)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        capture = FakeCapture([frame, None, frame])
        count = run_capture(
            capture,
            model=ConstantModel(),
            transform=transform,
            class_names=["a"],
            colors=[[255, 0, 0]],
            device=torch.device("cpu"),
            threshold=0.5,
            display=False,
            draw=False,
        )
        self.assertEqual(count, 1)
        self.assertTrue(capture.released)

    def test_single_frame_loop_uses_actual_frame_shape(self):
        transform = build_eval_transform(image_size=224, with_bboxes=False)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        count = run_capture(
            FakeCapture([frame]),
            model=ConstantModel(),
            transform=transform,
            class_names=["a"],
            colors=[[255, 0, 0]],
            device=torch.device("cpu"),
            threshold=0.5,
            max_frames=1,
            display=False,
            draw=False,
        )
        self.assertEqual(count, 1)

    def test_stop_after_max_frames_releases_capture(self):
        transform = build_eval_transform(image_size=224, with_bboxes=False)
        capture = FakeCapture([np.zeros((100, 200, 3), dtype=np.uint8)] * 3)
        count = run_capture(
            capture,
            model=ConstantModel(),
            transform=transform,
            class_names=["a"],
            colors=[[255, 0, 0]],
            device=torch.device("cpu"),
            threshold=0.5,
            max_frames=2,
            display=False,
            draw=False,
        )
        self.assertEqual(count, 2)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
