import unittest

import numpy as np
import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from data import build_eval_transform
from utils.inference import (
    average_or_none,
    compute_loss_total,
    count_detections,
    postprocess_detections,
    preprocess_frame,
)


class InferencePostprocessTests(unittest.TestCase):
    def _output(self):
        logits = torch.tensor(
            [[[10.0, 0.0, -10.0], [0.0, 10.0, -10.0], [-10.0, -10.0, 10.0]]]
        )
        boxes = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ]
        )
        return {"pred_logits": logits, "pred_boxes": boxes}

    def test_detections_are_scaled_to_actual_frame_size(self):
        detections = postprocess_detections(
            self._output(), [(200, 100)], ["a", "b"], threshold=0.5
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(detections[0]), 2)
        first = detections[0][0]
        self.assertEqual(first["class_name"], "a")
        expected_confidence = torch.softmax(
            self._output()["pred_logits"], dim=-1
        )[0, 0, 0].item()
        self.assertAlmostEqual(first["confidence"], expected_confidence, places=6)
        self.assertTrue(
            np.allclose(first["bbox"], [80.0, 40.0, 120.0, 60.0], atol=1e-4)
        )
        second = detections[0][1]
        self.assertEqual(second["class_name"], "b")
        self.assertTrue(
            np.allclose(second["bbox"], [50.0, 25.0, 150.0, 75.0], atol=1e-4)
        )

    def test_predictions_are_clipped_to_frame(self):
        output = {
            "pred_logits": torch.tensor([[[10.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.01, 0.5, 0.2, 0.2]]]),
        }
        detections = postprocess_detections(
            output, [(200, 100)], ["a"], threshold=0.5
        )
        self.assertEqual(detections[0][0]["bbox"][0], 0.0)
        self.assertLessEqual(detections[0][0]["bbox"][2], 200.0)

    def test_detection_count_uses_same_probability_rule(self):
        self.assertEqual(count_detections(self._output(), threshold=0.5), 2)
        self.assertEqual(count_detections(self._output(), threshold=0.99999), 0)

    def test_no_detections_returns_empty_list(self):
        output = {
            "pred_logits": torch.zeros(1, 2, 2),
            "pred_boxes": torch.rand(1, 2, 4),
        }
        detections = postprocess_detections(
            output, [(200, 100)], ["a"], threshold=0.9
        )
        self.assertEqual(detections, [[]])

    def test_class_name_count_must_match_model_output(self):
        with self.assertRaises(ValueError):
            postprocess_detections(self._output(), [(200, 100)], ["only-one"], 0.5)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                postprocess_detections(
                    self._output(), [(200, 100)], ["a", "b"], threshold=threshold
                )

    def test_average_or_none_handles_empty_evaluation(self):
        self.assertIsNone(average_or_none(0.0, 0))
        self.assertEqual(average_or_none(4.0, 2), 2.0)

    def test_compute_loss_total_uses_declared_weights(self):
        losses = {
            "labels": {"loss_ce": torch.tensor(1.0)},
            "boxes": {"loss_bbox": torch.tensor(2.0), "loss_giou": torch.tensor(3.0)},
        }
        weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
        self.assertEqual(compute_loss_total(losses, weights).item(), 17.0)


class FramePreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.transform = build_eval_transform(image_size=224, with_bboxes=False)

    def test_bgr_frame_is_converted_to_rgb_before_normalization(self):
        # BGR channel 2 is red. If the frame were treated as RGB without
        # conversion, the red channel would land in tensor channel 2 instead.
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        tensor = preprocess_frame(frame, self.transform)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertGreater(tensor[0, 0].mean().item(), tensor[0, 1].mean().item())
        self.assertGreater(tensor[0, 0].mean().item(), tensor[0, 2].mean().item())

    def test_invalid_frame_is_rejected(self):
        for frame in (None, np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                preprocess_frame(frame, self.transform)


if __name__ == "__main__":
    unittest.main()
