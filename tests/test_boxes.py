import unittest

import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.boxes import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    clip_boxes,
    rescale_bboxes,
    stacker,
)


class BoxConversionTests(unittest.TestCase):
    def test_cxcywh_xyxy_roundtrip(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.25, 0.75, 0.1, 0.2]])
        self.assertTrue(
            torch.allclose(
                box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes)), boxes, atol=1e-6
            )
        )

    def test_rescale_uses_width_height_order(self):
        # (cx, cy, w, h) -> x1=80, y1=30, x2=120, y2=70 for 200x100
        box = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        out = rescale_bboxes(box, (200, 100))
        self.assertTrue(
            torch.allclose(out, torch.tensor([[80.0, 30.0, 120.0, 70.0]]))
        )

    def test_rescale_preserves_device_and_empty_shape(self):
        empty = torch.empty((0, 4), dtype=torch.float32)
        out = rescale_bboxes(empty, (640, 480))
        self.assertEqual(out.shape, (0, 4))
        self.assertEqual(out.dtype, torch.float32)

    def test_clip_boxes_to_image_bounds(self):
        boxes = torch.tensor(
            [[-10.0, -5.0, 50.0, 60.0], [190.0, 90.0, 250.0, 140.0]]
        )
        clipped = clip_boxes(boxes, (200, 100))
        self.assertTrue(
            torch.allclose(
                clipped,
                torch.tensor([[0.0, 0.0, 50.0, 60.0], [190.0, 90.0, 200.0, 100.0]]),
            )
        )

    def test_clip_boxes_leaves_boxes_inside_image_unchanged(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        self.assertTrue(torch.equal(clip_boxes(boxes, (200, 100)), boxes))

    def test_clip_boxes_empty(self):
        out = clip_boxes(torch.empty((0, 4)), (10, 20))
        self.assertEqual(out.shape, (0, 4))

    def test_stacker_keeps_variable_targets_as_list(self):
        batch = [
            (torch.zeros(3, 4, 4), {"labels": torch.tensor([0]), "boxes": torch.ones(1, 4)}),
            (
                torch.zeros(3, 4, 4),
                {"labels": torch.tensor([0, 1]), "boxes": torch.ones(2, 4)},
            ),
        ]
        images, targets = stacker(batch)
        self.assertEqual(tuple(images.shape), (2, 3, 4, 4))
        self.assertEqual(len(targets), 2)
        self.assertEqual(tuple(targets[1]["boxes"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
