import unittest

import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from loss import DETRLoss, HungarianMatcher


def make_criterion(num_classes=2):
    weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
    return DETRLoss(
        num_classes=num_classes,
        matcher=HungarianMatcher(weights),
        weight_dict=weights,
        eos_coef=0.1,
    )


class LossTests(unittest.TestCase):
    def test_loss_is_finite_and_backpropagates_for_normal_targets(self):
        criterion = make_criterion()
        logits = torch.randn(2, 4, 3, requires_grad=True)
        boxes = torch.rand(2, 4, 4, requires_grad=True)
        targets = [
            {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
            {"labels": torch.tensor([1, 0]), "boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]])},
        ]
        losses = criterion({"pred_logits": logits, "pred_boxes": boxes}, targets)
        total = (
            losses["labels"]["loss_ce"]
            + losses["boxes"]["loss_bbox"]
            + losses["boxes"]["loss_giou"]
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(boxes.grad)

    def test_batch_with_no_targets_is_finite(self):
        criterion = make_criterion()
        logits = torch.randn(2, 4, 3, requires_grad=True)
        boxes = torch.rand(2, 4, 4, requires_grad=True)
        targets = [
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        losses = criterion({"pred_logits": logits, "pred_boxes": boxes}, targets)
        total = (
            losses["labels"]["loss_ce"]
            + losses["boxes"]["loss_bbox"]
            + losses["boxes"]["loss_giou"]
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(boxes.grad)

    def test_target_label_outside_configured_classes_is_rejected(self):
        criterion = make_criterion(num_classes=2)
        logits = torch.randn(1, 4, 3)
        boxes = torch.rand(1, 4, 4)
        targets = [
            {"labels": torch.tensor([2]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
        ]
        with self.assertRaises(ValueError):
            criterion({"pred_logits": logits, "pred_boxes": boxes}, targets)


if __name__ == "__main__":
    unittest.main()
