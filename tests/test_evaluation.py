import unittest

import torch
from torch.utils.data import DataLoader

from tests.helpers import ROOT, SRC  # noqa: F401

from test import evaluate
from utils.boxes import stacker


class FakeModel:
    """Deterministic output for evaluation-loop tests."""

    def __init__(self, num_classes=2, num_queries=2):
        self.num_classes = num_classes
        self.num_queries = num_queries

    def eval(self):
        return self

    def __call__(self, images):
        batch_size = images.shape[0]
        logits = torch.full(
            (batch_size, self.num_queries, self.num_classes + 1), -10.0
        )
        logits[..., -1] = 10.0  # every query is background
        boxes = torch.full(
            (batch_size, self.num_queries, 4), 0.25, dtype=torch.float32
        )
        return {"pred_logits": logits, "pred_boxes": boxes}


class FakeCriterion:
    def __init__(self, weight_dict):
        self.weight_dict = weight_dict

    def __call__(self, output, targets):
        batch_size = output["pred_logits"].shape[0]
        return {
            "labels": {"loss_ce": torch.tensor(float(batch_size))},
            "boxes": {
                "loss_bbox": torch.tensor(0.0),
                "loss_giou": torch.tensor(0.0),
            },
        }


def _sample():
    return (
        torch.zeros(3, 8, 8),
        {
            "labels": torch.empty(0, dtype=torch.long),
            "boxes": torch.empty(0, 4, dtype=torch.float32),
        },
    )


class EvaluationLoopTests(unittest.TestCase):
    def test_empty_loader_returns_no_loss_instead_of_dividing_by_zero(self):
        result = evaluate(
            FakeModel(),
            DataLoader([], batch_size=2, collate_fn=stacker),
            FakeCriterion({"class_weighting": 1, "bbox_weighting": 1, "giou_weighting": 1}),
            torch.device("cpu"),
            threshold=0.8,
        )
        self.assertIsNone(result["loss"])
        self.assertEqual(result["batches"], 0)
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["detections"], 0)

    def test_incomplete_final_batch_is_evaluated(self):
        loader = DataLoader(
            [_sample(), _sample(), _sample()], batch_size=2, collate_fn=stacker
        )
        result = evaluate(
            FakeModel(),
            loader,
            FakeCriterion({"class_weighting": 1, "bbox_weighting": 1, "giou_weighting": 1}),
            torch.device("cpu"),
            threshold=0.8,
        )
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["samples"], 3)
        self.assertIsNotNone(result["loss"])
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_evaluation_uses_the_criterion_weight_dict(self):
        loader = DataLoader([_sample()], batch_size=1, collate_fn=stacker)
        criterion = FakeCriterion(
            {"class_weighting": 3, "bbox_weighting": 5, "giou_weighting": 2}
        )
        result = evaluate(
            FakeModel(), loader, criterion, torch.device("cpu"), threshold=0.8
        )
        self.assertAlmostEqual(result["loss"], 3.0)


if __name__ == "__main__":
    unittest.main()
