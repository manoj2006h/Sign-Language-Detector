import unittest

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.detection_metrics import (
    DetectionMetricsError,
    average_precision,
    evaluate_detection,
    match_image_detections,
)


def gt(class_id, box):
    return {"class_id": class_id, "bbox": box}


def pred(class_id, score, box):
    return {"class_id": class_id, "confidence": score, "bbox": box}


BOX_A = [0.1, 0.1, 0.3, 0.3]
BOX_B = [0.5, 0.5, 0.8, 0.8]
CLASSES = ["hello", "thankyou"]


class PerfectPredictionTests(unittest.TestCase):
    def test_perfect_predictions_score_one_at_every_iou(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)], [gt(1, BOX_B)]],
            [[pred(0, 0.9, BOX_A)], [pred(1, 0.8, BOX_B)]],
            CLASSES,
        )
        self.assertAlmostEqual(result["mAP@0.5"], 1.0)
        self.assertAlmostEqual(result["mAP@0.5:0.95"], 1.0)
        self.assertEqual(result["micro"]["true_positives"], 2)
        self.assertEqual(result["micro"]["false_positives"], 0)
        self.assertEqual(result["micro"]["false_negatives"], 0)
        self.assertEqual(result["micro"]["precision"], 1.0)
        self.assertEqual(result["micro"]["recall"], 1.0)

    def test_exact_boxes_remain_perfect_at_high_iou(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)]], [[pred(0, 0.9, BOX_A)]], CLASSES
        )
        self.assertEqual(result["ap_by_iou"]["0.95"]["hello"], 1.0)


class IncorrectClassTests(unittest.TestCase):
    def test_wrong_class_is_a_false_negative_and_false_positive(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)]], [[pred(1, 0.9, BOX_A)]], CLASSES
        )
        self.assertEqual(result["mAP@0.5"], 0.0)
        hello = result["per_class"]["hello"]
        self.assertEqual(hello["false_negatives"], 1)
        self.assertEqual(hello["true_positives"], 0)
        self.assertIsNone(result["per_class"]["thankyou"]["ap50"])
        self.assertIn("thankyou", result["undefined_classes"])
        self.assertEqual(result["class_confusion"]["hello"]["thankyou"], 1)

    def test_detection_matching_requires_class_equality(self):
        matching = match_image_detections(
            [gt(0, BOX_A)], [pred(1, 0.9, BOX_A)], CLASSES
        )
        self.assertEqual(matching["matches"], [])
        self.assertEqual(matching["false_positive_indices"], [0])
        self.assertEqual(matching["unmatched_ground_truth"], [0])


class LocalizationTests(unittest.TestCase):
    def test_misplaced_box_has_zero_ap(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)]], [[pred(0, 0.9, BOX_B)]], CLASSES
        )
        self.assertEqual(result["mAP@0.5"], 0.0)
        self.assertEqual(result["micro"]["false_negatives"], 1)

    def test_iou_threshold_boundary(self):
        ground = [[gt(0, [0.0, 0.0, 1.0, 1.0])]]
        below = evaluate_detection(
            ground, [[pred(0, 0.9, [0.0, 0.0, 1.0, 0.45])]], CLASSES
        )
        above = evaluate_detection(
            ground, [[pred(0, 0.9, [0.0, 0.0, 1.0, 0.55])]], CLASSES
        )
        self.assertEqual(below["mAP@0.5"], 0.0)
        self.assertEqual(above["mAP@0.5"], 1.0)

    def test_imperfect_localization_lowers_high_iou_ap(self):
        result = evaluate_detection(
            [[gt(0, [0.0, 0.0, 1.0, 1.0])]],
            [[pred(0, 0.9, [0.0, 0.0, 1.0, 0.6])]],
            CLASSES,
        )
        self.assertEqual(result["ap_by_iou"]["0.50"]["hello"], 1.0)
        self.assertEqual(result["ap_by_iou"]["0.75"]["hello"], 0.0)
        self.assertLess(result["mAP@0.5:0.95"], result["mAP@0.5"])


class DuplicateAndFalsePositiveTests(unittest.TestCase):
    def test_duplicate_prediction_is_a_false_positive(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)]],
            [[pred(0, 0.9, BOX_A), pred(0, 0.8, BOX_A)]],
            CLASSES,
        )
        hello = result["per_class"]["hello"]
        self.assertEqual(hello["true_positives"], 1)
        self.assertEqual(hello["false_positives"], 1)
        self.assertEqual(hello["precision"], 0.5)
        # COCO-style AP uses the precision envelope at each recall level. Both
        # predictions occur after full recall is reached, so the duplicate does
        # not reduce AP; it is still a false positive in the P/R counts above.
        self.assertEqual(result["mAP@0.5"], 1.0)

    def test_duplicate_before_recall_completion_lowers_ap(self):
        # Two GT boxes, three predictions: a duplicate of the first box is
        # scored above the true second box.
        result = evaluate_detection(
            [[gt(0, BOX_A), gt(0, BOX_B)]],
            [[
                pred(0, 0.9, BOX_A),
                pred(0, 0.8, BOX_A),
                pred(0, 0.7, BOX_B),
            ]],
            CLASSES,
        )
        self.assertLess(result["mAP@0.5"], 1.0)
        self.assertEqual(result["per_class"]["hello"]["false_positives"], 1)

    def test_low_confidence_false_positive_is_excluded_from_thresholded_pr(self):
        result = evaluate_detection(
            [[gt(0, BOX_A)]],
            [[pred(0, 0.9, BOX_A), pred(1, 0.2, BOX_B)]],
            CLASSES,
            confidence_threshold=0.5,
        )
        self.assertEqual(result["per_class"]["hello"]["false_positives"], 0)
        self.assertEqual(result["per_class"]["thankyou"]["predictions"], 0)
        self.assertEqual(
            result["predictions_at_confidence"], 1
        )

    def test_empty_predictions_are_zero_ap_not_undefined(self):
        result = evaluate_detection([[gt(0, BOX_A)]], [[]], CLASSES)
        self.assertEqual(result["mAP@0.5"], 0.0)
        self.assertEqual(result["per_class"]["hello"]["ap50"], 0.0)
        self.assertIsNone(result["micro"]["precision"])
        self.assertEqual(result["micro"]["recall"], 0.0)
        self.assertEqual(result["micro"]["false_negatives"], 1)

    def test_prediction_without_ground_truth_is_false_positive(self):
        result = evaluate_detection([[]], [[pred(0, 0.9, BOX_A)]], CLASSES)
        self.assertIsNone(result["mAP@0.5"])
        self.assertEqual(result["micro"]["false_positives"], 1)
        self.assertEqual(result["micro"]["precision"], 0.0)
        self.assertIsNone(result["micro"]["recall"])

    def test_empty_ground_truth_and_empty_predictions_are_undefined(self):
        result = evaluate_detection([[], []], [[], []], CLASSES)
        self.assertIsNone(result["mAP@0.5"])
        self.assertIsNone(result["mAP@0.5:0.95"])
        self.assertEqual(result["undefined_classes"], CLASSES)
        self.assertEqual(result["predictions_total"], 0)
        self.assertEqual(result["ground_truth_objects"], 0)


class PerClassTests(unittest.TestCase):
    def test_class_with_ground_truth_and_no_predictions_is_zero(self):
        result = evaluate_detection(
            [[gt(0, BOX_A), gt(1, BOX_B)]],
            [[pred(0, 0.9, BOX_A)]],
            CLASSES,
        )
        self.assertEqual(result["per_class"]["hello"]["ap50"], 1.0)
        self.assertEqual(result["per_class"]["thankyou"]["ap50"], 0.0)
        self.assertIsNone(result["per_class"]["thankyou"]["precision"])
        self.assertEqual(result["per_class"]["thankyou"]["recall"], 0.0)

    def test_aggregation_is_over_dataset_not_per_batch(self):
        # Two images: a class-0 box and a class-1 box. Batch averaging would
        # give 0.5 for each class if computed per image; dataset aggregation
        # must give 1.0 for both.
        result = evaluate_detection(
            [[gt(0, BOX_A)], [gt(1, BOX_B)]],
            [[pred(0, 0.9, BOX_A)], [pred(1, 0.9, BOX_B)]],
            CLASSES,
        )
        self.assertEqual(result["per_class"]["hello"]["ap50"], 1.0)
        self.assertEqual(result["per_class"]["thankyou"]["ap50"], 1.0)

    def test_average_precision_empty_ground_truth_is_none(self):
        self.assertIsNone(average_precision([], 0))
        self.assertEqual(average_precision([], 3), 0.0)


class InputValidationTests(unittest.TestCase):
    def test_invalid_box_is_rejected(self):
        with self.assertRaises(DetectionMetricsError):
            evaluate_detection(
                [[gt(0, [0.5, 0.5, 0.2, 0.2])]], [[]], CLASSES
            )

    def test_out_of_bounds_box_is_rejected(self):
        with self.assertRaises(DetectionMetricsError):
            evaluate_detection([[gt(0, [0.0, 0.0, 1.5, 1.0])]], [[]], CLASSES)

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(DetectionMetricsError):
            evaluate_detection([[gt(0, BOX_A)]], [[pred(0, 1.5, BOX_A)]], CLASSES)

    def test_unknown_class_is_rejected(self):
        with self.assertRaises(DetectionMetricsError):
            evaluate_detection([[gt(9, BOX_A)]], [[]], CLASSES)

    def test_image_count_mismatch_is_rejected(self):
        with self.assertRaises(DetectionMetricsError):
            evaluate_detection([[], []], [[]], CLASSES)


if __name__ == "__main__":
    unittest.main()
