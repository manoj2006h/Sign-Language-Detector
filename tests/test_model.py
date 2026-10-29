import unittest

import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from model import DETR
from utils.inference import postprocess_detections
from utils.setup import get_classes


class ModelConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DETR(num_classes=2, backbone_weights=None, verbose=False)

    def test_forward_shapes_for_configured_classes_and_queries(self):
        model = self.model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(output["pred_logits"].shape), (2, 25, 3))
        self.assertEqual(tuple(output["pred_boxes"].shape), (2, 25, 4))

    def test_boxes_are_normalized(self):
        model = self.model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertGreaterEqual(output["pred_boxes"].min().item(), 0.0)
        self.assertLessEqual(output["pred_boxes"].max().item(), 1.0)

    def test_architecture_config_is_documented(self):
        config = self.model.architecture_config()
        self.assertEqual(config["hidden_dim"], 256)
        self.assertEqual(config["num_queries"], 25)
        self.assertEqual(config["num_encoder_layers"], 1)
        self.assertEqual(config["num_decoder_layers"], 1)

    def test_missing_pretrained_checkpoint_raises_instead_of_silently_continuing(self):
        model = DETR(num_classes=2, backbone_weights=None, verbose=False)
        with self.assertRaises(Exception):
            model.load_pretrained("/definitely/missing.pt", num_classes=2)

    def test_configured_classes_drive_model_and_inference(self):
        classes = get_classes()
        model = DETR(
            num_classes=len(classes), backbone_weights=None, verbose=False
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(output["pred_logits"].shape[-1], len(classes) + 1)
        detections = postprocess_detections(
            output, [(10, 10)], classes, threshold=1.0
        )
        self.assertEqual(len(detections), 1)
        self.assertTrue(all(detection["class_name"] in classes for detection in detections[0]))


if __name__ == "__main__":
    unittest.main()
