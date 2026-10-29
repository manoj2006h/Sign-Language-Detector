import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.checkpoint import (
    CheckpointError,
    extract_state_dict,
    load_checkpoint,
    load_model_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_compatibility,
)


class TinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.linear_class = nn.Linear(4, num_classes + 1)
        self.query_pos = nn.Parameter(torch.zeros(3, 4))

    def architecture_config(self):
        return {"hidden_dim": 4, "num_queries": 3}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _wrapped_checkpoint(
        self,
        model,
        path,
        optimizer=None,
        classes=None,
        train_indices=None,
        val_indices=None,
    ):
        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return save_training_checkpoint(
            path,
            model,
            optimizer,
            None,
            epoch=3,
            best_val_loss=1.25,
            classes=classes or ["a", "b"],
            model_config=model.architecture_config(),
            val_loss=1.5,
            history=[{"epoch": 1, "val_loss": 2.0}],
            train_indices=train_indices,
            val_indices=val_indices,
        )

    def test_save_and_load_roundtrip_preserves_metadata_and_optimizer(self):
        model = TinyModel(num_classes=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.123)
        path = self.root / "nested" / "best.pt"
        self._wrapped_checkpoint(
            model,
            path,
            optimizer=optimizer,
            train_indices=[0, 1],
            val_indices=[2],
        )

        restored = TinyModel(num_classes=2)
        restored_opt = torch.optim.SGD(restored.parameters(), lr=0.5)
        meta = load_model_checkpoint(
            restored,
            path,
            num_classes=2,
            classes=["a", "b"],
            expected_model_config=restored.architecture_config(),
            restore_training=True,
            optimizer=restored_opt,
            scheduler=None,
        )
        self.assertEqual(meta["epoch"], 3)
        self.assertAlmostEqual(meta["best_val_loss"], 1.25)
        self.assertEqual(meta["train_indices"], [0, 1])
        self.assertEqual(meta["val_indices"], [2])
        self.assertAlmostEqual(restored_opt.param_groups[0]["lr"], 0.123)
        for a, b in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_raw_state_dict_is_supported(self):
        model = TinyModel(num_classes=2)
        path = self.root / "raw.pt"
        torch.save(model.state_dict(), path)
        restored = TinyModel(num_classes=2)
        meta = load_model_checkpoint(
            restored, path, num_classes=2, classes=["a", "b"]
        )
        self.assertEqual(meta, {})

    def test_missing_checkpoint_has_actionable_error(self):
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(self.root / "missing.pt")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_lfs_pointer_is_rejected_before_torch_load(self):
        path = self.root / "pointer.pt"
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:abc\nsize 108081916\n"
        )
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(path)
        message = str(ctx.exception).lower()
        self.assertIn("git lfs", message)
        self.assertIn("pointer", message)

    def test_corrupt_file_is_rejected(self):
        path = self.root / "corrupt.pt"
        path.write_bytes(b"this is not a torch checkpoint")
        with self.assertRaises(CheckpointError):
            load_checkpoint(path)

    def test_class_head_mismatch_is_rejected(self):
        model = TinyModel(num_classes=2)
        state = model.state_dict()
        with self.assertRaises(CheckpointError) as ctx:
            validate_checkpoint_compatibility(state, {}, num_classes=3)
        self.assertIn("class", str(ctx.exception).lower())

    def test_malformed_class_head_shape_is_rejected(self):
        state = {"linear_class.weight": torch.tensor(1.0)}
        with self.assertRaises(CheckpointError) as ctx:
            validate_checkpoint_compatibility(state, {}, num_classes=2)
        self.assertIn("shape", str(ctx.exception).lower())

    def test_metadata_class_names_must_match(self):
        model = TinyModel(num_classes=2)
        state = model.state_dict()
        with self.assertRaises(CheckpointError):
            validate_checkpoint_compatibility(
                state, {"classes": ["a", "c"]}, num_classes=2, classes=["a", "b"]
            )

    def test_metadata_model_config_must_match(self):
        model = TinyModel(num_classes=2)
        state = model.state_dict()
        with self.assertRaises(CheckpointError):
            validate_checkpoint_compatibility(
                state,
                {"model_config": {"hidden_dim": 99}},
                num_classes=2,
                expected_model_config={"hidden_dim": 4},
            )

    def test_malformed_metadata_is_a_checkpoint_error(self):
        state = TinyModel(num_classes=2).state_dict()
        with self.assertRaises(CheckpointError):
            validate_checkpoint_compatibility(
                state, {"num_classes": "not-a-number"}, num_classes=2
            )
        with self.assertRaises(CheckpointError):
            validate_checkpoint_compatibility(
                state, {"classes": 5}, num_classes=2, classes=["a", "b"]
            )

    def test_strict_architecture_load_failure_is_rejected(self):
        model = TinyModel(num_classes=2)
        path = self.root / "arch.pt"
        torch.save(model.state_dict(), path)
        other = nn.Sequential(nn.Linear(4, 4))
        with self.assertRaises(CheckpointError) as ctx:
            load_model_checkpoint(other, path, num_classes=2)
        self.assertIn("incompatible", str(ctx.exception).lower())

    def test_resume_requires_optimizer_state(self):
        model = TinyModel(num_classes=2)
        path = self.root / "raw.pt"
        torch.save(model.state_dict(), path)
        with self.assertRaises(CheckpointError):
            load_model_checkpoint(
                TinyModel(num_classes=2),
                path,
                num_classes=2,
                restore_training=True,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            )

    def test_save_is_atomic_and_leaves_no_tmp_file(self):
        model = TinyModel(num_classes=2)
        path = self.root / "best.pt"
        self._wrapped_checkpoint(model, path)
        self.assertTrue(path.is_file())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_save_preserves_existing_checkpoint_and_cleans_tmp(self):
        model = TinyModel(num_classes=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        path = self.root / "best.pt"
        path.write_bytes(b"original checkpoint")

        with mock.patch(
            "utils.checkpoint.torch.save", side_effect=RuntimeError("disk full")
        ):
            with self.assertRaises(RuntimeError):
                save_training_checkpoint(
                    path,
                    model,
                    optimizer,
                    None,
                    epoch=1,
                    best_val_loss=1.0,
                    classes=["a", "b"],
                    model_config=model.architecture_config(),
                )

        self.assertEqual(path.read_bytes(), b"original checkpoint")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_non_numeric_format_version_is_a_checkpoint_error(self):
        state = TinyModel(num_classes=2).state_dict()
        with self.assertRaises(CheckpointError):
            extract_state_dict(
                {"model_state_dict": state, "format_version": "not-an-int"}
            )


if __name__ == "__main__":
    unittest.main()
