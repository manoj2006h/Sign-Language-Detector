import unittest

from tests.helpers import ROOT, SRC  # noqa: F401

from train import check_resume_split, split_train_validation_indices
from utils.checkpoint import CheckpointError


class ValidationSplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_deterministic(self):
        train_a, val_a = split_train_validation_indices(10, 0.2, seed=42)
        train_b, val_b = split_train_validation_indices(10, 0.2, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 2)
        self.assertEqual(len(train_a), 8)
        self.assertEqual(set(train_a).intersection(val_a), set())
        self.assertEqual(sorted(train_a + val_a), list(range(10)))

    def test_zero_fraction_keeps_all_samples_for_training(self):
        train, val = split_train_validation_indices(10, 0.0, seed=1)
        self.assertEqual(train, list(range(10)))
        self.assertEqual(val, [])

    def test_invalid_fractions_are_rejected(self):
        for fraction in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                split_train_validation_indices(10, fraction, seed=1)

    def test_single_sample_cannot_be_split(self):
        with self.assertRaises(ValueError):
            split_train_validation_indices(1, 0.1, seed=1)
        train, val = split_train_validation_indices(1, 0.0, seed=1)
        self.assertEqual(train, [0])
        self.assertEqual(val, [])

    def test_empty_dataset_is_allowed(self):
        train, val = split_train_validation_indices(0, 0.1, seed=1)
        self.assertEqual(train, [])
        self.assertEqual(val, [])

    def test_resume_accepts_matching_split(self):
        metadata = {"train_indices": [0, 2], "val_indices": [1]}
        self.assertTrue(check_resume_split(metadata, [0, 2], [1]))

    def test_resume_rejects_a_changed_split(self):
        metadata = {"train_indices": [0, 2], "val_indices": [1]}
        with self.assertRaises(CheckpointError):
            check_resume_split(metadata, [0, 1], [2])

    def test_old_checkpoint_without_split_metadata_is_allowed(self):
        self.assertFalse(check_resume_split({}, [0], [1]))


if __name__ == "__main__":
    unittest.main()
