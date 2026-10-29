import unittest
from unittest import mock

import torch

from tests.helpers import ROOT, SRC  # noqa: F401

from utils.device import resolve_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_is_explicit(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(self):
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_unavailable_cuda_is_a_clear_error(self):
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_device("cuda")
        self.assertIn("not available", str(ctx.exception).lower())

    def test_invalid_device_name_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    @mock.patch("utils.device.torch.cuda.device_count", return_value=1)
    @mock.patch("utils.device.torch.cuda.is_available", return_value=True)
    def test_cuda_index_is_checked(self, _available, _count):
        self.assertEqual(resolve_device("cuda:0"), torch.device("cuda:0"))
        with self.assertRaises(RuntimeError) as ctx:
            resolve_device("cuda:1")
        self.assertIn("out of range", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
