from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulator.cuda_allocator import enable_expandable_cuda_segments


class CudaAllocatorTests(unittest.TestCase):
    def test_enable_calls_private_allocator_hook(self) -> None:
        hook = Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                memory=SimpleNamespace(_set_allocator_settings=hook),
            ),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            enabled = enable_expandable_cuda_segments()

        self.assertTrue(enabled)
        hook.assert_called_once_with("expandable_segments:True")

    def test_enable_returns_false_when_hook_is_missing(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(memory=SimpleNamespace()),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            enabled = enable_expandable_cuda_segments()

        self.assertFalse(enabled)

    def test_enable_returns_false_when_hook_rejects_setting(self) -> None:
        hook = Mock(side_effect=RuntimeError("allocator already initialized"))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                memory=SimpleNamespace(_set_allocator_settings=hook),
            ),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            enabled = enable_expandable_cuda_segments()

        self.assertFalse(enabled)
        hook.assert_called_once_with("expandable_segments:True")


if __name__ == "__main__":
    unittest.main()
