from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from simulator.lstm_utils import (
    DEFAULT_LSTM_MAX_BATCH_SIZE,
    resolve_lstm_max_batch_size,
)


class LSTMUtilsTests(unittest.TestCase):
    def test_default_lstm_max_batch_size_is_65536_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(DEFAULT_LSTM_MAX_BATCH_SIZE, 65536)
            self.assertEqual(
                resolve_lstm_max_batch_size(None), DEFAULT_LSTM_MAX_BATCH_SIZE
            )

    def test_environment_override_still_applies(self) -> None:
        with patch.dict(os.environ, {"SRS_LSTM_MAX_BATCH": "32768"}):
            self.assertEqual(resolve_lstm_max_batch_size(None), 32768)

    def test_explicit_value_overrides_environment_default(self) -> None:
        with patch.dict(os.environ, {"SRS_LSTM_MAX_BATCH": "32768"}):
            self.assertEqual(resolve_lstm_max_batch_size(1024), 1024)

    def test_off_disables_chunking(self) -> None:
        with patch.dict(os.environ, {"SRS_LSTM_MAX_BATCH": "off"}):
            self.assertIsNone(resolve_lstm_max_batch_size(None))


if __name__ == "__main__":
    unittest.main()
