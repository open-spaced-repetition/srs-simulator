from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simulator.anki_sm2_ap_policy import (
    ANKI_SM2_AP_DEFAULT_PARAMS,
    ANKI_SM2_AP_PARAM_BOUNDS,
    AnkiSM2APPolicy,
    clip_anki_sm2_ap_params,
    decode_parameter_delta,
)


class AnkiSM2APPolicyTests(unittest.TestCase):
    def test_bounds_cap_benchmark_interval_upper_bounds_at_100(self) -> None:
        high = [10_000.0] * 7

        clipped = clip_anki_sm2_ap_params(high)

        self.assertEqual(clipped[0], 100.0)
        self.assertEqual(clipped[1], 100.0)
        self.assertEqual(ANKI_SM2_AP_PARAM_BOUNDS[0], (1.0, 100.0))
        self.assertEqual(ANKI_SM2_AP_PARAM_BOUNDS[1], (1.0, 100.0))

    def test_decodes_zero_vector_to_default_params(self) -> None:
        decoded = decode_parameter_delta(
            ANKI_SM2_AP_DEFAULT_PARAMS,
            [0.0] * 7,
        )

        self.assertEqual(decoded, ANKI_SM2_AP_DEFAULT_PARAMS)

    def test_policy_round_trips_json(self) -> None:
        policy = AnkiSM2APPolicy.from_search_vector(search_vector=[0.0] * 7)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            policy.write_json(path)
            loaded = AnkiSM2APPolicy.from_json(path)

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.params, ANKI_SM2_AP_DEFAULT_PARAMS)
        self.assertEqual(loaded.params_dict()["ease_start"], 2.5)
        self.assertEqual(payload["policy_kind"], "anki-sm2-ap")


if __name__ == "__main__":
    unittest.main()
