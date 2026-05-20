from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from experiments.single_card_tradeoff.tradeoff import (
    _build_regret_auc_rows,
    _parse_user_ids_csv,
    _resolve_user_ids,
    _resolve_user_policy_template,
    _write_csv,
)


class SingleCardTradeoffMultiUserTests(unittest.TestCase):
    def test_parse_user_ids_csv(self) -> None:
        self.assertEqual(_parse_user_ids_csv("1,2,3"), [1, 2, 3])
        with self.assertRaises(SystemExit):
            _parse_user_ids_csv("")
        with self.assertRaises(SystemExit):
            _parse_user_ids_csv("1,1")
        with self.assertRaises(SystemExit):
            _parse_user_ids_csv("0")

    def test_resolve_user_ids_rejects_conflicting_user_id(self) -> None:
        args = argparse.Namespace(user_id=1, user_ids="1,2", engine="vectorized")
        with self.assertRaises(SystemExit):
            _resolve_user_ids(args)

    def test_resolve_user_ids_rejects_event_multiuser(self) -> None:
        args = argparse.Namespace(user_id=None, user_ids="1,2", engine="event")
        with self.assertRaises(SystemExit):
            _resolve_user_ids(args)

    def test_policy_template_substitutes_user_id(self) -> None:
        path = _resolve_user_policy_template(
            "artifacts/user_{user_id}_policy.pt",
            user_id=7,
        )
        self.assertEqual(path, Path("artifacts/user_7_policy.pt"))
        with self.assertRaises(SystemExit):
            _resolve_user_policy_template("artifacts/shared_policy.pt", user_id=7)

    def test_csv_fieldnames_include_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.csv"
            _write_csv(
                path,
                [
                    {
                        "user_id": 2,
                        "environment": "fsrs6_default",
                        "scheduler": "fsrs6",
                        "scheduler_spec": "fsrs6",
                    }
                ],
            )
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
        self.assertEqual(header[0], "user_id")

    def test_regret_auc_groups_by_user_id(self) -> None:
        rows = []
        for user_id, offset in ((1, 0.0), (2, 100.0)):
            for scheduler, minutes_delta in (("fsrs6", 0.0), ("fixed", 1.0)):
                for idx, memory in enumerate((1000.0 + offset, 2000.0 + offset)):
                    rows.append(
                        {
                            "user_id": user_id,
                            "environment": "fsrs6_default",
                            "review_markov_transition": False,
                            "scheduler": scheduler,
                            "scheduler_spec": scheduler,
                            "deck_expected_memorized": memory,
                            "deck_minutes_per_day": 10.0 + idx + minutes_delta,
                        }
                    )

        auc_rows = _build_regret_auc_rows(rows)
        fsrs_vs_fixed = [
            row
            for row in auc_rows
            if row["baseline_scheduler"] == "fsrs6" and row["scheduler"] == "fixed"
        ]

        self.assertEqual({row["user_id"] for row in fsrs_vs_fixed}, {1, 2})
        self.assertTrue(all(row["baseline_point_count"] == 2 for row in fsrs_vs_fixed))


if __name__ == "__main__":
    unittest.main()
