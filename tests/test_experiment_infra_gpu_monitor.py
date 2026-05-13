# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra.gpu_monitor import (
    GpuMonitor,
    parse_nvidia_smi_csv,
    parse_powershell_shared_memory_output,
)


class GpuMonitorParserTests(unittest.TestCase):
    def test_parses_powershell_csv_shared_usage(self) -> None:
        text = "\n".join(
            [
                '"Path","CookedValue"',
                '"\\\\host\\gpu adapter memory(luid_0x00000000_phys_0)\\shared usage","1048576"',
                '"\\\\host\\gpu adapter memory(luid_0x00000001_phys_0)\\shared usage","2097152"',
            ]
        )

        self.assertEqual(
            parse_powershell_shared_memory_output(text),
            [1048576, 2097152],
        )

    def test_parses_powershell_format_list_shared_usage(self) -> None:
        text = """
Path        : \\\\host\\gpu adapter memory(luid_0x00000000_phys_0)\\shared usage
CookedValue : 3145728

Path        : \\\\host\\gpu adapter memory(luid_0x00000001_phys_0)\\shared usage
CookedValue : 4194304
""".strip()

        self.assertEqual(
            parse_powershell_shared_memory_output(text),
            [3145728, 4194304],
        )

    def test_parses_nvidia_smi_csv(self) -> None:
        rows = parse_nvidia_smi_csv(
            "2026/05/13 01:02:03.000, 0, NVIDIA RTX, 1234, 87\n"
        )

        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["memory_used_mib"], 1234.0)
        self.assertEqual(rows[0]["utilization_gpu_percent"], 87.0)

    def test_summary_notes_missing_shared_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = GpuMonitor(output_dir=Path(tmp), interval_seconds=1.0)
            monitor._samples.append(  # noqa: SLF001 - direct parser-focused fixture
                {
                    "timestamp": "2026-05-13T00:00:00+00:00",
                    "shared_memory_bytes_by_adapter": [],
                    "nvidia_smi": [
                        {
                            "memory_used_mib": 512.0,
                            "utilization_gpu_percent": 50.0,
                        }
                    ],
                    "notes": [],
                }
            )

            summary = monitor.stop().to_dict()

        self.assertIsNone(summary["shared_memory_spill_detected"])
        self.assertEqual(summary["nvidia_smi_peak_memory_used_mib"], 512.0)
        self.assertIn("shared-memory spill detection", " ".join(summary["notes"]))


if __name__ == "__main__":
    unittest.main()
