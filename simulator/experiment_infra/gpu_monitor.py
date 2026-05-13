from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SHARED_GPU_SPILL_THRESHOLD_BYTES = 1024 * 1024 * 1024
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(slots=True)
class GpuMonitorSummary:
    enabled: bool
    output_dir: Path
    jsonl_path: Path
    summary_path: Path
    started_at: str | None = None
    finished_at: str | None = None
    sample_count: int = 0
    interval_seconds: float = 2.0
    shared_memory_peak_single_adapter_bytes: int | None = None
    shared_memory_peak_summed_bytes: int | None = None
    shared_memory_spill_threshold_bytes: int = SHARED_GPU_SPILL_THRESHOLD_BYTES
    shared_memory_spill_detected: bool | None = None
    nvidia_smi_peak_memory_used_mib: float | None = None
    nvidia_smi_peak_utilization_percent: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "output_dir": str(self.output_dir),
            "jsonl_path": str(self.jsonl_path),
            "summary_path": str(self.summary_path),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sample_count": self.sample_count,
            "interval_seconds": self.interval_seconds,
            "shared_memory_peak_single_adapter_bytes": (
                self.shared_memory_peak_single_adapter_bytes
            ),
            "shared_memory_peak_summed_bytes": self.shared_memory_peak_summed_bytes,
            "shared_memory_spill_threshold_bytes": (
                self.shared_memory_spill_threshold_bytes
            ),
            "shared_memory_spill_detected": self.shared_memory_spill_detected,
            "nvidia_smi_peak_memory_used_mib": self.nvidia_smi_peak_memory_used_mib,
            "nvidia_smi_peak_utilization_percent": (
                self.nvidia_smi_peak_utilization_percent
            ),
            "notes": list(dict.fromkeys(self.notes)),
        }


class GpuMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        interval_seconds: float = 2.0,
        spill_threshold_bytes: int = SHARED_GPU_SPILL_THRESHOLD_BYTES,
    ) -> None:
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self.spill_threshold_bytes = spill_threshold_bytes
        self.jsonl_path = output_dir / "gpu_memory.jsonl"
        self.summary_path = output_dir / "summary.json"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._started_at: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")
        self._started_at = _timestamp()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> GpuMonitorSummary:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))
        summary = self._build_summary(finished_at=_timestamp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = sample_gpu_memory()
            self._samples.append(sample)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
            self._stop_event.wait(self.interval_seconds)

    def _build_summary(self, *, finished_at: str) -> GpuMonitorSummary:
        notes: list[str] = []
        shared_single_peaks: list[int] = []
        shared_summed_peaks: list[int] = []
        nvidia_memory_values: list[float] = []
        nvidia_util_values: list[float] = []
        saw_shared_counter = False

        for sample in self._samples:
            notes.extend(str(note) for note in sample.get("notes", []))
            shared_values = sample.get("shared_memory_bytes_by_adapter")
            if isinstance(shared_values, list) and shared_values:
                saw_shared_counter = True
                int_values = [
                    int(value)
                    for value in shared_values
                    if isinstance(value, int) and value >= 0
                ]
                if int_values:
                    shared_single_peaks.append(max(int_values))
                    shared_summed_peaks.append(sum(int_values))
            for gpu in sample.get("nvidia_smi", []):
                if not isinstance(gpu, dict):
                    continue
                memory = gpu.get("memory_used_mib")
                utilization = gpu.get("utilization_gpu_percent")
                if isinstance(memory, (float, int)):
                    nvidia_memory_values.append(float(memory))
                if isinstance(utilization, (float, int)):
                    nvidia_util_values.append(float(utilization))

        peak_single = max(shared_single_peaks) if shared_single_peaks else None
        peak_summed = max(shared_summed_peaks) if shared_summed_peaks else None
        if not saw_shared_counter:
            notes.append(
                "Windows shared GPU memory counter unavailable; shared-memory "
                "spill detection is unavailable."
            )
        spill_detected = (
            peak_single is not None and peak_single >= self.spill_threshold_bytes
        )
        return GpuMonitorSummary(
            enabled=True,
            output_dir=self.output_dir,
            jsonl_path=self.jsonl_path,
            summary_path=self.summary_path,
            started_at=self._started_at,
            finished_at=finished_at,
            sample_count=len(self._samples),
            interval_seconds=self.interval_seconds,
            shared_memory_peak_single_adapter_bytes=peak_single,
            shared_memory_peak_summed_bytes=peak_summed,
            shared_memory_spill_threshold_bytes=self.spill_threshold_bytes,
            shared_memory_spill_detected=spill_detected if saw_shared_counter else None,
            nvidia_smi_peak_memory_used_mib=max(nvidia_memory_values)
            if nvidia_memory_values
            else None,
            nvidia_smi_peak_utilization_percent=max(nvidia_util_values)
            if nvidia_util_values
            else None,
            notes=notes,
        )


def disabled_monitor_summary(output_dir: Path) -> GpuMonitorSummary:
    return GpuMonitorSummary(
        enabled=False,
        output_dir=output_dir,
        jsonl_path=output_dir / "gpu_memory.jsonl",
        summary_path=output_dir / "summary.json",
        shared_memory_spill_detected=None,
    )


def sample_gpu_memory() -> dict[str, Any]:
    notes: list[str] = []
    shared_values = _sample_windows_shared_memory(notes)
    nvidia_smi = _sample_nvidia_smi(notes)
    return {
        "timestamp": _timestamp(),
        "shared_memory_bytes_by_adapter": shared_values,
        "shared_memory_single_adapter_bytes": max(shared_values)
        if shared_values
        else None,
        "shared_memory_summed_bytes": sum(shared_values) if shared_values else None,
        "nvidia_smi": nvidia_smi,
        "notes": notes,
    }


def parse_powershell_shared_memory_output(text: str) -> list[int]:
    csv_values = _parse_powershell_csv_shared_memory(text)
    if csv_values:
        return csv_values

    values: list[int] = []
    expect_numeric = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if "cookedvalue" in lower:
            parsed = _last_number(line)
            if parsed is not None:
                values.append(_counter_value_to_bytes(parsed))
            expect_numeric = False
            continue
        if "shared usage" in lower:
            shared_index = lower.find("shared usage")
            suffix = line[shared_index + len("shared usage") :]
            parsed = _last_number(suffix)
            if parsed is not None:
                values.append(_counter_value_to_bytes(parsed))
                expect_numeric = False
            else:
                expect_numeric = True
            continue
        if expect_numeric:
            parsed = _last_number(line)
            if parsed is not None:
                values.append(_counter_value_to_bytes(parsed))
                expect_numeric = False
    return values


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "index": _parse_int(parts[1]),
                "name": parts[2],
                "memory_used_mib": _parse_float(parts[3]),
                "utilization_gpu_percent": _parse_float(parts[4]),
            }
        )
    return rows


def _sample_windows_shared_memory(notes: list[str]) -> list[int]:
    powershell = _powershell_executable()
    if powershell is None:
        return []
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' "
                "| Select-Object -ExpandProperty CounterSamples "
                "| Select-Object Path,CookedValue "
                "| ConvertTo-Csv -NoTypeInformation",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"PowerShell shared-memory counter failed: {exc}")
        return []
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        notes.append(
            "PowerShell shared-memory counter returned non-zero"
            + (f": {stderr}" if stderr else ".")
        )
        return []
    values = parse_powershell_shared_memory_output(completed.stdout)
    if not values:
        notes.append("PowerShell shared-memory counter returned no samples.")
    return values


def _sample_nvidia_smi(notes: list[str]) -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        notes.append("nvidia-smi unavailable.")
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=timestamp,index,name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"nvidia-smi sampling failed: {exc}")
        return []
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        notes.append(
            "nvidia-smi returned non-zero" + (f": {stderr}" if stderr else ".")
        )
        return []
    return parse_nvidia_smi_csv(completed.stdout)


def _parse_powershell_csv_shared_memory(text: str) -> list[int]:
    try:
        reader = csv.DictReader(text.splitlines())
    except csv.Error:
        return []
    if not reader.fieldnames or "CookedValue" not in reader.fieldnames:
        return []
    values: list[int] = []
    for row in reader:
        value = row.get("CookedValue")
        parsed = _parse_float(value or "")
        if parsed is not None:
            values.append(_counter_value_to_bytes(parsed))
    return values


def _powershell_executable() -> str | None:
    if platform.system().lower() == "windows":
        return shutil.which("powershell.exe") or shutil.which("powershell")
    wsl_path = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if wsl_path.exists():
        return str(wsl_path)
    return None


def _counter_value_to_bytes(value: float) -> int:
    return max(0, int(round(value)))


def _last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_float(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_int(text: str) -> int | None:
    value = _parse_float(text)
    if value is None:
        return None
    return int(value)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
