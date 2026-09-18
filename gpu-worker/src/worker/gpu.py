"""GPU discovery, by asking the driver rather than guessing.

Everything here degrades to "no GPU visible" instead of raising, because the
same code runs in CI on a laptop. Preflight decides what an empty list means;
this module only reports.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["HOPPER", "GpuInfo", "GpuSurvey", "survey_gpus"]

# Compute capability 9.0. The FP8 path this image relies on is a Hopper feature;
# on anything older vLLM silently falls back and the worker becomes far slower
# than its capacity suggests, which is worse than refusing.
HOPPER = (9, 0)

_QUERY = "index,name,memory.total,compute_cap,driver_version"


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    compute_capability: tuple[int, int]
    driver_version: str

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def supports_fp8(self) -> bool:
        """Hardware FP8 arrives with Hopper (9.0) and Ada (8.9)."""
        return self.compute_capability >= (8, 9)

    @property
    def is_hopper_or_newer(self) -> bool:
        return self.compute_capability >= HOPPER

    def render(self) -> str:
        major, minor = self.compute_capability
        return (
            f"GPU {self.index}: {self.name}, {self.memory_total_gb:,.0f} GB, "
            f"compute {major}.{minor}, driver {self.driver_version}"
        )


@dataclass(frozen=True, slots=True)
class GpuSurvey:
    """What the driver reported, plus why it might have reported nothing."""

    gpus: tuple[GpuInfo, ...] = ()
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.gpus)

    @property
    def driver_version(self) -> str:
        return self.gpus[0].driver_version if self.gpus else "unknown"

    @property
    def total_memory_gb(self) -> float:
        return sum(gpu.memory_total_gb for gpu in self.gpus)

    @property
    def all_support_fp8(self) -> bool:
        return bool(self.gpus) and all(gpu.supports_fp8 for gpu in self.gpus)

    @property
    def homogeneous(self) -> bool:
        """Tensor parallelism across mismatched cards is a support nightmare."""
        return len({gpu.name for gpu in self.gpus}) <= 1

    def render(self) -> list[str]:
        if not self.gpus:
            return [f"GPUs: none visible ({self.error or 'nvidia-smi reported nothing'})"]
        return [gpu.render() for gpu in self.gpus]


def survey_gpus(*, timeout: float = 10.0) -> GpuSurvey:
    """Ask ``nvidia-smi``. Never raises."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return GpuSurvey(error="nvidia-smi is not on PATH")
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuSurvey(error=f"{type(exc).__name__}: {exc}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return GpuSurvey(error=detail[0] if detail else f"exit code {completed.returncode}")

    gpus = tuple(
        parsed
        for line in completed.stdout.splitlines()
        if (parsed := _parse_line(line)) is not None
    )
    if not gpus:
        return GpuSurvey(error="nvidia-smi listed no devices")
    return GpuSurvey(gpus)


def _parse_line(line: str) -> GpuInfo | None:
    fields = [part.strip() for part in line.split(",")]
    expected = len(_QUERY.split(","))
    if len(fields) != expected:
        return None
    index, name, memory, capability, driver = fields
    try:
        major, _, minor = capability.partition(".")
        return GpuInfo(
            index=int(index),
            name=name,
            memory_total_mb=int(float(memory)),
            compute_capability=(int(major), int(minor or 0)),
            driver_version=driver,
        )
    except ValueError:
        return None
