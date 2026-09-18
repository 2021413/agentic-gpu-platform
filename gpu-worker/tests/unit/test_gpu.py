"""GPU discovery: parsing what the driver says, and degrading when it says nothing."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from worker.gpu import HOPPER, GpuInfo, GpuSurvey, survey_gpus

ONE_H100 = "0, NVIDIA H100 80GB HBM3, 81559, 9.0, 550.54.15\n"
TWO_L40S = "0, NVIDIA L40S, 46068, 8.9, 535.183.06\n1, NVIDIA L40S, 46068, 8.9, 535.183.06\n"


class FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _driver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    present: bool = True,
    completed: FakeCompleted | None = None,
    raises: BaseException | None = None,
) -> list[list[str]]:
    """Stand in for ``nvidia-smi``; returns the argument vectors it was given."""
    seen: list[list[str]] = []

    monkeypatch.setattr(
        "worker.gpu.shutil.which", lambda _name: "/usr/bin/nvidia-smi" if present else None
    )

    def run(argv: list[str], **_kwargs: Any) -> FakeCompleted:
        seen.append(argv)
        if raises is not None:
            raise raises
        return completed if completed is not None else FakeCompleted()

    monkeypatch.setattr("worker.gpu.subprocess.run", run)
    return seen


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------


def test_a_single_gpu_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _driver(monkeypatch, completed=FakeCompleted(stdout=ONE_H100))

    survey = survey_gpus()

    assert survey.error is None
    assert survey.count == 1
    gpu = survey.gpus[0]
    assert gpu.index == 0
    assert gpu.name == "NVIDIA H100 80GB HBM3"
    assert gpu.memory_total_mb == 81559
    assert gpu.compute_capability == (9, 0)
    assert gpu.driver_version == "550.54.15"
    assert gpu.memory_total_gb == pytest.approx(79.6, abs=0.1)
    assert survey.driver_version == "550.54.15"
    assert survey.homogeneous is True
    assert survey.all_support_fp8 is True
    assert "compute 9.0" in gpu.render()

    # The query is an argument vector, never a shell string.
    assert calls[0][0] == "/usr/bin/nvidia-smi"
    assert "--format=csv,noheader,nounits" in calls[0]


def test_two_gpus_are_parsed_and_summed(monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(monkeypatch, completed=FakeCompleted(stdout=TWO_L40S))

    survey = survey_gpus()

    assert survey.count == 2
    assert [gpu.index for gpu in survey.gpus] == [0, 1]
    assert survey.total_memory_gb == pytest.approx(2 * 46068 / 1024)
    assert survey.homogeneous is True
    assert survey.all_support_fp8 is True
    assert len(survey.render()) == 2


def test_a_malformed_line_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(
        monkeypatch,
        completed=FakeCompleted(
            stdout=(
                "\n"
                "this line has too few fields\n"
                "0, NVIDIA H100 80GB HBM3, [N/A], 9.0, 550.54.15\n"
                "not-an-index, NVIDIA H100 80GB HBM3, 81559, 9.0, 550.54.15\n"
                "1, NVIDIA H100 80GB HBM3, 81559, ninepointzero, 550.54.15\n" + ONE_H100
            )
        ),
    )

    survey = survey_gpus()

    assert survey.count == 1
    assert survey.gpus[0].name == "NVIDIA H100 80GB HBM3"


def test_a_minor_capability_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(
        monkeypatch,
        completed=FakeCompleted(stdout="0, NVIDIA A100-SXM4-80GB, 81920, 8.0, 535.104.05\n"),
    )

    survey = survey_gpus()

    assert survey.gpus[0].compute_capability == (8, 0)
    assert survey.all_support_fp8 is False


def test_a_capability_without_a_minor_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(monkeypatch, completed=FakeCompleted(stdout="0, NVIDIA H100, 81559, 9, 550.54.15\n"))

    assert survey_gpus().gpus[0].compute_capability == (9, 0)


# ----------------------------------------------------------------------
# degrading, never raising
# ----------------------------------------------------------------------


def test_nvidia_smi_absent_gives_an_empty_survey_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _driver(monkeypatch, present=False)

    survey = survey_gpus()

    assert survey.gpus == ()
    assert survey.count == 0
    assert survey.error == "nvidia-smi is not on PATH"
    assert survey.driver_version == "unknown"
    assert survey.total_memory_gb == 0
    assert survey.all_support_fp8 is False
    assert survey.render() == ["GPUs: none visible (nvidia-smi is not on PATH)"]


def test_a_non_zero_exit_code_gives_an_empty_survey_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _driver(
        monkeypatch,
        completed=FakeCompleted(
            returncode=9,
            stderr="Failed to initialize NVML: Driver/library version mismatch\n",
        ),
    )

    survey = survey_gpus()

    assert survey.gpus == ()
    assert survey.error == "Failed to initialize NVML: Driver/library version mismatch"


def test_a_non_zero_exit_code_without_output_still_explains_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _driver(monkeypatch, completed=FakeCompleted(returncode=2))

    assert survey_gpus().error == "exit code 2"


def test_an_empty_listing_is_reported_as_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    _driver(monkeypatch, completed=FakeCompleted(stdout="\n\n"))

    assert survey_gpus().error == "nvidia-smi listed no devices"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10.0), id="timeout"),
        pytest.param(OSError("Exec format error"), id="oserror"),
        pytest.param(subprocess.SubprocessError("boom"), id="subprocess"),
    ],
)
def test_a_driver_that_hangs_or_explodes_never_raises(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    _driver(monkeypatch, raises=failure)

    survey = survey_gpus()

    assert survey.gpus == ()
    assert survey.error is not None
    assert type(failure).__name__ in survey.error


# ----------------------------------------------------------------------
# capability predicates
# ----------------------------------------------------------------------


def _gpu(capability: tuple[int, int], name: str = "NVIDIA H100 80GB HBM3") -> GpuInfo:
    return GpuInfo(
        index=0,
        name=name,
        memory_total_mb=81559,
        compute_capability=capability,
        driver_version="550.54.15",
    )


@pytest.mark.parametrize(
    ("capability", "supported"),
    [
        ((9, 0), True),  # Hopper
        ((8, 9), True),  # Ada
        ((9, 1), True),
        ((10, 0), True),
        ((8, 6), False),  # Ampere consumer
        ((8, 0), False),  # A100
        ((7, 5), False),  # Turing
    ],
)
def test_hardware_fp8_starts_at_compute_8_9(capability: tuple[int, int], supported: bool) -> None:
    assert _gpu(capability).supports_fp8 is supported


@pytest.mark.parametrize(
    ("capability", "hopper"),
    [((9, 0), True), ((10, 0), True), ((8, 9), False), ((8, 0), False)],
)
def test_hopper_or_newer(capability: tuple[int, int], hopper: bool) -> None:
    assert _gpu(capability).is_hopper_or_newer is hopper
    assert HOPPER == (9, 0)


def test_all_support_fp8_needs_every_card(monkeypatch: pytest.MonkeyPatch) -> None:
    survey = GpuSurvey((_gpu((9, 0)), _gpu((8, 0), name="NVIDIA A100-SXM4-80GB")))

    assert survey.all_support_fp8 is False


def test_mismatched_cards_are_not_homogeneous() -> None:
    survey = GpuSurvey((_gpu((9, 0)), _gpu((8, 9), name="NVIDIA L40S")))

    assert survey.homogeneous is False
    assert survey.count == 2


def test_identical_cards_are_homogeneous() -> None:
    assert GpuSurvey((_gpu((9, 0)), _gpu((9, 0)))).homogeneous is True
