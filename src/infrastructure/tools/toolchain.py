"""Toolchain profiles (spec section 7's C/C++ requirements).

The domain stores *what* to run for a project (``ToolchainConfig``) and refuses
to guess it. This module is the catalogue an operator picks from when filling
that configuration in: named, reviewable command sets for the toolchains the
spec calls out — gcc, clang, make, cmake, ninja, clang-tidy, cppcheck, and the
sanitizers.

Nothing here is applied automatically. A profile becomes a project's toolchain
only when somebody selects it, because a build command inferred from the shape
of a repository will eventually be wrong, and its failure would be reported to
the repair loop as if the code were broken.

Commands are single argument vectors: they are executed without a shell, so
``&&``, pipes and redirections are impossible by construction. Multi-step
toolchains express their configure step as ``install_command`` (for CMake, that
is the ``cmake -S . -B build`` call) and their compile step as ``build_command``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from domain.entities.project import ToolchainConfig

__all__ = ["PROFILES", "SANITIZERS", "ToolchainProfile", "profile", "with_sanitizer"]


@dataclass(frozen=True, slots=True)
class ToolchainProfile:
    """A reviewable set of commands for one way of building a project."""

    name: str
    language: str
    description: str
    build_command: str | None = None
    test_command: str | None = None
    static_analysis_command: str | None = None
    install_command: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def to_config(
        self,
        *,
        working_subdirectory: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> ToolchainConfig:
        """Materialise this profile as the project's stored configuration."""
        merged = {**self.environment, **(environment or {})}
        return ToolchainConfig(
            language=self.language,
            build_command=self.build_command,
            test_command=self.test_command,
            static_analysis_command=self.static_analysis_command,
            install_command=self.install_command,
            working_subdirectory=working_subdirectory,
            environment=merged,
        )


_SANITIZER_FLAGS: Mapping[str, str] = {
    "address": "-fsanitize=address -fno-omit-frame-pointer -g",
    "undefined": "-fsanitize=undefined -fno-omit-frame-pointer -g",
    "thread": "-fsanitize=thread -fno-omit-frame-pointer -g",
}

_SANITIZER_RUNTIME: Mapping[str, Mapping[str, str]] = {
    # ``abort_on_error`` matters: without it a sanitizer diagnostic can be
    # printed while the process still exits 0, and a tool layer that trusts exit
    # codes would report a clean run.
    "address": {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1:strict_string_checks=1"},
    "undefined": {"UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"},
    "thread": {"TSAN_OPTIONS": "halt_on_error=1"},
}

SANITIZERS: tuple[str, ...] = tuple(_SANITIZER_FLAGS)


PROFILES: Mapping[str, ToolchainProfile] = {
    profile_.name: profile_
    for profile_ in (
        ToolchainProfile(
            name="python-pytest",
            language="python",
            description="Byte-compile as a build check, pytest for tests, ruff for analysis.",
            build_command="python -m compileall -q .",
            test_command="pytest -q",
            static_analysis_command="ruff check .",
        ),
        ToolchainProfile(
            name="c-make",
            language="c",
            description="Plain Makefile project built with gcc.",
            build_command="make -j4",
            test_command="make test",
            static_analysis_command=(
                "cppcheck --enable=warning,performance,portability --error-exitcode=1 ."
            ),
            environment={"CC": "gcc", "CFLAGS": "-O2 -Wall -Wextra -Werror"},
        ),
        ToolchainProfile(
            name="c-cmake-ninja",
            language="c",
            description="CMake + Ninja, gcc, cppcheck over the generated compilation database.",
            install_command=(
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug "
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            ),
            build_command="cmake --build build --parallel",
            test_command="ctest --test-dir build --output-on-failure",
            static_analysis_command=(
                "cppcheck --project=build/compile_commands.json --enable=warning,performance "
                "--error-exitcode=1"
            ),
            environment={"CC": "gcc"},
        ),
        ToolchainProfile(
            name="cpp-cmake-ninja",
            language="cpp",
            description="CMake + Ninja, clang++, clang-tidy over the compilation database.",
            install_command=(
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug "
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            ),
            build_command="cmake --build build --parallel",
            test_command="ctest --test-dir build --output-on-failure",
            static_analysis_command="run-clang-tidy -p build -quiet",
            environment={"CC": "clang", "CXX": "clang++"},
        ),
        ToolchainProfile(
            name="cpp-make-clang",
            language="cpp",
            description="Makefile project built with clang++, analysed by cppcheck.",
            build_command="make -j4",
            test_command="make check",
            static_analysis_command=(
                "cppcheck --enable=warning,performance,portability --error-exitcode=1 ."
            ),
            environment={
                "CXX": "clang++",
                "CXXFLAGS": "-O2 -Wall -Wextra -Werror -std=c++20",
            },
        ),
    )
}


def profile(name: str) -> ToolchainProfile:
    """Look up a profile by name, failing loudly on a typo."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown toolchain profile {name!r}; known: {', '.join(sorted(PROFILES))}"
        ) from exc


def with_sanitizer(base: ToolchainProfile, sanitizer: str) -> ToolchainProfile:
    """Derive a sanitizer build of a C/C++ profile.

    Sanitizers are mutually exclusive in practice (ASan and TSan cannot coexist),
    so this returns one profile per sanitizer rather than accumulating flags, and
    the caller decides whether to run one validation pass or several.
    """
    if sanitizer not in _SANITIZER_FLAGS:
        raise ValueError(
            f"unknown sanitizer {sanitizer!r}; known: {', '.join(sorted(_SANITIZER_FLAGS))}"
        )
    flags = _SANITIZER_FLAGS[sanitizer]
    environment = {
        **base.environment,
        "CFLAGS": f"{base.environment.get('CFLAGS', '')} {flags}".strip(),
        "CXXFLAGS": f"{base.environment.get('CXXFLAGS', '')} {flags}".strip(),
        "LDFLAGS": f"{base.environment.get('LDFLAGS', '')} {flags}".strip(),
        **_SANITIZER_RUNTIME[sanitizer],
    }
    return replace(
        base,
        name=f"{base.name}-{sanitizer}",
        description=f"{base.description} Built with the {sanitizer} sanitizer.",
        environment=environment,
    )
