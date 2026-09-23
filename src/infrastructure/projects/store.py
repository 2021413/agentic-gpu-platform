"""Materialising an upload into a repository the orchestrator can branch from.

Every project used to point at one bind mount, `/projects/current`, whatever
its name said. Picking a project in the viewer therefore ran the agents against
whatever happened to be mounted at that moment — which is how a run named after
one repository planned, coded and reviewed a completely different one, and had
every candidate rejected for not doing what was asked. A project that does not
own its files cannot be selected from an interface at all.

Now each project owns a directory under one server-managed root, written from
what the caller uploaded and committed as the baseline the first worktree
branches from. The path is decided here, never accepted from a client.
"""

from __future__ import annotations

import asyncio
import io
import os
import shlex
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from application.dto.commands import UploadedFile
from application.ports import MaterialisedProject
from domain.entities.project import ToolchainConfig
from domain.exceptions import ProjectUploadError
from domain.value_objects.identifiers import ProjectId

__all__ = ["MAX_UPLOAD_BYTES", "LocalProjectFilesStore", "detect_toolchain", "missing_executable"]

# Generous for source code, and a hard stop for anything that is not. A zip
# that expands to more than this is refused before a byte is written, because
# the disk it would fill is the one every worktree on this host branches from.
MAX_UPLOAD_BYTES: Final = 200 * 1024 * 1024

# What a directory listing may not contain: nothing that escapes the project.
_FORBIDDEN_PARTS: Final = frozenset({"..", ""})


def _safe_relative(raw: str) -> PurePosixPath:
    """A path that stays inside the project, or a refusal that names it.

    Zip entries and multipart filenames are attacker-controlled strings. A
    single `../` would write outside the project root, and an absolute path
    would write wherever it pointed. Both are refused rather than sanitised:
    silently rewriting `../etc/passwd` to `etc/passwd` produces a project the
    caller did not upload.
    """
    text = raw.replace("\\", "/")
    # Refused before any normalisation: stripping the leading slash first would
    # turn `/etc/passwd` into `etc/passwd`, which is the sanitising this
    # function exists to not do.
    if not text.strip() or text.startswith("/") or text.startswith("./"):
        raise ProjectUploadError(f"unsafe path in upload: {raw!r}", path=raw)
    path = PurePosixPath(text.rstrip("/"))
    if any(part in _FORBIDDEN_PARTS for part in path.parts):
        raise ProjectUploadError(f"unsafe path in upload: {raw!r}", path=raw)
    return path


Entries = list[tuple[PurePosixPath, bytes]]


def _strip_common_root(files: Entries) -> Entries:
    """Drop a single top-level directory that every file shares.

    Archives made by `zip -r project/` or downloaded from a forge put everything
    under one folder. Keeping it would make the project root contain exactly
    one directory and no code, so every build command would run in the wrong
    place. Only a root shared by *all* entries is removed; a project with a
    real top-level `src/` next to a `README` keeps both.
    """
    if len(files) < 1:
        return files
    heads = {path.parts[0] for path, _ in files}
    if len(heads) != 1:
        return files
    if any(len(path.parts) == 1 for path, _ in files):
        # A file at the top level means the "root" is a real file, not a folder.
        return files
    return [(PurePosixPath(*path.parts[1:]), content) for path, content in files]


def _unpack_zip(blob: bytes) -> Entries:
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise ProjectUploadError("the upload is not a valid zip archive") from exc
    total = 0
    files: Entries = []
    with archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            if name.startswith("__MACOSX/") or PurePosixPath(name).name == ".DS_Store":
                continue
            total += entry.file_size
            if total > MAX_UPLOAD_BYTES:
                raise ProjectUploadError(
                    f"the archive expands past {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                    limit_bytes=MAX_UPLOAD_BYTES,
                )
            files.append((_safe_relative(name), archive.read(entry)))
    return files


def _files_to_write(files: Sequence[UploadedFile]) -> Entries:
    if not files:
        raise ProjectUploadError("the upload contains no files")
    if len(files) == 1 and files[0].path.lower().endswith(".zip"):
        entries = _unpack_zip(files[0].content)
    else:
        entries = [(_safe_relative(f.path), f.content) for f in files]
    total = sum(len(content) for _, content in entries)
    if total > MAX_UPLOAD_BYTES:
        raise ProjectUploadError(
            f"the upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            limit_bytes=MAX_UPLOAD_BYTES,
        )
    entries = _strip_common_root(entries)
    # `.git` from the caller's machine is not ours to adopt: its hooks, its
    # config and its remotes are theirs, and a baseline commit we did not make
    # is one we cannot reason about.
    entries = [(p, c) for p, c in entries if p.parts[0] != ".git"]
    if not entries:
        raise ProjectUploadError("the upload contains no files")
    return entries


# Marker file -> toolchain, first match wins. A table rather than a ladder of
# returns, so adding a language is one line and the order of precedence is
# visible at a glance (a repository with both package.json and pyproject is
# treated as JavaScript, which is what run-on.sh always did).
_BY_MARKER: Final[tuple[tuple[str, ToolchainConfig], ...]] = (
    (
        "package.json",
        ToolchainConfig(
            language="javascript",
            build_command="npm run build --if-present",
            test_command="npm test",
        ),
    ),
    (
        "pyproject.toml",
        ToolchainConfig(
            language="python", build_command="python -m compileall -q .", test_command="pytest -q"
        ),
    ),
    (
        "setup.py",
        ToolchainConfig(
            language="python", build_command="python -m compileall -q .", test_command="pytest -q"
        ),
    ),
    (
        "Cargo.toml",
        ToolchainConfig(language="rust", build_command="cargo build", test_command="cargo test"),
    ),
    (
        "go.mod",
        ToolchainConfig(
            language="go", build_command="go build ./...", test_command="go test ./..."
        ),
    ),
    (
        "CMakeLists.txt",
        ToolchainConfig(
            language="cpp",
            build_command="cmake --build build --parallel",
            test_command="ctest --test-dir build --output-on-failure",
        ),
    ),
)


def detect_toolchain(root: Path) -> ToolchainConfig:
    """Guess build and test commands from what the repository contains.

    Ported from `run-on.sh`, which did this in shell; here so the API and the
    script cannot drift into different guesses. Guessed, never invented: a
    project with no recognisable marker gets *no* commands, so validation is
    recorded as "did not run" rather than as a failure the code did not cause.
    """
    for marker, toolchain in _BY_MARKER:
        if (root / marker).is_file():
            return toolchain
    makefile = root / "Makefile"
    if not makefile.is_file():
        return ToolchainConfig(language="unknown", build_command=None, test_command=None)
    # C or C++ by Makefile. The build target is whatever `make` does by
    # default; the test target only exists if the project wrote one.
    text = makefile.read_text(encoding="utf-8", errors="replace")
    targets = {line.split(":", 1)[0] for line in text.splitlines() if ":" in line}
    test = next((f"make {target}" for target in ("test", "check") if target in targets), None)
    language = "cpp" if any(root.rglob("*.cpp")) else "c"
    return ToolchainConfig(language=language, build_command="make -j4", test_command=test)


def missing_executable(command: str | None) -> str | None:
    """The executable a command names, if it cannot be found where tools run.

    Tools are child processes of this very service, so `shutil.which` here
    answers the same question the sandbox will ask. It is asked at upload time
    because the alternative was observed: a test command of `analyse ce projet`
    — a sentence, typed into a field that looked like it wanted one — was
    accepted, the planner and the coder were paid for on the GPU, and only then
    did validation report `[Errno 2] No such file or directory`. A command that
    cannot start is refused before a run is spent on it, with its first word
    named, which is what `run-on.sh` already did in shell.

    ``None`` when there is nothing to object to, including no command at all.
    """
    if not command or not command.strip():
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return command.strip().split()[0]
    if not words:
        return None
    executable = words[0]
    if "/" in executable:
        return None if Path(executable).exists() else executable
    return None if shutil.which(executable) else executable


@dataclass(frozen=True, slots=True)
class LocalProjectFilesStore:
    """One directory per project under `root`, each a git repository."""

    root: Path

    def path_for(self, project_id: ProjectId) -> Path:
        return self.root / str(project_id)

    async def materialise(
        self, project_id: ProjectId, files: Sequence[UploadedFile]
    ) -> MaterialisedProject:
        entries = _files_to_write(files)
        target = self.path_for(project_id)
        return await asyncio.to_thread(self._write, target, entries)

    def _write(self, target: Path, entries: Entries) -> MaterialisedProject:
        if target.exists():
            # An id is minted per upload, so this is a collision with a
            # half-finished earlier attempt. Start clean rather than merge.
            shutil.rmtree(target)
        target.mkdir(parents=True)
        try:
            for relative, content in entries:
                destination = target / Path(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            toolchain = detect_toolchain(target)
            self._commit_baseline(target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return MaterialisedProject(path=target, toolchain=toolchain, file_count=len(entries))

    @staticmethod
    def _commit_baseline(target: Path) -> None:
        """The commit every worktree will branch from.

        The orchestrator integrates an accepted patch with a controlled merge,
        which needs a base to merge onto; an upload with no history has none.
        The identity is fixed and named for what it is, so a `git log` of the
        project says who made this commit and why.
        """
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "agentic",
            "GIT_AUTHOR_EMAIL": "agentic@localhost",
            "GIT_COMMITTER_NAME": "agentic",
            "GIT_COMMITTER_EMAIL": "agentic@localhost",
        }
        for argv in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "add", "-A"],
            ["git", "commit", "-q", "-m", "baseline: uploaded project"],
        ):
            completed = subprocess.run(
                argv, cwd=target, env=env, capture_output=True, text=True, check=False
            )
            if completed.returncode != 0:
                raise ProjectUploadError(
                    "could not create the project's baseline commit",
                    command=" ".join(argv),
                    stderr=completed.stderr.strip()[:500],
                )
