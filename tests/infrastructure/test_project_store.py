"""An upload becomes a repository, or is refused with the reason.

The property under test is the one the old design lacked: after this, a
project owns its files, at a path a client never chose.
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from application.dto.commands import UploadedFile
from domain.exceptions import ProjectUploadError
from domain.value_objects.identifiers import ProjectId
from infrastructure.projects import LocalProjectFilesStore, detect_toolchain


def zipped(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


async def git(cwd: Path, *args: str) -> str:
    """Run git off the event loop; a blocking call inside an async test is the
    thing the ASYNC lint rightly objects to."""
    completed = await asyncio.to_thread(
        subprocess.run, ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout


@pytest.fixture
def store(tmp_path: Path) -> LocalProjectFilesStore:
    return LocalProjectFilesStore(tmp_path / "projects")


# -- files land where the server says --------------------------------------
async def test_files_are_written_under_a_directory_the_server_chose(
    store: LocalProjectFilesStore,
) -> None:
    project_id = ProjectId.generate()

    stored = await store.materialise(
        project_id,
        [UploadedFile("src/app.py", b"print(1)\n"), UploadedFile("README.md", b"# hi\n")],
    )

    assert stored.path == store.root / str(project_id)
    assert (stored.path / "src" / "app.py").read_bytes() == b"print(1)\n"
    assert stored.file_count == 2


async def test_the_upload_is_committed_as_a_baseline(store: LocalProjectFilesStore) -> None:
    """The orchestrator integrates an accepted patch with a merge, which needs
    something to merge onto. An upload with no history has nothing."""
    stored = await store.materialise(ProjectId.generate(), [UploadedFile("a.py", b"x = 1\n")])

    log = await git(stored.path, "log", "--oneline")
    branch = await git(stored.path, "branch", "--show-current")
    assert "baseline" in log
    assert branch.strip() == "main"
    status = await git(stored.path, "status", "--porcelain")
    assert status == "", "the baseline must leave a clean tree"


# -- zips --------------------------------------------------------------------
async def test_a_single_zip_is_extracted(store: LocalProjectFilesStore) -> None:
    blob = zipped({"src/main.py": b"pass\n", "pyproject.toml": b"[project]\nname='x'\n"})

    stored = await store.materialise(ProjectId.generate(), [UploadedFile("project.zip", blob)])

    assert (stored.path / "src" / "main.py").exists()
    assert stored.file_count == 2


async def test_a_zip_with_one_top_folder_has_it_stripped(store: LocalProjectFilesStore) -> None:
    """`zip -r project/` and forge downloads put everything under one folder.
    Keeping it means the project root holds one directory and no code, so every
    build command runs in the wrong place."""
    blob = zipped({"myproj/pyproject.toml": b"[project]\n", "myproj/src/a.py": b"pass\n"})

    stored = await store.materialise(ProjectId.generate(), [UploadedFile("x.zip", blob)])

    assert (stored.path / "pyproject.toml").exists()
    assert not (stored.path / "myproj").exists()


async def test_a_real_top_level_file_means_the_root_is_kept(
    store: LocalProjectFilesStore,
) -> None:
    blob = zipped({"README.md": b"#\n", "src/a.py": b"pass\n"})

    stored = await store.materialise(ProjectId.generate(), [UploadedFile("x.zip", blob)])

    assert (stored.path / "README.md").exists()
    assert (stored.path / "src" / "a.py").exists()


async def test_a_corrupt_zip_is_refused_with_the_reason(store: LocalProjectFilesStore) -> None:
    with pytest.raises(ProjectUploadError, match="not a valid zip"):
        await store.materialise(ProjectId.generate(), [UploadedFile("bad.zip", b"not a zip")])


# -- paths that escape are refused, never sanitised --------------------------
@pytest.mark.parametrize("path", ["../outside.py", "/etc/passwd", "src/../../x", "..\\win.py"])
async def test_a_path_that_escapes_the_project_is_refused(
    store: LocalProjectFilesStore, path: str
) -> None:
    """Rewriting `../etc/passwd` to `etc/passwd` would create a project the
    caller did not upload; the honest answer is a refusal that names it."""
    with pytest.raises(ProjectUploadError, match="unsafe path"):
        await store.materialise(ProjectId.generate(), [UploadedFile(path, b"")])

    assert not any(store.root.rglob("*")) if store.root.exists() else True


async def test_a_zip_slip_entry_is_refused(store: LocalProjectFilesStore) -> None:
    blob = zipped({"../../escape.py": b"evil\n", "ok.py": b"fine\n"})

    with pytest.raises(ProjectUploadError, match="unsafe path"):
        await store.materialise(ProjectId.generate(), [UploadedFile("x.zip", blob)])


async def test_nothing_is_left_behind_when_an_upload_is_refused(
    store: LocalProjectFilesStore,
) -> None:
    """A half-written project directory is worse than none: it looks like a
    project and is not one."""
    project_id = ProjectId.generate()
    blob = zipped({"ok.py": b"fine\n", "../escape.py": b"evil\n"})

    with pytest.raises(ProjectUploadError):
        await store.materialise(project_id, [UploadedFile("x.zip", blob)])

    assert not store.path_for(project_id).exists()


async def test_an_empty_upload_is_refused(store: LocalProjectFilesStore) -> None:
    with pytest.raises(ProjectUploadError, match="no files"):
        await store.materialise(ProjectId.generate(), [])


async def test_the_callers_git_directory_is_not_adopted(store: LocalProjectFilesStore) -> None:
    """Their hooks, their remotes, a baseline commit we did not make."""
    blob = zipped({".git/config": b"[core]\n", ".git/HEAD": b"ref: x\n", "a.py": b"pass\n"})

    stored = await store.materialise(ProjectId.generate(), [UploadedFile("x.zip", blob)])

    head = (stored.path / ".git" / "HEAD").read_text()
    assert "ref: refs/heads/main" in head, "the .git that exists is ours"


# -- toolchain detection -----------------------------------------------------
def test_python_is_detected_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    toolchain = detect_toolchain(tmp_path)

    assert toolchain.language == "python"
    assert toolchain.test_command == "pytest -q"


def test_a_makefile_without_a_test_target_gets_no_test_command(tmp_path: Path) -> None:
    """Inventing `make test` for a Makefile that lacks it would fail every
    candidate for a reason unrelated to the code."""
    (tmp_path / "Makefile").write_text("all:\n\tgcc main.c\n")
    (tmp_path / "main.c").write_text("int main(){}\n")

    toolchain = detect_toolchain(tmp_path)

    assert toolchain.language == "c"
    assert toolchain.build_command == "make -j4"
    assert toolchain.test_command is None


def test_a_makefile_with_a_check_target_uses_it(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("all:\n\ttrue\ncheck:\n\t./run\n")

    assert detect_toolchain(tmp_path).test_command == "make check"


def test_an_unrecognised_project_gets_no_commands_at_all(tmp_path: Path) -> None:
    """Guessed, never invented: with nothing to go on, validation records
    'did not run' rather than a failure the code did not cause."""
    (tmp_path / "notes.txt").write_text("hello\n")

    toolchain = detect_toolchain(tmp_path)

    assert toolchain.language == "unknown"
    assert toolchain.build_command is None
    assert toolchain.test_command is None
