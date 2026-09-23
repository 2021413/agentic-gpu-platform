"""Workspace isolation against real git repositories.

Everything here runs on throwaway repositories created in ``tmp_path``: git's
behaviour around worktrees, ``apply --check`` and merges is the thing under
test, so faking it would test nothing.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from domain.entities.project import Project
from domain.exceptions import WorkspaceError
from domain.ports.workspace import WorkspaceManager
from domain.value_objects.identifiers import CandidateId, ProjectId, RunId
from domain.value_objects.patch import Patch
from domain.value_objects.workspace import WorkspaceKind, WorkspaceRole
from infrastructure.workspace.git_cli import GitCommandRunner, GitResult
from infrastructure.workspace.git_worktree import GitWorktreeWorkspaceManager


def run_git(repository: Path, *args: str) -> str:
    """Drive git synchronously while setting a test up, with no host config."""
    environment = {
        **os.environ,
        "HOME": str(repository),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    completed = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@invalid", *args],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def path_exists(path: str | Path) -> bool:
    """Filesystem probes live in helpers: blocking calls do not belong in async bodies."""
    return Path(path).exists()


def is_directory(path: str | Path) -> bool:
    return Path(path).is_dir()


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_symlink(link: Path, target: Path) -> None:
    link.symlink_to(target, target_is_directory=True)


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).samefile(right)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    run_git(base, "init", "-b", "main")
    (base / "README.md").write_text("hello\n", encoding="utf-8")
    (base / "src").mkdir()
    (base / "src" / "app.py").write_text("def main() -> int:\n    return 0\n", encoding="utf-8")
    run_git(base, "add", "--all")
    run_git(base, "commit", "-m", "initial")
    return base


@pytest.fixture
def project(repository: Path) -> Project:
    return Project.create(
        project_id=ProjectId.generate(),
        name="demo",
        now=datetime(2026, 1, 1, tzinfo=UTC),
        local_path=str(repository),
        default_branch="main",
    )


@pytest.fixture
def manager(tmp_path: Path) -> GitWorktreeWorkspaceManager:
    return GitWorktreeWorkspaceManager(root=tmp_path / "workspaces")


async def test_two_candidates_never_share_a_path_or_see_each_other(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """The invariant the whole parallelisation rests on."""
    run_id = RunId.generate()
    first = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    second = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )

    assert first.path != second.path
    assert first.branch != second.branch
    assert not same_path(first.path, second.path)

    (Path(first.path) / "only_in_first.txt").write_text("a\n", encoding="utf-8")
    (Path(first.path) / "README.md").write_text("changed by first\n", encoding="utf-8")

    assert not (Path(second.path) / "only_in_first.txt").exists()
    assert (Path(second.path) / "README.md").read_text(encoding="utf-8") == "hello\n"

    first_patch = await manager.diff(first)
    second_patch = await manager.diff(second)
    assert "only_in_first.txt" in first_patch.changed_paths
    assert second_patch.is_empty


async def test_allocating_a_path_twice_is_refused(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """Defends the invariant directly, without relying on uuid uniqueness."""
    run_id = RunId.generate()
    handle = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    with pytest.raises(WorkspaceError, match="already in use"):
        await manager._allocate_path(run_id, WorkspaceRole.CANDIDATE, handle.id)


async def test_read_only_roles_cannot_be_mutated(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    planner = await manager.create(
        project=project, run_id=RunId.generate(), role=WorkspaceRole.PLANNER
    )
    assert planner.is_writable is False
    assert planner.branch is None
    assert (Path(planner.path) / "README.md").read_text(encoding="utf-8") == "hello\n"

    with pytest.raises(WorkspaceError, match="read-only"):
        await manager.commit(planner, message="nope")
    with pytest.raises(WorkspaceError, match="read-only"):
        await manager.apply_patch(planner, Patch.from_unified_diff("diff --git a/x b/x\n"))


async def test_diff_reports_uncommitted_changes_including_new_files(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    (Path(handle.path) / "src" / "app.py").write_text(
        "def main() -> int:\n    return 1\n", encoding="utf-8"
    )
    (Path(handle.path) / "NOTES.md").write_text("note\n", encoding="utf-8")

    patch = await manager.diff(handle)
    assert set(patch.changed_paths) == {"src/app.py", "NOTES.md"}
    assert patch.total_churn > 0
    assert "def main" in patch.diff


async def test_a_conflicting_patch_leaves_the_workspace_untouched(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    run_id = RunId.generate()
    author = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    target = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )

    (Path(author.path) / "README.md").write_text("from the author\n", encoding="utf-8")
    patch = await manager.diff(author)

    (Path(target.path) / "README.md").write_text("conflicting content\n", encoding="utf-8")
    before = await manager.diff(target)

    with pytest.raises(WorkspaceError, match="does not apply cleanly"):
        await manager.apply_patch(target, patch)

    assert (Path(target.path) / "README.md").read_text(encoding="utf-8") == "conflicting content\n"
    assert (await manager.diff(target)).diff == before.diff


async def test_a_clean_patch_applies(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    run_id = RunId.generate()
    author = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    target = await manager.create(
        project=project,
        run_id=run_id,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    (Path(author.path) / "README.md").write_text("hello\nworld\n", encoding="utf-8")

    await manager.apply_patch(target, await manager.diff(author))

    assert (Path(target.path) / "README.md").read_text(encoding="utf-8") == "hello\nworld\n"


async def test_commit_returns_a_revision_and_never_creates_an_empty_commit(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    (Path(handle.path) / "README.md").write_text("committed\n", encoding="utf-8")

    revision = await manager.commit(handle, message="work")
    assert len(revision) == 40
    assert revision != handle.base_revision

    unchanged = await manager.commit(handle, message="nothing to do")
    assert unchanged == revision


async def test_integrate_is_how_work_reaches_the_project(
    manager: GitWorktreeWorkspaceManager, project: Project, repository: Path
) -> None:
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    (Path(handle.path) / "src" / "app.py").write_text(
        "def main() -> int:\n    return 42\n", encoding="utf-8"
    )

    merged = await manager.integrate(project=project, handle=handle, message="accept candidate")

    assert (repository / "src" / "app.py").read_text(encoding="utf-8").endswith("return 42\n")
    assert run_git(repository, "rev-parse", "HEAD").strip() == merged
    assert "accept candidate" in run_git(repository, "log", "--oneline", "-5")


async def test_integrate_refuses_a_dirty_base_repository(
    manager: GitWorktreeWorkspaceManager, project: Project, repository: Path
) -> None:
    """Merging into somebody else's work in progress is never acceptable."""
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    (Path(handle.path) / "README.md").write_text("candidate\n", encoding="utf-8")
    (repository / "README.md").write_text("somebody was editing this\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="uncommitted changes"):
        await manager.integrate(project=project, handle=handle, message="accept")

    assert (repository / "README.md").read_text(encoding="utf-8") == "somebody was editing this\n"


async def test_release_is_idempotent(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    path = Path(handle.path)
    assert is_directory(path)

    await manager.release(handle.id)
    await manager.release(handle.id)

    assert not path_exists(path)
    assert await manager.get(handle.id) is None


async def test_release_run_releases_every_workspace_of_the_run(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    run_id = RunId.generate()
    other_run = RunId.generate()
    handles = [
        await manager.create(
            project=project,
            run_id=run_id,
            role=WorkspaceRole.CANDIDATE,
            candidate_id=CandidateId.generate(),
        )
        for _ in range(2)
    ]
    survivor = await manager.create(
        project=project,
        run_id=other_run,
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )

    released = await manager.release_run(run_id)

    assert set(released) == {handle.id for handle in handles}
    assert all(not path_exists(handle.path) for handle in handles)
    assert is_directory(survivor.path)


class _WorktreeRefusingGit(GitCommandRunner):
    """A git that cannot create worktrees, to exercise the clone fallback."""

    async def run(self, *args: str, **kwargs: Any) -> GitResult:
        if args[:2] == ("worktree", "add"):
            return GitResult(
                args=args, exit_code=128, stdout="", stderr="worktrees are not supported here"
            )
        return await super().run(*args, **kwargs)


async def test_a_repository_without_worktrees_falls_back_to_a_clone(
    tmp_path: Path, project: Project, repository: Path
) -> None:
    manager = GitWorktreeWorkspaceManager(root=tmp_path / "workspaces", git=_WorktreeRefusingGit())
    handle = await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )
    assert handle.kind is WorkspaceKind.CLONE

    (Path(handle.path) / "README.md").write_text("from a clone\n", encoding="utf-8")
    await manager.integrate(project=project, handle=handle, message="accept clone")

    assert (repository / "README.md").read_text(encoding="utf-8") == "from a clone\n"


def test_the_manager_satisfies_the_domain_port(manager: GitWorktreeWorkspaceManager) -> None:
    assert isinstance(manager, WorkspaceManager)


# ---------------------------------------------------------------------------
# Whole-file writes.
#
# This is the path a coder answer now takes end to end: the model returns file
# contents, the workspace writes them, git computes the diff. It replaced the
# unified-diff path, which a real model got wrong twice in a row and failed a
# run with "No valid patches in input". Everything below runs against a real
# git repository on disk, because what is under test is exactly the part a
# double would have papered over.
# ---------------------------------------------------------------------------


async def writable_workspace(manager: GitWorktreeWorkspaceManager, project: Project) -> Any:
    return await manager.create(
        project=project,
        run_id=RunId.generate(),
        role=WorkspaceRole.CANDIDATE,
        candidate_id=CandidateId.generate(),
    )


async def test_written_files_become_a_diff_git_can_read(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """The whole contract in one test: contents in, unified diff out."""
    handle = await writable_workspace(manager, project)

    written = await manager.write_files(
        handle,
        {
            "src/app.py": "def main() -> int:\n    return 1\n",
            "src/retry/backoff.py": "WAIT = 3\n",
        },
    )

    assert sorted(written) == ["src/app.py", "src/retry/backoff.py"]
    patch = await manager.diff(handle)
    assert sorted(patch.changed_paths) == ["src/app.py", "src/retry/backoff.py"]
    assert not patch.is_empty
    # A new file in a directory that did not exist must be created, not skipped.
    assert read_text(Path(handle.path) / "src" / "retry" / "backoff.py") == "WAIT = 3\n"
    # And an existing file is replaced wholesale, not appended to.
    assert read_text(Path(handle.path) / "src" / "app.py").count("return") == 1


async def test_writing_the_same_file_twice_keeps_the_last_content(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """A repair loop rewrites the same file; the newest content must win."""
    handle = await writable_workspace(manager, project)

    await manager.write_files(handle, {"src/app.py": "first\n"})
    await manager.write_files(handle, {"src/app.py": "second\n"})

    assert read_text(Path(handle.path) / "src" / "app.py") == "second\n"


@pytest.mark.parametrize(
    "path",
    [
        "../escaped.txt",
        "src/../../escaped.txt",
        "/etc/escaped.txt",
    ],
)
async def test_a_path_leaving_the_workspace_is_refused(
    manager: GitWorktreeWorkspaceManager, project: Project, tmp_path: Path, path: str
) -> None:
    """A coder answer is untrusted input, and this is the filesystem boundary."""
    handle = await writable_workspace(manager, project)

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        await manager.write_files(handle, {path: "owned\n"})

    assert not path_exists(Path(handle.path).parent / "escaped.txt")
    assert not path_exists(tmp_path / "escaped.txt")


async def test_a_symlink_cannot_be_used_to_escape(
    manager: GitWorktreeWorkspaceManager, project: Project, tmp_path: Path
) -> None:
    """Containment is checked after resolution, so a link out is not a bypass."""
    handle = await writable_workspace(manager, project)
    outside = tmp_path / "outside"
    make_directory(outside)
    make_symlink(Path(handle.path) / "link", outside)

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        await manager.write_files(handle, {"link/owned.txt": "owned\n"})

    assert not path_exists(outside / "owned.txt")


async def test_a_refused_batch_writes_nothing_at_all(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """One bad path must not leave a half-applied workspace behind."""
    handle = await writable_workspace(manager, project)

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        await manager.write_files(handle, {"src/good.py": "kept\n", "../escaped.txt": "owned\n"})

    assert not path_exists(Path(handle.path) / "src" / "good.py")
    assert (await manager.diff(handle)).is_empty


async def test_a_read_only_workspace_refuses_writes(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """The planner reads the repository; it must not be able to change it."""
    planner = await manager.create(
        project=project, run_id=RunId.generate(), role=WorkspaceRole.PLANNER
    )

    with pytest.raises(WorkspaceError, match="read-only"):
        await manager.write_files(planner, {"src/app.py": "nope\n"})


# ---------------------------------------------------------------------------
# Build artefacts.
#
# Validation runs the project's build, which writes. `git add --all` then
# swept the result into the patch: a real run against a real model produced
# six changed files, four of which were .pyc. Those reach the reviewer, cost
# tokens, and land in the caller's repository on merge.
# ---------------------------------------------------------------------------


async def test_build_artefacts_stay_out_of_the_patch(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    handle = await writable_workspace(manager, project)
    await manager.write_files(handle, {"src/app.py": "def main() -> int:\n    return 1\n"})

    # What a build leaves behind, next to what the agent actually wrote.
    root = Path(handle.path)
    make_directory(root / "src" / "__pycache__")
    (root / "src" / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00compiled")
    make_directory(root / "target")
    (root / "target" / "app.o").write_bytes(b"\x00object")
    (root / "app.log").write_text("noise\n", encoding="utf-8")

    patch = await manager.diff(handle)

    assert "src/app.py" in patch.changed_paths
    polluted = [p for p in patch.changed_paths if p != "src/app.py"]
    assert not polluted, f"the patch carries build output: {polluted}"


async def test_the_exclusions_do_not_touch_the_caller_s_repository(
    manager: GitWorktreeWorkspaceManager, project: Project, repository: Path
) -> None:
    """They belong to the worktree, not to the project.

    Writing a .gitignore would be a change the caller never asked for, and it
    would show up in the very patch it is meant to keep clean.
    """
    await writable_workspace(manager, project)

    assert not path_exists(repository / ".gitignore")
    assert run_git(repository, "status", "--porcelain") == ""


async def test_a_file_the_agent_deliberately_writes_is_never_excluded(
    manager: GitWorktreeWorkspaceManager, project: Project
) -> None:
    """The rule must not silently drop work.

    A project whose source genuinely lives under a matched name would lose it,
    so an explicit write always wins over the exclusion.
    """
    handle = await writable_workspace(manager, project)

    await manager.write_files(
        handle,
        {
            "target/keep.py": "# deliberately written by the coder\n",
            "logs/app.log": "written on purpose\n",
        },
    )

    patch = await manager.diff(handle)
    assert "target/keep.py" in patch.changed_paths, patch.changed_paths
    assert "logs/app.log" in patch.changed_paths, patch.changed_paths


async def test_a_project_with_a_gitignore_still_works(
    manager: GitWorktreeWorkspaceManager, project: Project, repository: Path
) -> None:
    """The case the unit fixture did not have, and production did.

    `git add --all -- <any pathspec>` exits 1 the moment a .gitignore covers a
    file that is present, even though the staging it performed is correct. An
    earlier version of the exclusions passed a pathspec there and took down
    every run on any repository with a .gitignore — which is most of them.
    """
    (repository / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    run_git(repository, "add", ".gitignore")
    run_git(repository, "commit", "-m", "ignore build output")

    handle = await writable_workspace(manager, project)
    root = Path(handle.path)
    make_directory(root / "__pycache__")
    (root / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00compiled")

    await manager.write_files(handle, {"src/app.py": "def main() -> int:\n    return 1\n"})

    patch = await manager.diff(handle)
    assert patch.changed_paths == ("src/app.py",), patch.changed_paths

    # And the whole cycle must still complete, which is what actually broke.
    revision = await manager.commit(handle, message="work")
    assert revision
