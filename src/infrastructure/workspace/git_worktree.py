"""Git-worktree backed workspace manager (spec section 8).

The central invariant: **two writable workspaces never share a path**. Best-of-N
execution runs several coders at the same time against the same repository, and
the only reason that is safe is that each of them owns a directory nobody else
can reach. The invariant is defended twice here — once when a path is allocated
(an explicit refusal, not a comment) and once by construction, because each
workspace directory is named after a freshly generated ``WorkspaceId``.

A git worktree is preferred over a clone: it shares the object database with the
base repository, so creating one is cheap and the resulting branch is directly
mergeable by :meth:`GitWorktreeWorkspaceManager.integrate` without any transfer.
When the repository refuses a worktree (older git, a repository configuration
that forbids it, a path git dislikes), the manager falls back to a local clone
and integrates by fetching from it instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from domain.entities.project import Project
from domain.exceptions import WorkspaceError
from domain.value_objects.identifiers import CandidateId, RunId, WorkspaceId
from domain.value_objects.patch import Patch
from domain.value_objects.workspace import WorkspaceHandle, WorkspaceKind, WorkspaceRole
from infrastructure.workspace.git_cli import GitCommandRunner

__all__ = ["GitWorktreeWorkspaceManager"]


@dataclass(frozen=True, slots=True)
class _Record:
    """What the manager must remember to be able to undo what it created."""

    handle: WorkspaceHandle
    path: Path
    base_repository: Path


class GitWorktreeWorkspaceManager:
    """Creates one isolated checkout per candidate and merges the winner back.

    State (which workspace lives where) is held in memory. A process restart
    therefore forgets existing workspaces: they are not corrupted, but they are
    no longer released automatically. Operators recover with ``git worktree
    prune`` plus a sweep of the workspace root — a deliberate v1 trade-off,
    since persisting this belongs to the repository layer, not here.
    """

    __slots__ = ("_base_lock", "_git", "_lock", "_paths", "_records", "_repository_cache", "_root")

    def __init__(
        self,
        *,
        root: Path | str,
        git: GitCommandRunner | None = None,
        repository_cache: Path | str | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._git = git or GitCommandRunner()
        self._repository_cache = (
            Path(repository_cache).resolve() if repository_cache else self._root / "_repositories"
        )
        self._records: dict[WorkspaceId, _Record] = {}
        self._paths: set[Path] = set()
        self._lock = asyncio.Lock()
        self._base_lock = asyncio.Lock()

    # ------------------------------------------------------------------ create

    async def create(
        self,
        *,
        project: Project,
        run_id: RunId,
        role: WorkspaceRole,
        candidate_id: CandidateId | None = None,
        base_revision: str | None = None,
    ) -> WorkspaceHandle:
        base = await self._base_repository(project)
        revision = base_revision or await self._resolve_revision(base, project.default_branch)

        workspace_id = WorkspaceId.generate()
        path = await self._allocate_path(run_id, role, workspace_id)
        branch = _branch_name(run_id, role, workspace_id) if role.is_writable else None

        try:
            kind = await self._materialize(base, path, revision, branch)
        except BaseException:
            # A half-created workspace is worse than none: it would occupy a
            # path the invariant assumes is free.
            await self._forget(workspace_id, path)
            await asyncio.to_thread(shutil.rmtree, path, True)
            raise

        handle = WorkspaceHandle(
            id=workspace_id,
            run_id=run_id,
            role=role,
            kind=kind,
            path=str(path),
            base_revision=revision,
            branch=branch,
            candidate_id=candidate_id,
        )
        async with self._lock:
            self._records[workspace_id] = _Record(handle=handle, path=path, base_repository=base)
        return handle

    async def get(self, workspace_id: WorkspaceId) -> WorkspaceHandle | None:
        async with self._lock:
            record = self._records.get(workspace_id)
        return record.handle if record else None

    # -------------------------------------------------------------- inspection

    async def diff(self, handle: WorkspaceHandle) -> Patch:
        """Everything the agent changed since the workspace's own HEAD.

        ``--intent-to-add`` is what makes newly created files appear in the
        diff; without it a candidate that only adds files would produce an empty
        patch and look like it did nothing.
        """
        record = self._record(handle)
        await self._git.run("add", "--intent-to-add", "--all", cwd=record.path)
        head = await self._git.run("rev-parse", "--verify", "HEAD", cwd=record.path, check=False)
        against = head.out if head.succeeded else self._git.empty_tree
        result = await self._git.run(
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--find-renames",
            against,
            cwd=record.path,
        )
        return Patch.from_unified_diff(result.stdout, base_revision=against)

    # ----------------------------------------------------------------- mutation

    async def apply_patch(self, handle: WorkspaceHandle, patch: Patch) -> None:
        """Apply a unified diff, or change nothing at all.

        The dry run first: ``git apply --check`` touches no file, so a patch
        that conflicts leaves the workspace byte-for-byte as it was. Half-applied
        patches are the one outcome a repair loop cannot reason about.
        """
        record = self._require_writable(handle)
        if patch.is_empty:
            return

        diff = patch.diff if patch.diff.endswith("\n") else patch.diff + "\n"
        check = await self._git.run(
            "apply", "--check", "--whitespace=nowarn", "-", cwd=record.path, stdin=diff, check=False
        )
        if not check.succeeded:
            raise WorkspaceError(
                "patch does not apply cleanly",
                workspace_id=str(handle.id),
                exit_code=check.exit_code,
                stderr=check.stderr.strip()[:2000],
            )
        await self._git.run(
            "apply", "--whitespace=nowarn", "-", cwd=record.path, stdin=diff, check=True
        )

    async def commit(self, handle: WorkspaceHandle, *, message: str) -> str:
        """Commit everything in the workspace and return the resulting revision.

        A workspace with nothing to commit returns its current revision rather
        than creating an empty commit: candidates that produced no change are a
        normal outcome and must not pollute the history.
        """
        record = self._require_writable(handle)
        await self._git.run("add", "--all", cwd=record.path)
        staged = await self._git.run("diff", "--cached", "--quiet", cwd=record.path, check=False)
        if staged.succeeded:
            return await self._head(record.path)
        await self._git.run("commit", "--no-verify", "-m", message, cwd=record.path)
        return await self._head(record.path)

    async def integrate(self, *, project: Project, handle: WorkspaceHandle, message: str) -> str:
        """Merge the accepted candidate into the base repository.

        The single door through which agent work reaches the project, and
        therefore deliberately strict: the base repository must be on its
        default branch with a clean tree, or nothing happens. Merging into a
        repository somebody else is using would silently mix their work with the
        agent's.
        """
        record = self._require_writable(handle)
        base = record.base_repository
        await self.commit(handle, message=message)

        if (await self._git.run("rev-parse", "--is-bare-repository", cwd=base)).out == "true":
            raise WorkspaceError("cannot integrate into a bare repository", repository=str(base))
        current = await self._git.run("rev-parse", "--abbrev-ref", "HEAD", cwd=base)
        if current.out != project.default_branch:
            raise WorkspaceError(
                "base repository is not on its default branch",
                repository=str(base),
                expected=project.default_branch,
                actual=current.out,
            )
        dirty = await self._git.run("status", "--porcelain", cwd=base)
        if dirty.out:
            raise WorkspaceError("base repository has uncommitted changes", repository=str(base))

        ref = await self._integration_ref(record)
        merge = await self._git.run(
            "merge", "--no-ff", "--no-edit", "-m", message, ref, cwd=base, check=False
        )
        if not merge.succeeded:
            # Leave the base repository exactly as it was found.
            await self._git.run("merge", "--abort", cwd=base, check=False)
            raise WorkspaceError(
                "candidate does not merge cleanly into the base repository",
                workspace_id=str(handle.id),
                branch=handle.branch,
                stdout=merge.stdout.strip()[:2000],
                stderr=merge.stderr.strip()[:2000],
            )
        return await self._head(base)

    # ------------------------------------------------------------------ release

    async def release(self, workspace_id: WorkspaceId) -> None:
        """Destroy a workspace; calling it again is a no-op, never an error.

        Release is on the cancellation path, where it races with normal
        completion, so idempotence is a requirement rather than politeness.

        The workspace's branch is deliberately *kept*: the directory is
        disposable, the commits on it are not, and a rejected candidate is
        exactly the thing an engineer asks to look at afterwards. Pruning those
        refs is a housekeeping job, not part of releasing a directory.
        """
        async with self._lock:
            record = self._records.pop(workspace_id, None)
            if record is not None:
                self._paths.discard(record.path)
        if record is None:
            return

        if record.handle.kind is WorkspaceKind.GIT_WORKTREE:
            await self._git.run(
                "worktree",
                "remove",
                "--force",
                str(record.path),
                cwd=record.base_repository,
                check=False,
            )
        await asyncio.to_thread(shutil.rmtree, record.path, True)
        if record.handle.kind is WorkspaceKind.GIT_WORKTREE:
            await self._git.run("worktree", "prune", cwd=record.base_repository, check=False)

    async def release_run(self, run_id: RunId) -> Sequence[WorkspaceId]:
        async with self._lock:
            targets = [
                workspace_id
                for workspace_id, record in self._records.items()
                if record.handle.run_id == run_id
            ]
        for workspace_id in targets:
            await self.release(workspace_id)
        await asyncio.to_thread(_remove_if_empty, self._root / str(run_id))
        return tuple(targets)

    # ------------------------------------------------------------------ helpers

    async def _allocate_path(
        self, run_id: RunId, role: WorkspaceRole, workspace_id: WorkspaceId
    ) -> Path:
        """Reserve the one directory this workspace is allowed to own.

        This is the isolation invariant made executable: an already-known or
        already-existing path is refused instead of being shared.
        """
        path = self._root / str(run_id) / f"{role.value.lower()}-{workspace_id.value.hex[:12]}"
        async with self._lock:
            if path in self._paths or path.exists():
                raise WorkspaceError(
                    "workspace path is already in use", path=str(path), role=role.value
                )
            self._paths.add(path)
        await asyncio.to_thread(path.parent.mkdir, 0o700, True, True)
        return path

    async def _forget(self, workspace_id: WorkspaceId, path: Path) -> None:
        async with self._lock:
            self._records.pop(workspace_id, None)
            self._paths.discard(path)

    async def _materialize(
        self, base: Path, path: Path, revision: str, branch: str | None
    ) -> WorkspaceKind:
        """Create the checkout, preferring a worktree and falling back to a clone."""
        worktree_args = ["worktree", "add"]
        worktree_args.extend(["--no-track", "-b", branch] if branch else ["--detach"])
        worktree_args.extend([str(path), revision])
        attempt = await self._git.run(*worktree_args, cwd=base, check=False)
        if attempt.succeeded:
            return WorkspaceKind.GIT_WORKTREE

        await asyncio.to_thread(shutil.rmtree, path, True)
        clone = await self._git.run(
            "clone",
            "--local",
            "--no-hardlinks",
            "--no-checkout",
            str(base),
            str(path),
            cwd=base,
            check=False,
        )
        if not clone.succeeded:
            raise WorkspaceError(
                "workspace could not be created as a worktree or a clone",
                path=str(path),
                worktree_error=attempt.stderr.strip()[:1000],
                clone_error=clone.stderr.strip()[:1000],
            )
        checkout = ["checkout"] + (["-b", branch] if branch else []) + [revision]
        await self._git.run(*checkout, cwd=path)
        return WorkspaceKind.CLONE

    async def _integration_ref(self, record: _Record) -> str:
        """The ref the base repository can merge for this workspace.

        A worktree shares the base object database, so its branch is already
        visible there; a clone does not, so its commits are fetched first.
        """
        handle = record.handle
        if handle.kind is WorkspaceKind.GIT_WORKTREE and handle.branch:
            return handle.branch
        ref = f"refs/agentic/{handle.id.value.hex}"
        await self._git.run(
            "fetch",
            "--no-tags",
            str(record.path),
            f"{handle.branch or 'HEAD'}:{ref}",
            cwd=record.base_repository,
        )
        return ref

    async def _base_repository(self, project: Project) -> Path:
        """Resolve — cloning once if needed — the repository workspaces derive from."""
        if project.local_path:
            base = await asyncio.to_thread(_resolve, Path(project.local_path))
            probe = await self._git.run("rev-parse", "--git-dir", cwd=base, check=False)
            if not probe.succeeded:
                raise WorkspaceError("project local_path is not a git repository", path=str(base))
            return base
        if not project.repository_url:  # pragma: no cover - forbidden by Project.__post_init__
            raise WorkspaceError("project has neither a local path nor a repository url")

        cache = self._repository_cache / str(project.id)
        async with self._base_lock:
            if (cache / ".git").exists():
                await self._git.run("fetch", "--all", "--prune", cwd=cache, check=False)
            else:
                await asyncio.to_thread(cache.parent.mkdir, 0o700, True, True)
                await self._git.run("clone", project.repository_url, str(cache), cwd=cache.parent)
        return cache

    async def _resolve_revision(self, base: Path, branch: str) -> str:
        """Pin the exact commit workspaces start from.

        Candidates must all branch from the same commit, otherwise comparing
        them compares different starting points as much as different work.
        """
        for candidate in (f"{branch}^{{commit}}", "HEAD"):
            probe = await self._git.run("rev-parse", "--verify", candidate, cwd=base, check=False)
            if probe.succeeded:
                return probe.out
        raise WorkspaceError(
            "base repository has no resolvable revision", repository=str(base), branch=branch
        )

    async def _head(self, path: Path) -> str:
        return (await self._git.run("rev-parse", "HEAD", cwd=path)).out

    def _record(self, handle: WorkspaceHandle) -> _Record:
        record = self._records.get(handle.id)
        if record is None:
            raise WorkspaceError("unknown workspace", workspace_id=str(handle.id))
        return record

    def _require_writable(self, handle: WorkspaceHandle) -> _Record:
        """Read-only roles are refused here, not merely discouraged.

        The planner and the reviewer read the code; a mutation attempt from them
        is a bug in the orchestration, and failing loudly is how it gets found.
        """
        if not handle.is_writable:
            raise WorkspaceError(
                "workspace is read-only", workspace_id=str(handle.id), role=handle.role.value
            )
        return self._record(handle)


def _branch_name(run_id: RunId, role: WorkspaceRole, workspace_id: WorkspaceId) -> str:
    """A branch per workspace: two candidates can never contend for one ref."""
    return f"agentic/{run_id.value.hex[:8]}/{role.value.lower()}-{workspace_id.value.hex[:12]}"


def _resolve(path: Path) -> Path:
    """``Path.resolve`` touches the filesystem, so it is kept off the event loop."""
    return path.resolve()


def _remove_if_empty(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.rmdir()
