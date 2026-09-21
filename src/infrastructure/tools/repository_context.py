"""Repository context selection (spec section 28).

A model is never shown a repository; it is shown a budgeted selection. This v1
selects with the tools that already exist — ``git ls-files`` for the tree,
ripgrep (or grep) for relevance, targeted reads for content — and stops when
either the file budget or the token budget is spent.

About the token budget: it is an **estimate**, computed as four characters per
token. That rule of thumb is roughly right for code under a BPE tokenizer and
can be wrong by a third in either direction, especially for dense punctuation or
non-Latin text. The real count depends on the model's tokenizer, which this
layer deliberately does not know. Callers should therefore treat
``estimated_tokens`` as a budget guard with margin, never as an exact figure.
Every context carries that caveat, and what was left out, in its ``notes``;
``RepositoryContext.render`` does not include them, so a caller that wants the
model to know the view is partial must render the notes itself.

No embeddings, no index, no background job: v1 is greppable and explainable, and
every selection decision it makes can be reproduced by hand.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from collections.abc import Sequence
from pathlib import Path

from domain.ports.repository_context import ContextRequest, FileExcerpt, RepositoryContext
from domain.ports.tools import SandboxExecutor
from domain.value_objects.tools import ExecutionLimits
from domain.value_objects.workspace import WorkspaceHandle
from infrastructure.tools.search import SearchMatch, TextSearchBackend, parse_matches
from infrastructure.workspace.git_cli import GitCommandRunner

__all__ = [
    "CHARS_PER_TOKEN",
    "RipgrepRepositoryContextProvider",
    "estimate_tokens",
    "search_terms",
]

CHARS_PER_TOKEN = 4
"""The estimation constant. Documented, crude, and honest about being crude."""

_DEFAULT_EXCLUDES = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.zip",
    "*.tar*",
    "*.so",
    "*.o",
    "*.a",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.pyc",
    "*.lock",
    ".git/*",
    "node_modules/*",
    "build/*",
    "dist/*",
    "__pycache__/*",
)

_CONTEXT_LIMITS = ExecutionLimits(
    timeout_seconds=60.0, max_output_bytes=4_000_000, network_enabled=False
)


def estimate_tokens(text: str) -> int:
    """Crude character-based token estimate; see the module docstring."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


class RipgrepRepositoryContextProvider:
    """Builds a bounded view of a workspace for one prompt."""

    __slots__ = (
        "_backend",
        "_git",
        "_limits",
        "_match_context_lines",
        "_max_excerpt_lines",
        "_max_tree_entries",
    )

    def __init__(
        self,
        *,
        sandbox: SandboxExecutor,
        git: GitCommandRunner | None = None,
        backend: TextSearchBackend | None = None,
        limits: ExecutionLimits = _CONTEXT_LIMITS,
        match_context_lines: int = 40,
        max_excerpt_lines: int = 200,
        max_tree_entries: int = 400,
    ) -> None:
        self._backend = backend or TextSearchBackend(sandbox=sandbox)
        self._git = git or GitCommandRunner()
        self._limits = limits
        self._match_context_lines = match_context_lines
        self._max_excerpt_lines = max_excerpt_lines
        self._max_tree_entries = max_tree_entries

    async def build(
        self, *, workspace: WorkspaceHandle, request: ContextRequest
    ) -> RepositoryContext:
        root = Path(workspace.path)
        excludes = (*_DEFAULT_EXCLUDES, *request.exclude_globs)
        notes: list[str] = [
            f"token counts are estimated at {CHARS_PER_TOKEN} characters per token "
            "and are approximate",
            f"search engine: {self._backend.engine}",
        ]

        tree: tuple[str, ...] = ()
        if request.include_tree:
            files = await self._list_files(workspace, excludes)
            tree = files[: self._max_tree_entries]
            if len(files) > len(tree):
                notes.append(f"file tree truncated to {len(tree)} of {len(files)} files")

        ranked = await self._rank(workspace, request, excludes)
        excerpts: list[FileExcerpt] = []
        used_tokens = 0
        skipped_for_budget = 0

        for path, line, reason in ranked:
            if len(excerpts) >= request.max_files:
                skipped_for_budget += 1
                continue
            excerpt = await self._excerpt(root, path, line, reason)
            if excerpt is None:
                continue
            cost = estimate_tokens(excerpt.content)
            if used_tokens + cost > request.max_tokens:
                skipped_for_budget += 1
                continue
            excerpts.append(excerpt)
            used_tokens += cost

        if skipped_for_budget:
            notes.append(
                f"{skipped_for_budget} relevant file(s) omitted: "
                f"max_files={request.max_files}, max_tokens={request.max_tokens}"
            )
        if not excerpts:
            notes.append("no file matched the request; only the tree is provided")

        return RepositoryContext(
            excerpts=tuple(excerpts),
            file_tree=tree,
            notes=tuple(notes),
            estimated_tokens=used_tokens + estimate_tokens("\n".join(tree)),
        )

    async def search(
        self, *, workspace: WorkspaceHandle, pattern: str, limit: int = 100
    ) -> Sequence[FileExcerpt]:
        matches = await self._backend.matches(
            workspace=workspace, pattern=pattern, limits=self._limits, max_results=limit
        )
        return tuple(
            FileExcerpt(
                path=match.path,
                content=match.line,
                start_line=match.line_number,
                end_line=match.line_number,
                reason=f"matches {pattern!r}",
            )
            for match in matches
        )

    async def read_file(
        self, *, workspace: WorkspaceHandle, path: str, max_bytes: int = 200_000
    ) -> FileExcerpt | None:
        text = await asyncio.to_thread(_read_contained, Path(workspace.path), path, max_bytes)
        if text is None:
            return None
        lines = text.splitlines()
        return FileExcerpt(
            path=path,
            content=text,
            start_line=1,
            end_line=len(lines),
            reason="requested explicitly",
        )

    # ------------------------------------------------------------------ helpers

    async def _list_files(
        self, workspace: WorkspaceHandle, excludes: Sequence[str]
    ) -> tuple[str, ...]:
        """List tracked and untracked-but-not-ignored files.

        Going through git means the repository's own ``.gitignore`` decides what
        is noise, which is exactly the judgement its authors already made.
        """
        listed = await self._git.run(
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            cwd=workspace.path,
            check=False,
        )
        if listed.succeeded and listed.stdout.strip():
            paths = tuple(line for line in listed.stdout.splitlines() if line)
        else:
            paths = await asyncio.to_thread(_walk, Path(workspace.path))
        return tuple(sorted(path for path in paths if not _excluded(path, excludes)))

    async def _rank(
        self,
        workspace: WorkspaceHandle,
        request: ContextRequest,
        excludes: Sequence[str],
    ) -> tuple[tuple[str, int, str], ...]:
        """Order candidate files: explicit paths first, then by match count.

        Explicit paths win because the caller already knows something the search
        cannot: which file the task is actually about.
        """
        ordered: list[tuple[str, int, str]] = [
            (path, 1, "requested explicitly") for path in request.paths
        ]
        seen = {path for path, _, _ in ordered}

        # Explicit queries first and verbatim, then whatever the objective
        # yields. The objective used to be carried here and never read, so a
        # caller that passed only an objective — which is every caller in
        # production — got an empty ranking and the agents got a file tree with
        # no code in it.
        searches: list[tuple[str, bool]] = [(query, True) for query in request.queries]
        searches += [(term, False) for term in search_terms(request.objective)]

        for query, case_sensitive in searches:
            matches = await self._safe_matches(workspace, query, case_sensitive=case_sensitive)
            counts: dict[str, list[SearchMatch]] = {}
            for match in matches:
                counts.setdefault(match.path, []).append(match)
            for path, hits in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0])):
                if path in seen or _excluded(path, excludes):
                    continue
                seen.add(path)
                ordered.append((path, hits[0].line_number, f"{len(hits)} match(es) for {query!r}"))
        return tuple(ordered)

    async def _safe_matches(
        self, workspace: WorkspaceHandle, query: str, *, case_sensitive: bool = True
    ) -> tuple[SearchMatch, ...]:
        """A malformed query degrades the context; it never fails the run.

        A term taken from an objective is matched without regard to case: the
        user writes "the Ledger class", the code says ``class Ledger``, and
        being strict there would cost recall for nothing. An explicit query is
        matched exactly, because the caller chose those characters.
        """
        result = await self._backend.run(
            workspace=workspace,
            pattern=query,
            limits=self._limits,
            max_results=200,
            case_sensitive=case_sensitive,
        )
        if result.exit_code not in (0, 1):
            return ()
        return parse_matches(result.stdout, limit=200)

    async def _excerpt(self, root: Path, path: str, line: int, reason: str) -> FileExcerpt | None:
        text = await asyncio.to_thread(_read_text, root / path, 400_000)
        if text is None:
            return None
        lines = text.splitlines()
        if len(lines) <= self._max_excerpt_lines:
            return FileExcerpt(
                path=path, content=text, start_line=1, end_line=len(lines), reason=reason
            )
        half = self._match_context_lines // 2
        start = max(1, line - half)
        end = min(len(lines), start + self._max_excerpt_lines - 1)
        return FileExcerpt(
            path=path,
            content="\n".join(lines[start - 1 : end]),
            start_line=start,
            end_line=end,
            reason=f"{reason} (excerpt of a {len(lines)}-line file)",
        )


def _excluded(path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in globs)


def _read_contained(root: Path, relative: str, max_bytes: int) -> str | None:
    """Read a workspace-relative path, refusing anything that escapes the root."""
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return None
    return _read_text(target, max_bytes)


_MAX_DERIVED_TERMS = 6
"""How many terms an objective may contribute. Each one costs a search."""

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# Words that carry no location information. Matching on them would return the
# whole repository, which is the same as returning nothing useful. English and
# French because the objective is written by the user, in the user's language.
# Kept as text rather than a list literal so it stays readable and so the
# formatter leaves it alone.
_STOPWORD_TEXT = """
    a able about across add address adjust after ajoute ajouter all allow also ameliore
    ameliorer an analyse analyser and any are as assure at au aux avec base be because been
    before being better bien both bug build but by can cannot ce ces cette change changer
    check class clean code complete completer compléte correct corrige corriger could create
    creer creé current dans data de dear default des did do does dossier du either elle else
    en ensure erreur error et eux ever every faire fais fait feature fichier fichiers file
    files fix fixed fonction fonctionne for from function get got had has have how however if
    il implement implemente implementer improve in into is issue it its jamais je just la le
    least les let leur like likely look lui ma mais make maniere may me meme merci mes method
    mettre might modifie modifier module moi mon most must my ne need needs neither new no nor
    nos not notre nous of off often on only or other ou our own par pas peu peux please pour
    problem probleme project projet proper qu que qui rather refactor regarde remove rends
    repare reparer repo repository review run sa said say says se ses she should since so soit
    some son support sur sure ta take te tes test tests than that the their them then there
    these they thing this tis to toi ton too tous tout toute toutes tu twas un une update us
    use using value verifie verifier veux vos votre vous want wants was we were what when
    where which while who whom why will with work working would yet you your
"""

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def search_terms(objective: str, *, limit: int = _MAX_DERIVED_TERMS) -> tuple[str, ...]:
    """Terms worth grepping for, taken from a sentence a human wrote.

    Identifiers come first and are kept verbatim: ``compute_total``, ``Ledger``,
    ``parser.decode`` are the words that actually locate code, and a user who
    names one is telling us exactly where to look. Ordinary prose contributes
    only what survives a stopword list, because grepping for "fix" or "projet"
    selects the whole repository, which is no selection at all.

    Returns an empty tuple when the objective says nothing greppable; the
    caller then falls back to the file tree rather than to a random file.
    """
    identifiers: list[str] = []
    words: list[str] = []
    seen: set[str] = set()

    for raw in _WORD_RE.findall(objective):
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        # Shaped like code rather than like prose: snake_case, camelCase, a
        # dotted path, or a digit. These are worth searching whatever they mean.
        if "_" in raw or "." in raw or any(c.isdigit() for c in raw) or raw[1:] != raw[1:].lower():
            identifiers.append(raw)
        elif len(raw) >= 4 and key not in _STOPWORDS:
            words.append(raw)

    return tuple((identifiers + words)[:limit])


def _read_text(path: Path, max_bytes: int) -> str | None:
    """Read a text file, or return ``None`` if it is missing or binary."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes)
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def _walk(root: Path) -> tuple[str, ...]:
    """Filesystem fallback for a workspace that is not a git repository."""
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            found.append(str(path.relative_to(root)))
    return tuple(found)
