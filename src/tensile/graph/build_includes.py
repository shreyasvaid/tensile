from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tensile.graph.io import GraphEdge

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^">]+)[">]')

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "build",
    "dist",
    "out",
    ".venv",
    "venv",
    "__pycache__",
    "test",
    "tests",
}

@dataclass
class BuildGraphResult:
    nodes: List[str]
    edges: List[GraphEdge]
    unresolved_includes: Dict[str, int]
    include_total: Dict[str, int]
    unresolved_targets: Dict[str, int]  # include token -> count


def iter_c_files(repo_root: Path, exclude_dirs: Set[str]) -> Iterable[Path]:
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".c", ".h"}:
            continue
        parts = set(p.relative_to(repo_root).parts)
        if parts & exclude_dirs:
            continue
        yield p

def repo_rel(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()

def build_repo_file_index(repo_root: Path, exclude_dirs: Set[str]) -> Dict[str, Path]:
    """
    Maps repo-relative paths (posix) -> absolute Path.
    """
    idx: Dict[str, Path] = {}
    for f in iter_c_files(repo_root, exclude_dirs):
        idx[repo_rel(repo_root, f)] = f
    return idx

def resolve_include(
    including_file: Path,
    repo_root: Path,
    include_token: str,
    kind: str,
    include_dirs: List[Path],
    repo_index: Dict[str, Path],
) -> Optional[str]:
    """
    Returns repo-relative path of resolved include target if it exists inside repo,
    else None.

    Rules:
    - For quoted includes ("..."): try relative to including_file directory first,
      then search include_dirs in order.
    - For angle includes (<...>): only search include_dirs.
    """
    token = include_token.strip()

    candidates: List[Path] = []

    if kind == '"':
        candidates.append((including_file.parent / token).resolve())

    for d in include_dirs:
        candidates.append((d / token).resolve())

    # Keep only candidates that are within repo_root and exist.
    repo_root_resolved = repo_root.resolve()
    for c in candidates:
        if not c.exists() or not c.is_file():
            continue
        try:
            rel = c.relative_to(repo_root_resolved).as_posix()
        except ValueError:
            # Not inside repo
            continue
        if rel in repo_index:
            return rel
    return None

def build_includes_graph(
    repo_root: Path,
    include_dirs: Optional[List[Path]] = None,
    exclude_dirs: Optional[Set[str]] = None,
) -> BuildGraphResult:
    """
    Build file-level dependency graph using #include relations.

    Nodes: all .c/.h files under repo_root (minus excludes)
    Edges: A -> B if A includes B and B resolves to a repo file
    """
    unresolved_targets: Dict[str, int] = {}
    repo_root = repo_root.resolve()
    exclude_dirs = exclude_dirs or set(DEFAULT_EXCLUDE_DIRS)

    dirs: List[Path] = [repo_root]

    for guess in ["include", "src", "ext"]:
        d = repo_root / guess
        if d.exists() and d.is_dir():
            dirs.append(d)


    if include_dirs:
        dirs.extend([d.resolve() for d in include_dirs])

    # De-duplicate while preserving order
    seen = set()
    include_dirs_final: List[Path] = []
    for d in dirs:
        if d not in seen:
            include_dirs_final.append(d)
            seen.add(d)

    repo_index = build_repo_file_index(repo_root, exclude_dirs)
    nodes = list(repo_index.keys())

    edges: List[GraphEdge] = []
    unresolved_includes: Dict[str, int] = {n: 0 for n in nodes}
    include_total: Dict[str, int] = {n: 0 for n in nodes}

    for rel, abs_path in repo_index.items():
        try:
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            kind = m.group(1)  # < or "
            target = m.group(2)
            include_total[rel] += 1

            # kind is '<' or '"'
            resolved = resolve_include(
                including_file=abs_path,
                repo_root=repo_root,
                include_token=target,
                kind='"' if kind == '"' else "<",
                include_dirs=include_dirs_final,
                repo_index=repo_index,
            )
            if resolved is None:
                unresolved_includes[rel] += 1
                unresolved_targets[target] = unresolved_targets.get(target, 0) + 1
                continue
            if resolved == rel:
                continue

            edges.append(GraphEdge(src=rel, dst=resolved, type="include"))

    return BuildGraphResult(
        nodes=nodes,
        edges=edges,
        unresolved_includes=unresolved_includes,
        include_total=include_total,
        unresolved_targets=unresolved_targets,
    )

