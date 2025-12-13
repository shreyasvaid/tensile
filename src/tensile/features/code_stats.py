from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# Simple, deterministic token matchers (v1 lexical features)
RE_MALLOC = re.compile(r"\b(malloc|calloc|realloc)\s*\(")
RE_FREE = re.compile(r"\bfree\s*\(")
RE_MEMFUN = re.compile(r"\b(memcpy|memmove|strcpy|strncpy|sprintf|snprintf)\s*\(")
RE_GOTO = re.compile(r"\bgoto\b")
RE_EXTERN = re.compile(r"\bextern\b")
RE_STATIC = re.compile(r"\bstatic\b")

def _count_substring(s: str, sub: str) -> int:
    return s.count(sub)

def _nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())

def compute_code_stats(repo_root: Path, files: Iterable[str]) -> pd.DataFrame:
    """
    Compute cheap lexical code signals for each file at current checkout.

    Columns:
      file
      c_loc
      c_ptr_tokens_per_loc
      c_arrow_tokens_per_loc
      c_malloc_calls_per_loc
      c_free_calls_per_loc
      c_mem_calls_per_loc
      c_goto_per_loc
      c_extern_per_loc
      c_static_per_loc
    """
    repo_root = repo_root.resolve()
    rows: list[dict] = []

    for f in files:
        path = repo_root / f
        if not path.exists() or not path.is_file():
            # If missing (renames, generated headers), keep row with NaNs/zeros
            rows.append({"file": f, "c_loc": 0})
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        loc = _nonempty_lines(text)
        if loc <= 0:
            loc = 1  # avoid divide by zero; keep per_loc stable

        star = _count_substring(text, "*")
        arrow = _count_substring(text, "->")

        malloc_ct = len(RE_MALLOC.findall(text))
        free_ct = len(RE_FREE.findall(text))
        mem_ct = len(RE_MEMFUN.findall(text))
        goto_ct = len(RE_GOTO.findall(text))
        extern_ct = len(RE_EXTERN.findall(text))
        static_ct = len(RE_STATIC.findall(text))

        rows.append(
            {
                "file": f,
                "c_loc": loc,
                "c_ptr_tokens_per_loc": star / loc,
                "c_arrow_tokens_per_loc": arrow / loc,
                "c_malloc_calls_per_loc": malloc_ct / loc,
                "c_free_calls_per_loc": free_ct / loc,
                "c_mem_calls_per_loc": mem_ct / loc,
                "c_goto_per_loc": goto_ct / loc,
                "c_extern_per_loc": extern_ct / loc,
                "c_static_per_loc": static_ct / loc,
            }
        )

    df = pd.DataFrame(rows)
    # Fill missing columns for missing files
    for col in df.columns:
        if col != "file":
            df[col] = df[col].fillna(0)
    return df
