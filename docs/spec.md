# TENSILE — Design Specification

This document defines the authoritative specification for TENSILE’s
graph model, features, and learning setup.

---

## 1. Scope

- **Language:** C
- **Applicability:** Any sufficiently large C codebase
- **Benchmark:** SQLite (used for evaluation)
- **Goal:** Predict structural fragility, not detect bugs directly

---

## 2. Graph Model

### Nodes
- One node per source file (`.c`, `.h`)
- Node ID: repository-relative path

Excluded by default:
- `.git/`
- `test/`, `tests/` (configurable)

---

### Edges

#### Include Edges (v1)
For each file `A`:
- If `A` contains `#include X` and `X` resolves to file `B` in the repo,
  create a directed edge:

A -> B


Unresolved system includes are ignored.

#### Call Edges (v2)
Optional enhancement:
- Cross-file function call dependencies
- Best-effort resolution using symbol mapping

---

## 3. Feature Sets

### Graph Features
- In-degree / out-degree (unique neighbors)
- PageRank
- Betweenness centrality (optional)
- Strongly connected component size

---

### History Features (as-of time T)
- Commit count
- Author count
- Lines added / deleted
- Total churn
- Days since last modification
- Recency-weighted churn (exponential decay)

---

### Code Features
- Lines of code (LOC)
- Pointer token density
- Memory-related call density (`malloc`, `free`, `memcpy`, etc.)
- `goto` frequency
- `extern` / `static` declaration density

---

## 4. Labels (Supervised Learning)

For an as-of date `T` and horizon `Δ`:

- A file is labeled **1 (risky)** if it is modified by any
  bug-fix commit in `(T, T + Δ]`
- Bug-fix commits are identified via message heuristics
  (e.g., `fix`, `bug`, `crash`, `leak`, `regression`)

---

## 5. Outputs

- Dependency graph (`graph.json`)
- Feature table (`features.csv`)
- Labeled dataset (`dataset.csv`)
- Trained model artifact
- Ranked risk report
- Per-file risk explanations

---

## 6. Non-Goals

- Full C semantic correctness
- Compilation or build-system integration
- Bug detection or vulnerability scanning

TENSILE intentionally focuses on **risk modeling**, not correctness checking.
