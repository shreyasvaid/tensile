# TENSILE

**TENSILE** is a graph-based code risk analysis tool for large **C codebases**.  
It models source files as a dependency graph and learns which components are
**structurally fragile** by combining graph topology with historical change and
defect data.

Rather than attempting to find bugs directly, TENSILE answers a higher-level
question:

> *Given the current structure and history of a codebase, which files are most
> likely to be involved in future bug fixes?*

TENSILE is designed to be **language-specific but repository-agnostic**.
While it is applicable to any sufficiently large C codebase, SQLite is used as
the primary evaluation benchmark due to its rich history and well-documented
defect patterns.

---

## Objectives

- Model a codebase as a **directed dependency graph**
- Combine **structural metrics** (centrality, coupling) with **historical signals**
  (churn, recency, bug-fix history)
- Learn a **risk score** that ranks files by predicted fragility
- Provide **explanations** for why a file is considered high risk

---

## Architecture Overview
Git Repository
&darr;
C Dependency Extraction
&darr;
Dependency Graph
&darr;
Graph Metrics + History + Code Signals
&darr;
Risk Model
&darr;
Ranked Risk Report + Explanations


---

## Quick Start (Planned Interface)

```bash
tensile analyze path/to/repo
tensile report --top 20
tensile explain src/btree.c

Note: Commands are implemented incrementally. See docs/spec.md for the authoritative design specification.
