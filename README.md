# TENSILE

**TENSILE** is a static and historical risk analysis tool for large **C codebases**.  
It identifies files that are *most likely to require bug fixes in the near future* by combining:

- Dependency graph structure  
- Git history dynamics  
- Code-level complexity signals  
- Supervised machine learning  

TENSILE is designed for **maintainers, reviewers, and engineers** working in large, legacy, or safety-critical C projects.

---

## What TENSILE Does

Given a C repository and a point in time, TENSILE:

1. Builds a file-level dependency graph (currently include-based)
2. Extracts historical change and ownership signals from Git
3. Computes static code features (pointer density, memory usage, etc.)
4. Trains a supervised risk model using future bug-fix commits as labels
5. Ranks files by predicted risk
6. Explains *why* a file is risky

The output is a ranked list of files with:
- A risk score
- Supporting metrics
- Optional Markdown reports
- Per-file feature explanations

---

## Why This Is Important

Large C codebases fail in **predictable places**:

- Heavily coupled headers  
- Frequently modified core files  
- Pointer-dense, memory-heavy modules  
- Code owned by many contributors  

Yet most review processes treat all files equally.

TENSILE helps teams:
- Focus review effort where it matters most
- Prioritize refactors and audits
- Reduce regressions
- Surface “hidden risk” in legacy systems

This is **risk triage**, not bug detection.

---

## Installation

### Prerequisites

- Python **3.10+**
- Git (for history extraction)
- A C codebase under Git version control
- macOS or Linux (Windows not tested)

### Clone and Install

```bash
git clone https://github.com/<your-username>/tensile.git
cd tensile
```

Install dependencies using Poetry:
```bash
pip install poetry
poetry install
```

Verify installation:
```bash
poetry run tensile --help
```

### QUICK START:
Analyze a C repository in one command:
```bash
poetry run tensile analyze /path/to/repo --asof 2025-12-01
```
This will: 
- Build dependency graphs
- Extract Git history
- Compute static features
- Train a risk model
- Generate a risk report

Results are cached automatically for fast re-runs.

### Core Commands
analyze (Recommended Entry Point): Runs the full pipeline with caching.
```bash
poetry run tensile analyze /path/to/repo \
  --asof 2025-12-01 \
  --horizon-days 180 \
  --top 20 \
  --out-md
```
Options:
- --asof – analysis cutoff date (required)
- --horizon-days – future window for labeling (default: 180)
- --top – number of risky files to report
- --out-md – write data/cache/report.md
- --force – recompute all artifacts (ignore cache)

report: Print the top-K risky files using cached artifacts.
```bash
poetry run tensile report --top 20 --out-md
```

explain: Explain why a specific file is risky.
```bash
poetry run tensile explain src/alter.c
```

### Advanced / Pipeline Commands
You typically don’t need these, but they are available:
- build-graph
- extract-history
- build-features
- label
- join-labels
- train
- evaluate-model
These are useful for debugging or research workflows.

### Output Artifacts
All outputs are written to data/cache/

Key files:
- graph.json – file dependency graph
- metrics.csv – graph metrics
- history.csv – Git history features
- dataset.csv – joined feature table
- dataset_labeled.csv – supervised dataset
- model.joblib – trained model
- eval.json – evaluation metrics
- report.md – Markdown risk report

These artifacts are intentionally cached and should not be committed.

### Common Pitfalls and Nuances

# Cache Reuse
If you change --asof or --horizon-days, use:

```bash
--force
```

Otherwise cached artifacts will be reused.

# Git History Cost
Initial runs may take several minutes on large repositories.
Subsequent runs are fast (≈1–2 seconds) due to caching.

# Label Noise

Bug-fix labels are inferred from commit messages.
They are heuristic, not perfect ground truth.

TENSILE measures relative risk, not certainty.

### Model Interpretation

High risk does not mean bad code. Often it indicates:
- Central dependency position
- High change frequency
- Complex ownership or interface role


