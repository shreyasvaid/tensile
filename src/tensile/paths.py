from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def cache_dir(self) -> Path:
        return self.root / "data" / "cache"

    @property
    def graph_json(self) -> Path:
        return self.cache_dir / "graph.json"

    @property
    def metrics_csv(self) -> Path:
        return self.cache_dir / "metrics.csv"
    
    @property
    def history_csv(self) -> Path:
        return self.cache_dir / "history.csv"


def artifact_paths(project_root: Path) -> ArtifactPaths:
    return ArtifactPaths(root=project_root.resolve())
