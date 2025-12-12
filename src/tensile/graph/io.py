from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

EdgeType = Literal["include"]

@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    type: EdgeType

def write_graph_json(
    out_path: Path,
    nodes: List[str],
    edges: List[GraphEdge],
    meta: Optional[Dict] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "0.1",
        "node_type": "file",
        "edge_types": ["include"],
        "nodes": [{"id": n} for n in sorted(set(nodes))],
        "edges": [{"src": e.src, "dst": e.dst, "type": e.type} for e in edges],
        "meta": meta or {},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def read_graph_json(path: Path) -> Tuple[List[str], List[GraphEdge], Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = [n["id"] for n in payload.get("nodes", [])]
    edges = [
        GraphEdge(src=e["src"], dst=e["dst"], type=e["type"]) for e in payload.get("edges", [])
    ]
    meta = payload.get("meta", {})
    return nodes, edges, meta
