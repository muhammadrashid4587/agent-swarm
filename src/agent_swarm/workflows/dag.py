"""DAG-based workflow engine for agent execution pipelines."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowNode:
    id: str
    task: str
    agent_role: str = "coder"
    config: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    status: str = "pending"


class DAGWorkflow:
    """Directed Acyclic Graph workflow engine for orchestrating agent pipelines."""

    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: dict[str, WorkflowNode] = {}
        self._edges: dict[str, list[str]] = defaultdict(list)
        self._reverse_edges: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: WorkflowNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Both nodes must exist: {from_id} -> {to_id}")
        self._edges[from_id].append(to_id)
        self._reverse_edges[to_id].append(from_id)
        if self._has_cycle():
            self._edges[from_id].remove(to_id)
            self._reverse_edges[to_id].remove(from_id)
            raise ValueError(f"Adding edge {from_id} -> {to_id} would create a cycle")

    def _has_cycle(self) -> bool:
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for targets in self._edges.values():
            for t in targets:
                in_degree[t] += 1

        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        count = 0
        while queue:
            node = queue.popleft()
            count += 1
            for neighbor in self._edges.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return count != len(self.nodes)

    def topological_sort(self) -> list[list[str]]:
        """Return nodes in topological order, grouped into parallel batches."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for targets in self._edges.values():
            for t in targets:
                in_degree[t] += 1

        batches: list[list[str]] = []
        queue = [nid for nid, d in in_degree.items() if d == 0]

        while queue:
            batches.append(sorted(queue))
            next_queue: list[str] = []
            for node in queue:
                for neighbor in self._edges.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue

        return batches

    def get_dependencies(self, node_id: str) -> list[str]:
        return self._reverse_edges.get(node_id, [])

    def get_dependents(self, node_id: str) -> list[str]:
        return self._edges.get(node_id, [])

    def visualize(self) -> str:
        """Generate an ASCII representation of the DAG."""
        lines = [f"DAG: {self.name}", "=" * 40]
        for batch_idx, batch in enumerate(self.topological_sort()):
            lines.append(f"\nBatch {batch_idx} (parallel):")
            for node_id in batch:
                node = self.nodes[node_id]
                deps = self.get_dependencies(node_id)
                dep_str = f" <- [{', '.join(deps)}]" if deps else ""
                lines.append(f"  [{node.agent_role}] {node_id}: {node.task[:60]}{dep_str}")
        return "\n".join(lines)
