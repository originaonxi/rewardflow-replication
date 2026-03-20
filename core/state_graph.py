"""
STATE GRAPH BUILDER -- The core of RewardFlow.

Builds a directed graph where:
- Nodes = reasoning states (individual steps across rollouts)
- Edges = transitions between steps (what step came after what)
- Terminal nodes = final states (correct or incorrect)

The graph is built by aggregating ALL rollouts for a problem.
Steps that appear in multiple rollouts get merged into shared nodes.
This is what allows reward to flow across rollout boundaries.

State identity: we use semantic similarity of step text.
Two steps with the same mathematical operation are the same state.
"""

from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class StateNode:
    node_id: str
    step_text: str
    step_index: int        # position in reasoning chain
    is_terminal: bool = False
    is_success: bool = False
    rollout_ids: List[int] = field(default_factory=list)
    reward: float = 0.0    # filled by propagator
    visit_count: int = 0

@dataclass
class StateEdge:
    from_node: str
    to_node: str
    rollout_id: int
    weight: float = 1.0

class StateGraph:
    """
    Represents the aggregate state graph from multiple rollouts.
    Built from rollout data. Used by reward propagator.
    """

    def __init__(self):
        self.nodes: Dict[str, StateNode] = {}
        self.edges: List[StateEdge] = []
        self.success_nodes: List[str] = []
        self.failure_nodes: List[str] = []
        self.root_node: Optional[str] = None

    def build_from_rollouts(self, rollouts: list, problem: dict) -> None:
        """
        Build state graph from list of rollouts.
        Each rollout = {steps: [...], is_correct: bool, rollout_id: int}
        """
        # Create root node (the problem itself)
        root_id = "ROOT"
        self.nodes[root_id] = StateNode(
            node_id=root_id,
            step_text=problem["problem"][:80],
            step_index=-1,
            visit_count=len(rollouts),
        )
        self.root_node = root_id

        for rollout in rollouts:
            steps = rollout.get("steps", [])
            if not steps:
                continue

            prev_node_id = root_id

            for step_idx, step_text in enumerate(steps):
                # Create node ID based on step content + position
                # Steps at the same position with similar content = same node
                node_id = self._get_node_id(step_text, step_idx)

                if node_id not in self.nodes:
                    self.nodes[node_id] = StateNode(
                        node_id=node_id,
                        step_text=step_text[:120],
                        step_index=step_idx,
                        rollout_ids=[rollout["rollout_id"]],
                        visit_count=1,
                    )
                else:
                    # Node already exists -- this step appears in multiple rollouts
                    self.nodes[node_id].visit_count += 1
                    if rollout["rollout_id"] not in self.nodes[node_id].rollout_ids:
                        self.nodes[node_id].rollout_ids.append(rollout["rollout_id"])

                # Add edge
                edge = StateEdge(
                    from_node=prev_node_id,
                    to_node=node_id,
                    rollout_id=rollout["rollout_id"],
                )
                self.edges.append(edge)
                prev_node_id = node_id

            # Mark terminal node
            terminal_id = prev_node_id
            if terminal_id in self.nodes:
                self.nodes[terminal_id].is_terminal = True
                self.nodes[terminal_id].is_success = rollout["is_correct"]

                if rollout["is_correct"]:
                    if terminal_id not in self.success_nodes:
                        self.success_nodes.append(terminal_id)
                else:
                    if terminal_id not in self.failure_nodes:
                        self.failure_nodes.append(terminal_id)

        # Remove self-loops (as per paper)
        self.edges = [e for e in self.edges if e.from_node != e.to_node]

    def _get_node_id(self, step_text: str, step_idx: int) -> str:
        """
        Create a node ID that groups similar steps together.
        Steps at the same index with same key numbers = same node.
        """
        # Extract key numbers from step
        numbers = re.findall(r'\d+\.?\d*', step_text)
        key_numbers = ','.join(sorted(numbers[:4]))

        # Hash based on position + key content
        content = f"{step_idx}_{key_numbers}_{step_text[:30].lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_adjacency(self) -> Dict[str, List[str]]:
        """Get adjacency list (undirected, as per paper)."""
        adj = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            if edge.from_node in adj and edge.to_node in adj:
                if edge.to_node not in adj[edge.from_node]:
                    adj[edge.from_node].append(edge.to_node)
                if edge.from_node not in adj[edge.to_node]:
                    adj[edge.to_node].append(edge.from_node)
        return adj

    def summary(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "success_nodes": len(self.success_nodes),
            "failure_nodes": len(self.failure_nodes),
            "shared_nodes": sum(
                1 for n in self.nodes.values() if len(n.rollout_ids) > 1
            ),
        }
