"""
REWARD PROPAGATOR -- Implements RewardFlow's two algorithms.

Algorithm 1: BFS Propagation
  Start from success nodes.
  Propagate reward outward level by level.
  Reward decays by BFS distance from success.

Algorithm 2: Personalized PageRank (simplified)
  Success nodes are "teleport" nodes.
  Random walk with restart -- nodes closer to success
  in graph topology get higher stationary probability.

Both produce DENSE reward signals -- every intermediate
node gets a reward, not just terminal nodes.

Compare to SPARSE baseline:
  Only terminal nodes get reward (+1 success, -1 failure).
  All intermediate nodes get reward = 0.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict
from core.state_graph import StateGraph

DECAY_FACTOR = 0.8     # reward decay per BFS hop
DAMPING = 0.85         # PageRank damping factor
PPR_ITERATIONS = 20    # PageRank iterations

class RewardPropagator:

    def propagate_bfs(self, graph: StateGraph) -> Dict[str, float]:
        """
        BFS reward propagation from success nodes.

        reward(node) = max over all success nodes s of:
            base_reward * decay^(BFS_distance(node, s))

        Penalize nodes connected to failure nodes.
        """
        rewards = {nid: 0.0 for nid in graph.nodes}
        adj = graph.get_adjacency()

        # Forward pass: propagate from success nodes
        for success_node in graph.success_nodes:
            visited = {success_node: 0}
            queue = deque([success_node])

            while queue:
                current = queue.popleft()
                distance = visited[current]
                reward = DECAY_FACTOR ** distance
                rewards[current] = max(rewards[current], reward)

                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = distance + 1
                        queue.append(neighbor)

        # Backward pass: penalize nodes near failure nodes
        for failure_node in graph.failure_nodes:
            visited = {failure_node: 0}
            queue = deque([failure_node])

            while queue:
                current = queue.popleft()
                distance = visited[current]
                penalty = -0.3 * (DECAY_FACTOR ** distance)
                rewards[current] = max(-1.0, rewards[current] + penalty)

                if distance < 2:  # only penalize close neighbors
                    for neighbor in adj.get(current, []):
                        if neighbor not in visited:
                            visited[neighbor] = distance + 1
                            queue.append(neighbor)

        # Apply visit count bonus (shared nodes are more reliable)
        for nid, node in graph.nodes.items():
            if node.visit_count > 1:
                rewards[nid] = min(1.0, rewards[nid] * (1 + 0.1 * (node.visit_count - 1)))

        return rewards

    def propagate_pagerank(self, graph: StateGraph) -> Dict[str, float]:
        """
        Personalized PageRank with success nodes as teleport targets.

        At each step, random walker either:
        - Follows a random edge (prob = damping)
        - Teleports to a success node (prob = 1 - damping)

        Nodes that are topologically close to success nodes
        get higher stationary probability = higher reward.
        """
        n_nodes = len(graph.nodes)
        if n_nodes == 0:
            return {}

        node_ids = list(graph.nodes.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = graph.get_adjacency()

        # Initialize uniform distribution
        ranks = {nid: 1.0 / n_nodes for nid in node_ids}

        # Teleport distribution: uniform over success nodes
        n_success = max(len(graph.success_nodes), 1)
        teleport = {
            nid: (1.0 / n_success if nid in graph.success_nodes else 0.0)
            for nid in node_ids
        }

        # Power iteration
        for _ in range(PPR_ITERATIONS):
            new_ranks = {}
            for nid in node_ids:
                # Contribution from random walk
                walk_contrib = 0.0
                for neighbor in adj.get(nid, []):
                    n_out = len(adj.get(neighbor, []))
                    if n_out > 0:
                        walk_contrib += ranks[neighbor] / n_out

                # New rank = damping * walk + (1-damping) * teleport
                new_ranks[nid] = (DAMPING * walk_contrib +
                                  (1 - DAMPING) * teleport[nid])

            ranks = new_ranks

        # Normalize to [-1, 1] range
        max_rank = max(ranks.values()) if ranks else 1.0
        min_rank = min(ranks.values()) if ranks else 0.0
        range_r = max_rank - min_rank

        normalized = {}
        for nid, rank in ranks.items():
            if range_r > 0:
                normalized[nid] = 2 * (rank - min_rank) / range_r - 1
            else:
                normalized[nid] = 0.0

        return normalized

    def sparse_baseline(self, graph: StateGraph) -> Dict[str, float]:
        """
        BASELINE: sparse terminal-only reward.
        Success terminal = +1
        Failure terminal = -1
        All intermediate nodes = 0

        This is what standard RL training uses.
        RewardFlow should outperform this.
        """
        rewards = {nid: 0.0 for nid in graph.nodes}
        for nid in graph.success_nodes:
            rewards[nid] = 1.0
        for nid in graph.failure_nodes:
            rewards[nid] = -1.0
        return rewards
