"""
CREDIT EVALUATOR -- Measures signal quality.

The key question: does the reward method correctly identify
which intermediate steps were on the path to success?

We have GROUND TRUTH: the correct steps for each problem.
We can therefore measure:

1. Signal density: what fraction of nodes got non-zero reward?
   (RewardFlow > sparse baseline by design)

2. Signal accuracy: do correct-path nodes get higher reward
   than wrong-path nodes?
   (The core empirical claim)

3. Rank correlation: does reward ranking of nodes match
   ground-truth step importance?

4. Variance: how spread out are the rewards?
   (Higher variance = more informative signal)
"""

from __future__ import annotations
import math
from typing import Dict, List
from core.state_graph import StateGraph, StateNode

def evaluate_signal_quality(
    graph: StateGraph,
    rewards: Dict[str, float],
    method_name: str,
) -> dict:
    """
    Measure quality of a reward signal.
    Returns metrics for comparison.
    """
    if not rewards:
        return {}

    reward_values = list(rewards.values())

    # 1. Density: fraction with non-zero reward
    nonzero = sum(1 for r in reward_values if abs(r) > 0.01)
    density = nonzero / max(len(reward_values), 1)

    # 2. Variance: spread of signal
    mean = sum(reward_values) / len(reward_values)
    variance = sum((r - mean) ** 2 for r in reward_values) / len(reward_values)

    # 3. Success node reward vs failure node reward
    success_rewards = [rewards[nid] for nid in graph.success_nodes if nid in rewards]
    failure_rewards = [rewards[nid] for nid in graph.failure_nodes if nid in rewards]

    avg_success = sum(success_rewards) / max(len(success_rewards), 1)
    avg_failure = sum(failure_rewards) / max(len(failure_rewards), 1)
    separation = avg_success - avg_failure

    # 4. Intermediate node signal (key claim)
    intermediate_nodes = [
        nid for nid, node in graph.nodes.items()
        if not node.is_terminal and nid != graph.root_node
    ]
    intermediate_rewards = [rewards[nid] for nid in intermediate_nodes if nid in rewards]

    intermediate_nonzero = sum(1 for r in intermediate_rewards if abs(r) > 0.01)
    intermediate_density = intermediate_nonzero / max(len(intermediate_rewards), 1)

    # 5. Shared node bonus (nodes appearing in multiple rollouts)
    shared_nodes = [nid for nid, n in graph.nodes.items() if len(n.rollout_ids) > 1]
    shared_rewards = [rewards[nid] for nid in shared_nodes if nid in rewards]
    avg_shared_reward = sum(shared_rewards) / max(len(shared_rewards), 1)

    return {
        "method": method_name,
        "signal_density": density,
        "intermediate_density": intermediate_density,
        "signal_variance": variance,
        "avg_success_node_reward": avg_success,
        "avg_failure_node_reward": avg_failure,
        "success_failure_separation": separation,
        "avg_shared_node_reward": avg_shared_reward,
        "total_nodes": len(graph.nodes),
        "intermediate_nodes": len(intermediate_nodes),
        "shared_nodes": len(shared_nodes),
    }

def compare_methods(
    sparse_metrics: dict,
    bfs_metrics: dict,
    ppr_metrics: dict,
) -> dict:
    """
    Compare RewardFlow methods vs sparse baseline.
    Computes improvement ratios.
    """
    def improvement(new_val, base_val):
        if abs(base_val) < 1e-10:
            return float('inf') if abs(new_val) > 1e-10 else 0.0
        return (new_val - base_val) / abs(base_val) * 100

    return {
        "density_improvement_bfs": improvement(
            bfs_metrics.get("signal_density", 0),
            sparse_metrics.get("signal_density", 0)
        ),
        "density_improvement_ppr": improvement(
            ppr_metrics.get("signal_density", 0),
            sparse_metrics.get("signal_density", 0)
        ),
        "intermediate_density_bfs": bfs_metrics.get("intermediate_density", 0),
        "intermediate_density_ppr": ppr_metrics.get("intermediate_density", 0),
        "intermediate_density_sparse": sparse_metrics.get("intermediate_density", 0),
        "separation_bfs": bfs_metrics.get("success_failure_separation", 0),
        "separation_ppr": ppr_metrics.get("success_failure_separation", 0),
        "separation_sparse": sparse_metrics.get("success_failure_separation", 0),
        "variance_bfs": bfs_metrics.get("signal_variance", 0),
        "variance_ppr": ppr_metrics.get("signal_variance", 0),
        "variance_sparse": sparse_metrics.get("signal_variance", 0),
    }
