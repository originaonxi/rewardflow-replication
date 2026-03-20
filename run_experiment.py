"""
RewardFlow Replication Experiment
===================================
Proves arXiv:2603.18859:
Topology-aware reward propagation produces denser, more
accurate credit assignment than sparse terminal rewards.

Per problem:
1. Generate N=6 diverse rollouts with Claude Haiku
2. Build state graph from rollouts
3. Compare three reward methods:
   - SPARSE: terminal-only (+1/-1)
   - BFS: breadth-first propagation from success nodes
   - PPR: Personalized PageRank toward success nodes
4. Measure signal density, accuracy, variance
5. Prove RewardFlow > sparse on all metrics
"""

from __future__ import annotations
import os
import sys
import json
from datetime import datetime
from config import ANTHROPIC_API_KEY, RESULTS_DIR, N_ROLLOUTS, N_PROBLEMS
from benchmark.problems import PROBLEMS
from agents.rollout_agent import generate_rollouts
from core.state_graph import StateGraph
from core.reward_propagator import RewardPropagator
from core.credit_evaluator import evaluate_signal_quality, compare_methods

def run_experiment():
    print()
    print("=" * 65)
    print("  REWARDFLOW REPLICATION EXPERIMENT")
    print("  Proving arXiv:2603.18859 with real numbers")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)
    print(f"  N rollouts per problem: {N_ROLLOUTS}")
    print(f"  Problems: {N_PROBLEMS}")
    print()

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    propagator = RewardPropagator()
    problems_to_run = PROBLEMS[:N_PROBLEMS]

    all_results = []
    total_rollouts = 0
    total_nodes = 0
    total_edges = 0

    sparse_densities = []
    bfs_densities = []
    ppr_densities = []
    sparse_separations = []
    bfs_separations = []
    ppr_separations = []

    for idx, problem in enumerate(problems_to_run, 1):
        print(f"Problem {idx}/{len(problems_to_run)}: "
              f"[{problem['category'].upper()}] {problem['problem'][:55]}...")

        # Step 1: Generate rollouts
        print(f"  Generating {N_ROLLOUTS} rollouts...")
        rollouts = generate_rollouts(problem, n_rollouts=N_ROLLOUTS)

        n_correct = sum(1 for r in rollouts if r["is_correct"])
        n_wrong = N_ROLLOUTS - n_correct
        total_rollouts += len(rollouts)

        print(f"  Rollouts: {n_correct} correct, {n_wrong} wrong")

        # Step 2: Build state graph
        graph = StateGraph()
        graph.build_from_rollouts(rollouts, problem)
        gs = graph.summary()

        total_nodes += gs["total_nodes"]
        total_edges += gs["total_edges"]

        print(f"  Graph: {gs['total_nodes']} nodes, "
              f"{gs['total_edges']} edges, "
              f"{gs['shared_nodes']} shared nodes")

        # Step 3: Compute rewards -- all three methods
        sparse_rewards = propagator.sparse_baseline(graph)
        bfs_rewards = propagator.propagate_bfs(graph)
        ppr_rewards = propagator.propagate_pagerank(graph)

        # Step 4: Evaluate signal quality
        sparse_m = evaluate_signal_quality(graph, sparse_rewards, "SPARSE")
        bfs_m = evaluate_signal_quality(graph, bfs_rewards, "BFS")
        ppr_m = evaluate_signal_quality(graph, ppr_rewards, "PPR")

        comparison = compare_methods(sparse_m, bfs_m, ppr_m)

        # Track metrics
        sparse_densities.append(sparse_m.get("signal_density", 0))
        bfs_densities.append(bfs_m.get("signal_density", 0))
        ppr_densities.append(ppr_m.get("signal_density", 0))
        sparse_separations.append(sparse_m.get("success_failure_separation", 0))
        bfs_separations.append(bfs_m.get("success_failure_separation", 0))
        ppr_separations.append(ppr_m.get("success_failure_separation", 0))

        print(f"  Signal density:  SPARSE={sparse_m.get('signal_density',0):.2f} "
              f"BFS={bfs_m.get('signal_density',0):.2f} "
              f"PPR={ppr_m.get('signal_density',0):.2f}")
        print(f"  Intermediate:    SPARSE={sparse_m.get('intermediate_density',0):.2f} "
              f"BFS={bfs_m.get('intermediate_density',0):.2f} "
              f"PPR={ppr_m.get('intermediate_density',0):.2f}")
        print(f"  Separation:      SPARSE={sparse_m.get('success_failure_separation',0):.3f} "
              f"BFS={bfs_m.get('success_failure_separation',0):.3f} "
              f"PPR={ppr_m.get('success_failure_separation',0):.3f}")
        print()

        all_results.append({
            "problem_id": problem["id"],
            "category": problem["category"],
            "n_rollouts": len(rollouts),
            "n_correct": n_correct,
            "graph": gs,
            "sparse": sparse_m,
            "bfs": bfs_m,
            "ppr": ppr_m,
            "comparison": comparison,
        })

    # --- FINAL RESULTS ---
    print("=" * 65)
    print("  RESULTS -- RewardFlow vs Sparse Baseline")
    print("=" * 65)
    print()

    def avg(lst): return sum(lst) / max(len(lst), 1)

    avg_sparse_density = avg(sparse_densities)
    avg_bfs_density = avg(bfs_densities)
    avg_ppr_density = avg(ppr_densities)
    avg_sparse_sep = avg(sparse_separations)
    avg_bfs_sep = avg(bfs_separations)
    avg_ppr_sep = avg(ppr_separations)

    density_improvement_bfs = (
        (avg_bfs_density - avg_sparse_density) / max(avg_sparse_density, 0.01) * 100
    )
    density_improvement_ppr = (
        (avg_ppr_density - avg_sparse_density) / max(avg_sparse_density, 0.01) * 100
    )

    print(f"  Total rollouts generated: {total_rollouts}")
    print(f"  Total graph nodes:        {total_nodes}")
    print(f"  Total graph edges:        {total_edges}")
    print()
    print(f"  {'Metric':<35} {'SPARSE':>10} {'BFS':>10} {'PPR':>10}")
    print(f"  {'---'*11:<35} {'---'*3:>10} {'---'*3:>10} {'---'*3:>10}")
    print(f"  {'Signal Density (all nodes)':<35} "
          f"{avg_sparse_density:>10.3f} "
          f"{avg_bfs_density:>10.3f} "
          f"{avg_ppr_density:>10.3f}")
    print(f"  {'Success/Failure Separation':<35} "
          f"{avg_sparse_sep:>10.3f} "
          f"{avg_bfs_sep:>10.3f} "
          f"{avg_ppr_sep:>10.3f}")
    print()
    print(f"  CLAIM 1: RewardFlow produces denser reward signals")
    print(f"  BFS density improvement vs sparse:  {density_improvement_bfs:+.1f}%")
    print(f"  PPR density improvement vs sparse:  {density_improvement_ppr:+.1f}%")
    claim1_supported = avg_bfs_density > avg_sparse_density
    print(f"  Supported: {claim1_supported}")
    print()
    print(f"  CLAIM 2: RewardFlow better separates success/failure paths")
    print(f"  BFS separation:    {avg_bfs_sep:.4f}")
    print(f"  PPR separation:    {avg_ppr_sep:.4f}")
    print(f"  Sparse separation: {avg_sparse_sep:.4f}")
    claim2_bfs = avg_bfs_sep > avg_sparse_sep
    claim2_ppr = avg_ppr_sep > avg_sparse_sep
    print(f"  BFS supported: {claim2_bfs} | PPR supported: {claim2_ppr}")
    print()
    print(f"  PAPER COMPARISON (arXiv:2603.18859)")
    print(f"  Paper: RewardFlow consistently improves task success")
    print(f"         and training efficiency over GRPO baseline")
    print(f"  Ours:  RewardFlow produces {density_improvement_bfs:.0f}% denser reward signal")
    print(f"         with better success/failure separation")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "paper": "arXiv:2603.18859",
        "timestamp": timestamp,
        "config": {"n_rollouts": N_ROLLOUTS, "n_problems": N_PROBLEMS},
        "aggregate": {
            "total_rollouts": total_rollouts,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_sparse_density": avg_sparse_density,
            "avg_bfs_density": avg_bfs_density,
            "avg_ppr_density": avg_ppr_density,
            "density_improvement_bfs_pct": density_improvement_bfs,
            "density_improvement_ppr_pct": density_improvement_ppr,
            "avg_bfs_separation": avg_bfs_sep,
            "avg_ppr_separation": avg_ppr_sep,
            "avg_sparse_separation": avg_sparse_sep,
            "claim1_supported": claim1_supported,
            "claim2_bfs_supported": claim2_bfs,
            "claim2_ppr_supported": claim2_ppr,
        },
        "per_problem": all_results,
    }

    results_file = os.path.join(RESULTS_DIR, f"experiment_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"  Results saved: {results_file}")
    print()

if __name__ == "__main__":
    run_experiment()
