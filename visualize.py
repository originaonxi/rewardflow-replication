"""Print final results from saved JSON."""
import json, glob, os

def load_latest():
    files = glob.glob("results/experiment_*.json")
    if not files: return None
    with open(max(files)) as f: return json.load(f)

def print_table(data):
    print()
    print("=" * 65)
    print("  REWARDFLOW REPLICATION -- RESULTS")
    print("=" * 65)
    agg = data["aggregate"]
    print(f"\n  Total rollouts:  {agg['total_rollouts']}")
    print(f"  Total nodes:     {agg['total_nodes']}")
    print(f"  Total edges:     {agg['total_edges']}")
    print(f"\n  Signal Density:  SPARSE={agg['avg_sparse_density']:.3f}  "
          f"BFS={agg['avg_bfs_density']:.3f}  PPR={agg['avg_ppr_density']:.3f}")
    print(f"  Separation:      SPARSE={agg['avg_sparse_separation']:.4f}  "
          f"BFS={agg['avg_bfs_separation']:.4f}  PPR={agg['avg_ppr_separation']:.4f}")
    print(f"\n  BFS density improvement: {agg['density_improvement_bfs_pct']:+.1f}%")
    print(f"  PPR density improvement: {agg['density_improvement_ppr_pct']:+.1f}%")
    print(f"\n  Claim 1 (denser signal): {agg['claim1_supported']}")
    print(f"  Claim 2 BFS (better separation): {agg['claim2_bfs_supported']}")
    print(f"  Claim 2 PPR (better separation): {agg['claim2_ppr_supported']}")
    print()

if __name__ == "__main__":
    data = load_latest()
    if data: print_table(data)
