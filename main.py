"""
main.py
Full analysis pipeline for the Facebook Ego-Network project.

Phases
------
1. Data cleaning & overview
2. Graph construction & global metrics
3. Ego network analysis
4. Synthetic network comparison (BA vs ER vs Real)
5. Information diffusion simulation
6. Visualisations (all saved to figures - Modularity curves, Power Law plots, degree distributions, ego network plots, diffusion spread)

Run
---
    python main.py
"""
import os
import sys
import random

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import networkx as nx

from src.config            import (DATA_RAW, DATA_PROCESSED, FIGURES_PATH, EGO_NODE, BA_M, RANDOM_SEED, DIFFUSION_P)
from src.cleaning          import clean_facebook_dataset
from src.graph_builder     import build_facebook_graph, extract_ego_network
from src.metrics           import (global_summary, degree_distribution, top_nodes_by_centrality, detect_communities_greedy)
from src.ego_analysis      import top_ego_candidates, compare_ego_networks
from src.synthetic_networks import (generate_ba_network, generate_er_network, save_synthetic_edges)
from src.diffusion         import independent_cascade
from src.visualization     import (plot_degree_distribution, plot_degree_distribution_loglog, plot_synthetic_comparison, plot_ego_network, plot_diffusion_spread, plot_ego_comparison_heatmap, plot_top_centrality, plot_community_size_powerlaw, plot_modularity_curve)


def phase1_clean() -> pd.DataFrame:
    """Load and clean the raw edge list."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — Data Cleaning & Overview")
    print("=" * 60)
    edges = clean_facebook_dataset(raw_path=DATA_RAW, save=True)
    print(f"  Edges : {len(edges):,}  |  "f"Unique nodes : {pd.concat([edges['node1'], edges['node2']]).nunique():,}")

    # Save degree frequency table
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    deg_freq = (
        pd.concat([edges["node1"], edges["node2"]])
        .value_counts()
        .rename_axis("Node")
        .reset_index(name="Degree")
    )
    deg_freq.to_csv(os.path.join(DATA_PROCESSED, "degree_frequency.csv"), index=False)
    print(f"  Top 5 hubs: {deg_freq.head()['Node'].tolist()}")
    print("  Phase 1 complete.\n")
    return edges

def phase2_graph_metrics(edges: pd.DataFrame) -> nx.Graph:
    """Build graph and compute global metrics."""
    print("=" * 60)
    print("  PHASE 2 — Graph Construction & Global Metrics")
    print("=" * 60)

    G = build_facebook_graph(edges)
    summary = global_summary(G)

    print("\n  Global Network Summary")
    for k, v in summary.items():
        print(f"    {k:<22}: {v}")

    # Degree distribution
    dd = degree_distribution(G)
    dd.to_csv(os.path.join(DATA_PROCESSED, "degree_distribution.csv"), index=False)

    # Top centrality nodes
    print("\n  Top-10 Nodes by Centrality")
    top = top_nodes_by_centrality(G, k=10)
    print(top.to_string(index=False))
    top.to_csv(os.path.join(DATA_PROCESSED, "top_centrality_nodes.csv"), index=False)

    # Community detection
    print("\n  Community Detection (greedy modularity)...")
    community_map = detect_communities_greedy(G)
    comm_df = pd.DataFrame(community_map.items(), columns=["node", "community"])
    comm_df.to_csv(os.path.join(DATA_PROCESSED, "communities.csv"), index=False)

    # Figures
    plot_degree_distribution(G, filename="degree_distribution.png")
    plot_degree_distribution_loglog(G, filename="degree_dist_loglog.png")
    plot_community_size_powerlaw(community_map, filename="community_size_powerlaw.png")
    plot_modularity_curve(G, filename="modularity_curve.png")

    print("  Phase 2 complete.\n")
    return G

def phase3_ego_analysis(G: nx.Graph) -> pd.DataFrame:
    """Analyse top ego networks."""
    print("=" * 60)
    print("  PHASE 3 — Ego Network Analysis")
    print("=" * 60)

    ego_nodes = top_ego_candidates(G, k=10)
    print(f"  Ego centres (top-10 by degree): {ego_nodes}")

    ego_df = compare_ego_networks(G, ego_nodes, radius=1)
    print("\n  Ego Network Comparison Table")
    display_cols = ["ego_node", "n_nodes", "n_edges", "density",
                    "avg_clustering", "ego_degree", "effective_size"]
    print(ego_df[display_cols].to_string(index=False))
    ego_df.to_csv(os.path.join(DATA_PROCESSED, "ego_analysis.csv"), index=False)

    # Plot ego network for the highest-degree node
    top_ego = ego_nodes[0]
    ego_net = extract_ego_network(G, ego_node=top_ego, radius=1)
    plot_ego_network(ego_net, ego_node=top_ego, filename=f"ego_network_node{top_ego}.png")
    plot_ego_comparison_heatmap(ego_df, filename="ego_heatmap.png")

    print("  Phase 3 complete.\n")
    return ego_df

def phase4_synthetic_comparison(G: nx.Graph) -> None:
    """Generate BA and ER synthetic graphs; compare degree distributions."""
    print("=" * 60)
    print("  PHASE 4 — Synthetic Network Comparison")
    print("=" * 60)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    p_er = (2 * m) / (n * (n - 1))   # expected density matching real graph

    print(f"  Real graph   : n={n:,}, m={m:,}")
    print(f"  Generating BA: n={n:,}, m={BA_M} per node")
    print(f"  Generating ER : n={n:,}, p={p_er:.6f}")

    G_ba = generate_ba_network(n, BA_M, seed=RANDOM_SEED)
    G_er = generate_er_network(n, p_er, seed=RANDOM_SEED)

    # Save synthetic edge lists
    save_synthetic_edges(G_ba, os.path.join(DATA_PROCESSED, "edges_ba.csv"))
    save_synthetic_edges(G_er, os.path.join(DATA_PROCESSED, "edges_er.csv"))

    plot_synthetic_comparison(G, G_ba, G_er, filename="synthetic_comparison.png")

    # Summary comparison
    for label, Gx in [("Real", G), ("BA", G_ba), ("ER", G_er)]:
        s = global_summary(Gx)
        print(f"  {label:<6} | nodes={s['nodes']:,} | edges={s['edges']:,} | "f"avg_deg={s['avg_degree']:.2f} | clustering={s['clustering']:.4f}")

    print("  Phase 4 complete.\n")

def phase5_diffusion(G: nx.Graph) -> None:
    """Run independent cascade diffusion from top hubs."""
    print("=" * 60)
    print("  PHASE 5 — Information Diffusion (Independent Cascade)")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    seed_nodes = top_ego_candidates(G, k=3)   # top-3 hubs as seeds
    print(f"  Seed nodes : {seed_nodes}")
    print(f"  Propagation probability p = {DIFFUSION_P}")

    # Track spread step-by-step
    spread_by_step = [len(seed_nodes)]
    activated      = set(seed_nodes)
    newly_activated = set(seed_nodes)

    for step in range(100):
        if not newly_activated:
            break
        current_new = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated and random.random() < DIFFUSION_P:
                    current_new.add(neighbor)
        newly_activated = current_new
        activated.update(current_new)
        spread_by_step.append(len(activated))

    final_fraction = len(activated) / G.number_of_nodes()
    print(f"  Steps to termination : {len(spread_by_step) - 1}")
    print(f"  Final reach          : {len(activated):,} / {G.number_of_nodes():,} "f"nodes ({final_fraction:.1%})")

    plot_diffusion_spread(spread_by_step, G.number_of_nodes(), filename="diffusion_spread.png")

    # Save results
    diff_df = pd.DataFrame({"step": range(len(spread_by_step)), "activated": spread_by_step, "fraction":  [s / G.number_of_nodes() for s in spread_by_step]})
    diff_df.to_csv(os.path.join(DATA_PROCESSED, "diffusion_results.csv"), index=False)

    print("  Phase 5 complete.\n")

def main():
    print("\n" + "#" * 60)
    print("  Facebook Ego-Network Analysis Pipeline")
    print("#" * 60)

    edges  = phase1_clean()
    G      = phase2_graph_metrics(edges)
    ego_df = phase3_ego_analysis(G)
    phase4_synthetic_comparison(G)
    phase5_diffusion(G)

    print("\n" + "#" * 60)
    print(f"  All figures saved to : {FIGURES_PATH}")
    print(f"  All CSVs saved to    : {DATA_PROCESSED}")
    print("  Pipeline complete.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
