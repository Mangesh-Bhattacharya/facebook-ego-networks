import logging
import random
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, FIGURES_PATH, BA_M, RANDOM_SEED, DIFFUSION_P
from src.cleaning import clean_facebook_dataset
from src.graph_builder import build_facebook_graph, extract_ego_network
from src.metrics import global_summary, top_nodes_by_centrality, detect_communities_label_propagation
from src.ego_analysis import top_ego_candidates, compare_ego_networks
from src.synthetic_networks import generate_ba_network, generate_er_network
from src.visualization import (
    plot_degree_distribution, plot_degree_distribution_loglog,
    plot_synthetic_comparison, plot_diffusion_spread,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_preprocessing():
    logger.info("Initializing Data Cleaning & Overview...")

    if not Path(DATA_RAW).exists():
        logger.error(f"Raw data not found at {DATA_RAW}. Please check your config.")
        return None

    edges = clean_facebook_dataset(raw_path=DATA_RAW, save=True)
    n_nodes = pd.concat([edges['node1'], edges['node2']]).nunique()
    logger.info(f"Dataset loaded: {len(edges):,} edges across {n_nodes:,} unique nodes.")

    Path(DATA_PROCESSED).mkdir(parents=True, exist_ok=True)
    deg_freq = (
        pd.concat([edges["node1"], edges["node2"]])
        .value_counts()
        .rename_axis("Node")
        .reset_index(name="Degree")
    )
    deg_freq.to_csv(Path(DATA_PROCESSED) / "degree_frequency.csv", index=False)
    return edges


def run_structural_analysis(edges):
    logger.info("Constructing graph and calculating global metrics...")

    G = build_facebook_graph(edges)
    summary = global_summary(G)

    for metric, value in summary.items():
        logger.info(f"{metric.replace('_', ' ').title()}: {value}")

    top_centrality = top_nodes_by_centrality(G, k=10)
    top_centrality.to_csv(Path(DATA_PROCESSED) / "top_centrality_nodes.csv", index=False)

    partition = detect_communities_label_propagation(G)
    pd.DataFrame(partition.items(), columns=["node", "community"]).to_csv(
        Path(DATA_PROCESSED) / "communities.csv", index=False
    )

    plot_degree_distribution(G, filename="degree_distribution.png")
    plot_degree_distribution_loglog(G, filename="degree_dist_loglog.png")

    return G


def run_synthetic_benchmarking(G):
    logger.info("Generating synthetic benchmarks (Barabasi-Albert & Erdos-Renyi)...")

    n, m = G.number_of_nodes(), G.number_of_edges()
    p_er = (2 * m) / (n * (n - 1))

    G_ba = generate_ba_network(n, BA_M, seed=RANDOM_SEED)
    G_er = generate_er_network(n, p_er, seed=RANDOM_SEED)

    plot_synthetic_comparison(G, G_ba, G_er, filename="synthetic_comparison.png")

    for label, model in [("Real", G), ("BA", G_ba), ("ER", G_er)]:
        clust = nx.average_clustering(model)
        logger.info(f"[{label}] Clustering Coeff: {clust:.4f}")


def run_diffusion_sim(G):
    logger.info("Starting Information Diffusion simulation...")

    random.seed(RANDOM_SEED)
    seeds = top_ego_candidates(G, k=3)
    activated = set(seeds)
    newly_activated = set(seeds)
    history = [len(activated)]

    while newly_activated:
        next_round = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated and random.random() < DIFFUSION_P:
                    next_round.add(neighbor)
        newly_activated = next_round
        activated.update(next_round)
        history.append(len(activated))

    reach = len(activated) / G.number_of_nodes()
    logger.info(f"Simulation ended. Final reach: {reach:.1%} ({len(activated)} nodes).")
    plot_diffusion_spread(history, G.number_of_nodes(), filename="diffusion_spread.png")


def main():
    print("\n--- Facebook Ego-Network Research Pipeline ---\n")

    edges = run_preprocessing()
    if edges is not None:
        G = run_structural_analysis(edges)
        compare_ego_networks(G, top_ego_candidates(G, k=10), radius=1)
        run_synthetic_benchmarking(G)
        run_diffusion_sim(G)
        print(f"\nPipeline complete. Results available in {FIGURES_PATH} and {DATA_PROCESSED}.\n")


if __name__ == "__main__":
    main()
