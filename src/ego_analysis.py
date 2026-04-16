from itertools import combinations

import networkx as nx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def analyze_ego_network(G: nx.Graph, ego_node: int, radius: int = 1) -> dict:
    if ego_node not in G:
        raise ValueError(f"Node {ego_node} not found in graph.")

    ego_net = nx.ego_graph(G, ego_node, radius=radius)

    n = ego_net.number_of_nodes()
    m = ego_net.number_of_edges()

    betweenness = nx.betweenness_centrality(ego_net)
    closeness   = nx.closeness_centrality(ego_net)

    ego_degree    = int(G.degree(ego_node))
    neighbors     = list(ego_net.neighbors(ego_node))
    ego_triangles = sum(1 for u, v in combinations(neighbors, 2) if ego_net.has_edge(u, v))
    ego_clustering = (
        (2 * ego_triangles) / (ego_degree * (ego_degree - 1))
        if ego_degree > 1 else 0.0
    )

    alters = [node for node in ego_net.nodes() if node != ego_node]
    alter_subgraph = ego_net.subgraph(alters)
    redundancy = (
        sum(d for _, d in alter_subgraph.degree()) / len(alters)
        if alters else 0
    )
    effective_size = len(alters) - redundancy

    return {
        "ego_node":        ego_node,
        "radius":          radius,
        "n_nodes":         n,
        "n_edges":         m,
        "density":         round(nx.density(ego_net), 6),
        "avg_clustering":  round(nx.average_clustering(ego_net), 6),
        "ego_clustering":  round(ego_clustering, 6),
        "transitivity":    round(nx.transitivity(ego_net), 6),
        "ego_degree":      ego_degree,
        "ego_betweenness": round(betweenness.get(ego_node, 0), 6),
        "ego_closeness":   round(closeness.get(ego_node, 0), 6),
        "n_triangles":     ego_triangles,
        "effective_size":  round(effective_size, 4),
        "efficiency":      round(effective_size / len(alters), 4) if alters else 0,
    }


def top_ego_candidates(G: nx.Graph, k: int = 10) -> list[int]:
    return sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:k]


def compare_ego_networks(G: nx.Graph, ego_nodes: list[int], radius: int = 1) -> pd.DataFrame:
    def _analyze_safe(node):
        try:
            return analyze_ego_network(G, node, radius=radius)
        except ValueError as exc:
            print(f"  Skipping node {node}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=min(len(ego_nodes), 4)) as pool:
        results = list(pool.map(_analyze_safe, ego_nodes))

    records = [r for r in results if r is not None]
    return pd.DataFrame(records)


def run(G: nx.Graph, ego_nodes: list[int] | None = None, radius: int = 1) -> pd.DataFrame:
    print()
    print("=" * 50)
    print("  Phase 3 — Ego Network Analysis")
    print("=" * 50)

    if ego_nodes is None:
        ego_nodes = top_ego_candidates(G, k=10)
        print(f"\n  Ego centres (top-10 by degree): {ego_nodes}")

    df = compare_ego_networks(G, ego_nodes, radius=radius)

    display_cols = ["ego_node", "n_nodes", "n_edges", "density",
                    "avg_clustering", "ego_degree", "effective_size", "efficiency"]
    print(f"\n  Ego Network Comparison (radius={radius})")
    print(df[display_cols].to_string(index=False))

    print()
    print("  Phase 3 complete.")
    print()

    return df
