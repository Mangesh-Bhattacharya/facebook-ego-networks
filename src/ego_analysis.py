"""
ego_analysis.py
Ego-network extraction and analysis.

An ego network for node v at radius r contains v (the ego), all nodes
reachable within r hops (alters), and all edges among them.
"""
import networkx as nx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def analyze_ego_network(G: nx.Graph, ego_node: int,
                        radius: int = 1) -> dict:
    """
    Compute a comprehensive profile of a single ego network.

    Parameters
    ----------
    G        : The full Facebook graph.
    ego_node : Central node (ego).
    radius   : Hop distance for neighbourhood inclusion.

    Returns
    -------
    dict with keys:
        ego_node, radius, n_nodes, n_edges, density,
        avg_clustering, ego_clustering, transitivity,
        ego_degree, ego_betweenness, ego_closeness,
        n_triangles, effective_size, efficiency
    """
    if ego_node not in G:
        raise ValueError(f"Node {ego_node} not found in graph.")

    ego_net = nx.ego_graph(G, ego_node, radius=radius)

    n = ego_net.number_of_nodes()
    m = ego_net.number_of_edges()

    # Structural holes (Burt 1992) — only meaningful when ego_net is connected
    betweenness = nx.betweenness_centrality(ego_net)
    closeness   = nx.closeness_centrality(ego_net)

    # Effective size = n_alters - avg redundancy of alters
    alters = list(ego_net.nodes())
    alters.remove(ego_node)
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
        "ego_clustering":  round(nx.clustering(ego_net, ego_node), 6),
        "transitivity":    round(nx.transitivity(ego_net), 6),
        "ego_degree":      G.degree(ego_node),
        "ego_betweenness": round(betweenness.get(ego_node, 0), 6),
        "ego_closeness":   round(closeness.get(ego_node, 0), 6),
        "n_triangles":     sum(nx.triangles(ego_net, ego_node)
                               for ego_node in [ego_node]) // 1,
        "effective_size":  round(effective_size, 4),
        "efficiency":      round(effective_size / len(alters), 4) if alters else 0,
    }


def top_ego_candidates(G: nx.Graph, k: int = 10) -> list[int]:
    """
    Return the k nodes with the highest degree — natural ego-network centres.
    """
    return sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:k]


def compare_ego_networks(G: nx.Graph,
                         ego_nodes: list[int],
                         radius: int = 1) -> pd.DataFrame:
    """
    Analyze multiple ego nodes and return a comparison DataFrame.

    Parameters
    ----------
    G         : Full graph.
    ego_nodes : List of ego node IDs to analyse.
    radius    : Hop radius for each ego network.

    Returns
    -------
    pd.DataFrame — one row per ego node.
    """
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


def run(G: nx.Graph, ego_nodes: list[int] | None = None,
        radius: int = 1) -> pd.DataFrame:
    """
    Phase-3 runner: analyse ego networks and print a summary table.

    Parameters
    ----------
    G         : Full Facebook graph.
    ego_nodes : Nodes to analyse. Defaults to top-10 by degree.
    radius    : Hop radius.
    """
    print("=" * 55)
    print("  PHASE 3 — Ego Network Analysis")
    print("=" * 55)

    if ego_nodes is None:
        ego_nodes = top_ego_candidates(G, k=10)
        print(f"\n  Using top-10 degree nodes as ego centres: {ego_nodes}")

    df = compare_ego_networks(G, ego_nodes, radius=radius)

    print(f"\n  Ego Network Comparison (radius={radius})")
    display_cols = ["ego_node", "n_nodes", "n_edges", "density",
                    "avg_clustering", "ego_degree", "effective_size", "efficiency"]
    print(df[display_cols].to_string(index=False))

    print("\n  Phase 3 complete.\n")
    return df
