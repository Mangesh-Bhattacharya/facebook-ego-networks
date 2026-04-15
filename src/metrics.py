"""
metrics.py
Global and node-level network metrics for the Facebook graph.
"""
import networkx as nx
import pandas as pd
from collections import Counter


# ── Global summary ─────────────────────────────────────────────────────────

def _approximate_diameter(G: nx.Graph, n_samples: int = 10, seed: int = 42) -> int:
    """
    Estimate the diameter by running BFS from a small random sample of nodes
    and returning the maximum eccentricity found.  Much faster than the exact
    O(V * (V+E)) computation; usually within 1–2 of the true value.
    """
    import random as _random
    rng = _random.Random(seed)
    nodes = rng.sample(list(G.nodes()), min(n_samples, G.number_of_nodes()))
    return max(max(nx.single_source_shortest_path_length(G, v).values()) for v in nodes)


def global_summary(G: nx.Graph) -> dict:
    """Compute global network summary metrics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_degree = 2 * m / n if n > 0 else 0
    clustering = nx.average_clustering(G)
    diameter   = _approximate_diameter(G) if nx.is_connected(G) else float("inf")

    return {
        "nodes":      n,
        "edges":      m,
        "avg_degree": round(avg_degree, 4),
        "clustering": round(clustering, 6),
        "diameter":   diameter,
    }


# ── Degree distribution ────────────────────────────────────────────────────

def degree_distribution(G: nx.Graph) -> pd.DataFrame:
    """
    Return a DataFrame of (degree, count, fraction) sorted by degree.
    Useful for log-log power-law plots.
    """
    deg_seq   = [d for _, d in G.degree()]
    count_map = Counter(deg_seq)
    total     = len(deg_seq)
    df = pd.DataFrame(
        [(k, v, v / total) for k, v in sorted(count_map.items())],
        columns=["degree", "count", "fraction"],
    )
    return df


# ── Centrality measures ────────────────────────────────────────────────────

def degree_centrality(G: nx.Graph) -> pd.DataFrame:
    """Normalised degree centrality for every node."""
    dc = nx.degree_centrality(G)
    return _centrality_df(dc, "degree_centrality")


def betweenness_centrality(G: nx.Graph,
                           k: int | None = None,
                           seed: int = 42) -> pd.DataFrame:
    """
    Betweenness centrality.
    Pass k for an approximate calculation on large graphs (k = sample size).
    """
    bc = nx.betweenness_centrality(G, k=k, seed=seed, normalized=True)
    return _centrality_df(bc, "betweenness_centrality")


def closeness_centrality(G: nx.Graph) -> pd.DataFrame:
    """Closeness centrality for every node."""
    cc = nx.closeness_centrality(G)
    return _centrality_df(cc, "closeness_centrality")


def eigenvector_centrality(G: nx.Graph,
                           max_iter: int = 1000) -> pd.DataFrame:
    """Eigenvector centrality (power iteration)."""
    ec = nx.eigenvector_centrality(G, max_iter=max_iter)
    return _centrality_df(ec, "eigenvector_centrality")


def _centrality_df(centrality_dict: dict, col_name: str) -> pd.DataFrame:
    df = (
        pd.DataFrame(centrality_dict.items(), columns=["node", col_name])
        .sort_values(col_name, ascending=False)
        .reset_index(drop=True)
    )
    return df


def top_nodes_by_centrality(G: nx.Graph, k: int = 10) -> pd.DataFrame:
    """
    Return a combined DataFrame of the top-k nodes across four centrality
    measures. Useful for quick comparisons.
    """
    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), seed=42)
    cc = nx.closeness_centrality(G)
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        ec = {n: float("nan") for n in G.nodes()}

    df = pd.DataFrame({
        "node":                  list(G.nodes()),
        "degree":                [G.degree(n) for n in G.nodes()],
        "degree_centrality":     [dc[n] for n in G.nodes()],
        "betweenness_centrality":[bc[n] for n in G.nodes()],
        "closeness_centrality":  [cc[n] for n in G.nodes()],
        "eigenvector_centrality":[ec[n] for n in G.nodes()],
    })

    # Rank by degree, return top-k
    return df.sort_values("degree", ascending=False).head(k).reset_index(drop=True)


# ── Community detection ────────────────────────────────────────────────────

def detect_communities_greedy(G: nx.Graph) -> dict:
    """
    Detect communities with the Clauset-Newman-Moore greedy modularity
    algorithm. Returns a dict mapping node -> community_id.
    """
    from networkx.algorithms.community import greedy_modularity_communities
    communities = greedy_modularity_communities(G)
    node_community = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_community[node] = cid
    n_comm = len(communities)
    sizes  = sorted([len(c) for c in communities], reverse=True)
    print(f"  Communities (greedy modularity): {n_comm} "
          f"| largest: {sizes[0]} nodes | smallest: {sizes[-1]} nodes")
    return node_community


def detect_communities_label_propagation(G: nx.Graph) -> dict:
    """
    Detect communities with asynchronous label propagation.
    Faster than greedy modularity; non-deterministic.
    """
    from networkx.algorithms.community import asyn_lpa_communities
    communities = list(asyn_lpa_communities(G, seed=42))
    node_community = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_community[node] = cid
    n_comm = len(communities)
    sizes  = sorted([len(c) for c in communities], reverse=True)
    print(f"  Communities (label propagation): {n_comm} "
          f"| largest: {sizes[0]} nodes | smallest: {sizes[-1]} nodes")
    return node_community


# ── Phase-2 runner ─────────────────────────────────────────────────────────

def run(G: nx.Graph) -> dict:
    """Phase-2 runner: compute and print all global metrics."""
    print("=" * 55)
    print("  PHASE 2 — Global Network Metrics")
    print("=" * 55)

    summary = global_summary(G)
    print("\n  Global Summary")
    for key, val in summary.items():
        print(f"    {key:<20}: {val}")

    print("\n  Top-10 nodes by centrality")
    top = top_nodes_by_centrality(G, k=10)
    print(top.to_string(index=False))

    print("\n  Phase 2 complete.\n")
    return summary
