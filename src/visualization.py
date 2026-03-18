"""
visualization.py
All plotting functions for the Facebook ego-network project.
Figures are saved to the FIGURES_PATH directory defined in config.py.
"""
import os
import sys
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    from .config import FIGURES_PATH
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import FIGURES_PATH

os.makedirs(FIGURES_PATH, exist_ok=True)


# ── Helper ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(FIGURES_PATH, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path}")
    return path


# ── 1. Degree Distribution (linear) ───────────────────────────────────────

def plot_degree_distribution(G: nx.Graph, title: str = "Degree Distribution", filename: str = "degree_distribution.png") -> str:
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(degrees, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_yscale("log")
    return _save(fig, filename)


# ── 2. Degree Distribution (log-log, power-law check) ─────────────────────

def plot_degree_distribution_loglog(G: nx.Graph,
                                    filename: str = "degree_dist_loglog.png") -> str:
    from collections import Counter
    deg_count = Counter(d for _, d in G.degree())
    degrees   = np.array(sorted(deg_count.keys()))
    counts    = np.array([deg_count[d] for d in degrees])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(degrees, counts, s=20, color="steelblue", alpha=0.7, label="Empirical")

    # Fit a power-law line on the log-log scale
    mask = degrees > 0
    log_x, log_y = np.log10(degrees[mask]), np.log10(counts[mask])
    coeffs = np.polyfit(log_x, log_y, 1)
    fit_y  = 10 ** np.polyval(coeffs, log_x)
    ax.plot(degrees[mask], fit_y, "r--", linewidth=1.5,
            label=f"Power-law fit (γ ≈ {-coeffs[0]:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Degree Distribution (log-log)", fontsize=14)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Count P(k)")
    ax.legend()
    return _save(fig, filename)


# ── 3. Synthetic network comparison ───────────────────────────────────────

def plot_synthetic_comparison(G_real: nx.Graph, G_ba:   nx.Graph, G_er:   nx.Graph, filename: str = "synthetic_comparison.png") -> str:
    """Plot side-by-side degree distribution histograms (linear scale)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, G, label, color in zip(
        axes,
        [G_real, G_ba, G_er],
        ["Real (Facebook)", "Barabasi-Albert", "Erdos-Renyi"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        degrees = [d for _, d in G.degree()]
        ax.hist(degrees, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{label}\n(n={G.number_of_nodes():,})", fontsize=11)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")

    fig.suptitle("Degree Distribution Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# ── 4. Ego network graph ───────────────────────────────────────────────────

def plot_ego_network(G_ego: nx.Graph, ego_node: int, community_map: Optional[dict] = None, filename: str = "ego_network.png") -> str:
    """
    Draw the ego network with the ego node highlighted.
    Optionally colour nodes by community.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G_ego, seed=42, k=0.5)

    # Node colours
    if community_map:
        cmap   = plt.colormaps.get_cmap("tab20")
        colors = [cmap(community_map.get(n, 0) % 20) for n in G_ego.nodes()]
    else:
        colors = ["tomato" if n == ego_node else "steelblue" for n in G_ego.nodes()]

    sizes  = [600 if n == ego_node else 100 for n in G_ego.nodes()]
    labels = {ego_node: str(ego_node)}

    nx.draw_networkx_edges(G_ego, pos, ax=ax, alpha=0.3, edge_color="gray") 
    nx.draw_networkx_nodes(G_ego, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(G_ego, pos, labels=labels, ax=ax, font_size=10, font_color="white", font_weight="bold")
    ax.set_title(f"Ego Network — Node {ego_node} "f"({G_ego.number_of_nodes()} nodes, "f"{G_ego.number_of_edges()} edges)", fontsize=13)
    ax.axis("off")
    return _save(fig, filename)


# ── 5. Centrality comparison bar chart ────────────────────────────────────

def plot_top_centrality(df_centrality: pd.DataFrame,
                        col: str = "betweenness_centrality",
                        k: int = 15,
                        filename: Optional[str] = None) -> str:
    top = df_centrality.nlargest(k, col)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["node"].astype(str), top[col], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_title(f"Top-{k} Nodes by {col.replace('_', ' ').title()}", fontsize=13)
    if filename is None:
        filename = f"top_{col}.png"
    return _save(fig, filename)


# ── 6. Diffusion spread curve ──────────────────────────────────────────────

def plot_diffusion_spread(spread_by_step: list[int], n_total: int, filename: str = "diffusion_spread.png") -> str:
    """
    Plot the fraction of nodes activated at each diffusion step.

    Parameters
    ----------
    spread_by_step : List where index = step, value = cumulative activated count.
    n_total        : Total nodes in the graph.
    """
    steps     = list(range(len(spread_by_step)))
    fractions = [s / n_total for s in spread_by_step]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, fractions, marker="o", color="tomato", linewidth=2, markersize=5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Full coverage")
    ax.set_xlabel("Diffusion step")
    ax.set_ylabel("Fraction of nodes activated")
    ax.set_title("Independent Cascade — Spread over Time", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _save(fig, filename)


# ── 7. Multi-ego comparison heatmap ───────────────────────────────────────

def plot_ego_comparison_heatmap(ego_df: pd.DataFrame,
                                filename: str = "ego_heatmap.png") -> str:
    """
    Heatmap of normalised ego-network metrics for easy comparison.

    Parameters
    ----------
    ego_df : DataFrame returned by ego_analysis.compare_ego_networks().
    """
    numeric_cols = ["n_nodes", "n_edges", "density", "avg_clustering",
                    "ego_degree", "effective_size", "efficiency"]
    available    = [c for c in numeric_cols if c in ego_df.columns]
    data         = ego_df[["ego_node"] + available].set_index("ego_node")

    # Normalise each column to [0, 1]
    normed = (data - data.min()) / (data.max() - data.min()).replace(0, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.2),
                                    max(4, len(normed) * 0.5)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Normalised value")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=9)
    ax.set_yticks(range(len(normed)))
    ax.set_yticklabels([f"Node {idx}" for idx in normed.index], fontsize=9)
    ax.set_title("Ego Network Metrics — Normalised Comparison", fontsize=13)

    # Annotate cells with raw values
    for i in range(len(normed)):
        for j in range(len(available)):
            raw_val = data.values[i, j]
            text    = f"{raw_val:.2f}" if isinstance(raw_val, float) else str(raw_val)
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="black" if normed.values[i, j] < 0.6 else "white")

    plt.tight_layout()
    return _save(fig, filename)
