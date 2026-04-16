from __future__ import annotations

import math
import os
import random
import sys
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

matplotlib.use("Agg")
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})

try:
    from .config import FIGURES_PATH
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import FIGURES_PATH

FIGURES_DIR = Path(FIGURES_PATH)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _save(fig: plt.Figure, filename: str) -> str:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return str(path)

def _degree_sequence(G: nx.Graph) -> list[int]:
    return [d for _, d in G.degree()]

def _safe_loglog_fit(xs: np.ndarray, ys: np.ndarray):
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return None
    log_x = np.log10(xs[mask])
    log_y = np.log10(ys[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = 10 ** (intercept + slope * np.log10(xs[mask]))
    return float(-slope), fitted

def _greedy_communities_or_single(G: nx.Graph) -> dict:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G)
        mapping = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                mapping[node] = cid
        return mapping
    except Exception:
        return {n: 0 for n in G.nodes()}

def plot_degree_distribution(
    G: nx.Graph,
    title: str = "Degree Distribution",
    filename: str = "degree_distribution.png",
) -> str:
    degrees = _degree_sequence(G)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(degrees, bins=50, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_yscale("log")
    return _save(fig, filename)

def plot_degree_distribution_loglog(
    G: nx.Graph,
    filename: str = "degree_dist_loglog.png",
) -> str:
    degree_counts = Counter(_degree_sequence(G))
    degrees = np.array(sorted(degree_counts.keys()), dtype=float)
    counts = np.array([degree_counts[d] for d in degrees], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(degrees, counts, label="Observed", alpha=0.7)

    fit = _safe_loglog_fit(degrees, counts)
    if fit:
        gamma, fitted = fit
        mask = (degrees > 0) & (counts > 0)
        ax.plot(degrees[mask], fitted, "r--", label=f"Fit (γ ≈ {gamma:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Degree Distribution (log-log)")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.legend()

    return _save(fig, filename)

def plot_synthetic_comparison(
    G_real: nx.Graph,
    G_ba: nx.Graph,
    G_er: nx.Graph,
    filename: str = "synthetic_comparison.png",
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    configs = [
        (G_real, "Real", "steelblue"),
        (G_ba, "Barabasi-Albert", "orange"),
        (G_er, "Erdos-Renyi", "green"),
    ]
    for ax, (G, title, color) in zip(axes, configs):
        ax.hist(_degree_sequence(G), bins=30, color=color)
        ax.set_title(title)
        ax.set_yscale("log")

    return _save(fig, filename)

def plot_ego_network(
    G_ego: nx.Graph,
    ego_node: int,
    community_map: Optional[dict] = None,
    filename: str = "ego_network.png",
) -> str:
    if ego_node not in G_ego:
        raise ValueError("Ego node not in graph")

    if community_map is None:
        community_map = _greedy_communities_or_single(G_ego)

    pos = nx.spring_layout(G_ego, seed=42)
    colors = [community_map.get(n, 0) for n in G_ego.nodes()]
    sizes = [300 if n == ego_node else 50 for n in G_ego.nodes()]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G_ego, pos, node_color=colors, node_size=sizes, with_labels=False, ax=ax)

    return _save(fig, filename)

def plot_top_centrality(
    df: pd.DataFrame,
    col: str,
    k: int = 10,
    filename: str = "centrality.png",
) -> str:
    top = df.nlargest(k, col)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["node"].astype(str), top[col])
    ax.invert_yaxis()
    ax.set_title(f"Top {k} by {col}")

    return _save(fig, filename)


def plot_diffusion_spread(
    spread: list[int],
    total: int,
    filename: str = "diffusion.png",
) -> str:
    steps = range(len(spread))
    fractions = [s / total for s in spread]

    fig, ax = plt.subplots()
    ax.plot(steps, fractions, marker="o")
    ax.set_ylim(0, 1)
    ax.set_title("Diffusion over time")

    return _save(fig, filename)

def plot_community_sizes(
    community_map: dict,
    filename: str = "community_sizes.png",
) -> str:
    counts = Counter(community_map.values())

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_yscale("log")

    return _save(fig, filename)

def plot_community_size_powerlaw(
    community_map: dict,
    filename: str = "community_size_powerlaw.png",
) -> str:
    from collections import Counter

    sizes_sorted = np.array(
        sorted(Counter(community_map.values()).values(), reverse=True), dtype=float
    )
    ranks = np.arange(1, len(sizes_sorted) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ranks, sizes_sorted, color="#e8793a", s=80, zorder=5, label="Community size")

    if len(ranks) >= 3:
        coeffs = np.polyfit(np.log10(ranks), np.log10(sizes_sorted), 1)
        fit_s  = 10 ** np.polyval(coeffs, np.log10(ranks))
        ax.plot(ranks, fit_s, "--", color="#e8793a", alpha=0.55, linewidth=1.5,
                label=f"Power-law fit (slope ≈ {coeffs[0]:.2f})")

    last_lr, last_ls = None, None
    for r, s in zip(ranks, sizes_sorted):
        lr = np.log10(r)
        ls = np.log10(s) if s > 0 else 0
        if last_lr is None or (abs(lr - last_lr) >= 0.20 and abs(ls - last_ls) >= 0.15):
            ax.text(r, s * 1.18, f"{int(s)}", ha="center", va="bottom",
                    fontsize=7, color="#222222")
            last_lr, last_ls = lr, ls

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Community Size Distribution — Power Law")
    ax.set_xlabel("Rank (largest → smallest)")
    ax.set_ylabel("Community Size")
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.legend(fontsize=9)

    return _save(fig, filename)


def plot_baseline_boxplots(
    G_real: nx.Graph,
    G_ba: nx.Graph,
    G_er: nx.Graph,
    filename: str = "baseline.png",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [
        _degree_sequence(G_real),
        _degree_sequence(G_ba),
        _degree_sequence(G_er),
    ]
    axes[0].boxplot(data)
    axes[0].set_title("Degree")

    return _save(fig, filename)
