"""
visualization.py
All plotting functions for the Facebook ego-network project.
Figures are saved to the FIGURES_PATH directory defined in config.py.
"""
# cspell:ignore zorder colormaps markerfacecolor fontsize linewidth edgecolor facecolor labelcolor markersize
# cspell:ignore whiskerprops capprops axisbelow suptitle polyfit Gsub fontweight
# cspell:ignore Assortativity assortativity Jaccard jaccard Barabasi Erdos Renyi
import os
import sys
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
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

def plot_ego_network(G_ego: nx.Graph, ego_node: int, community_map: Optional[dict] = None, filename: str = "ego_network.png", top_hubs_per_comm: int = 2, max_intra_edges: int = 400, max_ego_spokes: int = 300) -> str:
    """
    Sunflower layout — each community is its own clearly separated cluster
    blob orbiting the ego node at the centre.

    Layout
    ------
    * Ego node at (0, 0), drawn large with a glowing ring.
    * Each community occupies a distinct circular blob whose centroid sits on
        an outer orbit ring.  The centroid angle is evenly spaced so no two
        communities overlap.
    * Within each community blob, nodes are arranged in tight concentric
        rings (innermost = highest degree).  Blob radius scales with
        sqrt(community_size) so large communities don't crowd each other.
    * If community_map is None, greedy modularity detection is run on G_ego
        so the picture always shows meaningful groups.

    Visual encoding
    ---------------
    * Node size   ∝ degree within G_ego  (hubs are clearly larger)
    * Node colour = community  (tab20 palette)
    * Ego node    = vivid red, 3x larger than biggest hub
    * Background filled circles per community make clusters unmistakable
    * Dashed spokes from ego centroid to each community centroid
    * Intra-community edges drawn as thin coloured lines (sampled if dense)
    * Ego-to-alter edges drawn as faint red lines (sampled if > max_ego_spokes)
    * Top 2 hubs per community are labelled in white
    * Legend lists every community with its node count

    Parameters
    ----------
    G_ego              : Ego-network subgraph (ego node included).
    ego_node           : ID of the central node.
    community_map      : Optional dict  node -> community_id.
    filename           : Output PNG filename.
    top_hubs_per_comm  : Hub labels shown per community.
    max_intra_edges    : Cap on intra-community edges drawn per community.
    max_ego_spokes     : Cap on ego→alter lines drawn.
    """
    import math
    import random
    from collections import defaultdict

    random.seed(42)
    alters  = [n for n in G_ego.nodes() if n != ego_node]
    n_alters = len(alters)

    # ── auto community detection if none provided ────────────────────────────
    if community_map is None:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            raw = greedy_modularity_communities(G_ego)
            community_map = {}
            for cid, comm in enumerate(raw):
                for node in comm:
                    community_map[node] = cid
        except Exception:
            community_map = {n: 0 for n in G_ego.nodes()}

    # ── degree within ego network ────────────────────────────────────────────
    ego_deg = dict(G_ego.degree())
    max_deg = max((ego_deg[n] for n in alters), default=1)

    # ── group alters into communities ────────────────────────────────────────
    groups: dict = defaultdict(list)
    for n in alters:
        groups[community_map.get(n, 0)].append(n)
    # sort largest community first; within each group sort by degree desc
    ordered_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for _, members in ordered_groups:
        members.sort(key=lambda n: ego_deg.get(n, 0), reverse=True)

    n_comms = len(ordered_groups)
    cmap    = plt.colormaps["tab20"]

    # ── place community blobs on orbit ring ──────────────────────────────────
    R_orbit      = 9.0     # distance from ego to community centroid
    comm_centers = {}
    pos          = {ego_node: (0.0, 0.0)}

    for g_idx, (cid, members) in enumerate(ordered_groups):
        angle           = 2 * math.pi * g_idx / n_comms - math.pi / 2
        cx              = R_orbit * math.cos(angle)
        cy              = R_orbit * math.sin(angle)
        comm_centers[cid] = (cx, cy)

        m        = len(members)
        r_blob   = 1.0 + 0.22 * math.sqrt(m)   # cluster radius grows with size

        # fill concentric rings innermost → outermost
        remaining = list(members)
        ring_r    = r_blob * 0.25
        while remaining:
            # how many nodes fit on this ring at ~0.6-unit spacing
            cap        = max(1, int(2 * math.pi * ring_r / 0.6))
            ring_nodes = remaining[:cap]
            remaining  = remaining[cap:]
            for k, node in enumerate(ring_nodes):
                a        = 2 * math.pi * k / max(len(ring_nodes), 1)
                pos[node] = (cx + ring_r * math.cos(a),
                             cy + ring_r * math.sin(a))
            ring_r += 0.65

    # ── node colours and sizes ───────────────────────────────────────────────
    comm_id_to_gidx = {cid: i for i, (cid, _) in enumerate(ordered_groups)}
    node_colors, node_sizes = [], []

    for n in G_ego.nodes():
        if n == ego_node:
            node_colors.append("#e74c3c")
            node_sizes.append(2200)
        else:
            g_idx = comm_id_to_gidx.get(community_map.get(n, 0), 0)
            node_colors.append(cmap((g_idx % 20) / 20))
            sz = 35 + 380 * (ego_deg.get(n, 1) / max_deg) ** 0.6
            node_sizes.append(sz)

    # ── edge sets ─────────────────────────────────────────────────────────────
    # ego spokes
    ego_edges = [(ego_node, v) for v in alters if G_ego.has_edge(ego_node, v)]
    if len(ego_edges) > max_ego_spokes:
        ego_edges = random.sample(ego_edges, max_ego_spokes)

    # intra-community edges (per community to avoid large-community domination)
    intra_edges = []
    for cid, members in ordered_groups:
        comm_set = set(members)
        c_edges  = [(u, v) for u, v in G_ego.edges()
                    if u in comm_set and v in comm_set]
        if len(c_edges) > max_intra_edges:
            c_edges = random.sample(c_edges, max_intra_edges)
        intra_edges.extend(c_edges)

    # ── hub labels ───────────────────────────────────────────────────────────
    label_dict = {ego_node: str(ego_node)}
    for cid, members in ordered_groups:
        for hub in members[:top_hubs_per_comm]:
            label_dict[hub] = str(hub)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 18), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")

    # ── community blobs (filled circles) ─────────────────────────────────────
    for g_idx, (cid, members) in enumerate(ordered_groups):
        color    = cmap((g_idx % 20) / 20)
        cx, cy   = comm_centers[cid]
        m        = len(members)
        r_fill   = 1.15 + 0.22 * math.sqrt(m)

        # outer glow
        glow = Circle((cx, cy), r_fill + 0.35, color=color, alpha=0.06, zorder=1)
        ax.add_patch(glow)
        # solid fill
        blob = Circle((cx, cy), r_fill, color=color, alpha=0.14, zorder=2)
        ax.add_patch(blob)
        # border ring
        border = Circle((cx, cy), r_fill, color=color, alpha=0.55, fill=False, linewidth=1.2, zorder=3)
        ax.add_patch(border)

        # community label below blob
        ax.text(cx, cy - r_fill - 0.55,
                f"Community {cid}  (n={m})",
                ha="center", va="top", fontsize=8,
                color=color, fontweight="bold")

    # ── spoke lines ego → community centroid ─────────────────────────────────
    for g_idx, (cid, _) in enumerate(ordered_groups):
        color  = cmap((g_idx % 20) / 20)
        cx, cy = comm_centers[cid]
        ax.plot([0, cx], [0, cy],
                color=color, alpha=0.22, lw=1.0,
                linestyle="--", zorder=4)

    # ── ego glow ring ─────────────────────────────────────────────────────────
    for r, a in [(1.0, 0.06), (0.65, 0.12), (0.38, 0.20)]:
        ax.add_patch(Circle((0, 0), r, color="#e74c3c",
                                alpha=a, zorder=5, fill=True))

    # ── edges ─────────────────────────────────────────────────────────────────
    ec_intra = nx.draw_networkx_edges(G_ego, pos, edgelist=intra_edges, ax=ax, alpha=0.18, edge_color="#95a5a6", width=0.5)
    if ec_intra is not None:
        ec_intra.set_zorder(6)
    ec_ego = nx.draw_networkx_edges(G_ego, pos, edgelist=ego_edges, ax=ax, alpha=0.12, edge_color="#e74c3c", width=0.5)
    if ec_ego is not None:
        ec_ego.set_zorder(7)

    # ── nodes ─────────────────────────────────────────────────────────────────
    nodes_collection = nx.draw_networkx_nodes(G_ego, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.93)
    if nodes_collection is not None:
        nodes_collection.set_zorder(8)

    # ── labels ────────────────────────────────────────────────────────────────
    nx.draw_networkx_labels(G_ego, pos,
                            labels={ego_node: str(ego_node)},
                            ax=ax, font_size=11,
                            font_color="white", font_weight="bold")
    alter_labels = {k: v for k, v in label_dict.items() if k != ego_node}
    nx.draw_networkx_labels(G_ego, pos, labels=alter_labels,
                            ax=ax, font_size=6.5, font_color="white")

    # ── legend ────────────────────────────────────────────────────────────────
    shown   = min(n_comms, 18)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap((ordered_groups[i][0] % 20) / 20), markersize=9, label=f"C{ordered_groups[i][0]}  " f"(n={len(ordered_groups[i][1])})")
        for i in range(shown)
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    fontsize=8, title="Communities",
                    title_fontsize=9, ncol=2,
                    framealpha=0.25, facecolor="#1a1d27",
                    labelcolor="white", edgecolor="#555")
    leg.get_title().set_color("white")

    # ── title ─────────────────────────────────────────────────────────────────
    n_intra_total = sum(1 for u, v in G_ego.edges()
                        if u != ego_node and v != ego_node)
    ax.set_title(
        f"Ego Network — Node {ego_node}\n"
        f"{n_alters:,} alters  ·  {G_ego.number_of_edges():,} edges  ·  "
        f"{n_comms} communities  ·  "
        f"{len(intra_edges)}/{n_intra_total} intra-alter edges shown",
        fontsize=14, pad=16, color="white"
    )
    ax.axis("off")
    plt.tight_layout()
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


# ── 8. Greedy Modularity — community-aware layout, proportional node sizes ─

def plot_community_graph(G: nx.Graph, community_map: dict, max_nodes: int = 1200, top_hubs_per_comm: int = 2, filename: str = "community_graph.png") -> str:
    """
    Draw the graph with a community-aware layout so each cluster is visually
    separated.  Node sizes are proportional to degree so hubs stand out.
    Top hub nodes in each community are labelled.

    Layout strategy
    ---------------
    1. Place community centroids on a circle (well separated).
    2. Within each community, distribute nodes in a small circle around
        their centroid — no community overlaps with its neighbour.
    3. Node area ∝ degree  (sqrt scaling keeps very high-degree hubs readable).
    4. Intra-community edges are drawn lightly; inter-community edges are
        drawn even more faintly so the cluster structure is obvious.

    Parameters
    ----------
    G                  : Full NetworkX graph.
    community_map      : dict  node -> community_id
    max_nodes          : Subsample cap for speed / readability.
    top_hubs_per_comm  : How many hub nodes per community to label.
    filename           : Output PNG filename.
    """
    from collections import defaultdict
    import math

    # ── subsample if needed ──────────────────────────────────────────────────
    if G.number_of_nodes() > max_nodes:
        # keep the highest-degree nodes so structure is preserved
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        G = G.subgraph(top_nodes[:max_nodes]).copy()

    # ── community bookkeeping ────────────────────────────────────────────────
    comm_members = defaultdict(list)
    for node in G.nodes():
        comm_members[community_map.get(node, 0)].append(node)

    comm_ids   = sorted(comm_members.keys(),
                        key=lambda c: len(comm_members[c]), reverse=True)
    n_comms    = len(comm_ids)
    cmap       = plt.colormaps["tab20"]

    # ── build positions ──────────────────────────────────────────────────────
    pos           = {}
    centroid_pos  = {}
    R_outer       = 10.0          # radius of the ring of community centres
    r_inner_base  = 2.0           # base inner-cluster radius
    scale_inner   = 0.55          # shrink large communities slightly

    for idx, cid in enumerate(comm_ids):
        angle    = 2 * math.pi * idx / n_comms
        cx, cy   = R_outer * math.cos(angle), R_outer * math.sin(angle)
        centroid_pos[cid] = (cx, cy)

        members  = comm_members[cid]
        m        = len(members)
        r_inner  = r_inner_base + scale_inner * math.sqrt(m)

        for k, node in enumerate(members):
            a = 2 * math.pi * k / max(m, 1)
            pos[node] = (cx + r_inner * math.cos(a),
                         cy + r_inner * math.sin(a))

    # ── node sizes proportional to degree ────────────────────────────────────
    degrees    = dict(G.degree())
    max_deg    = max(degrees.values()) if degrees else 1
    node_sizes = [30 + 600 * (degrees[n] / max_deg) ** 0.6
                    for n in G.nodes()]

    # ── colours ──────────────────────────────────────────────────────────────
    node_colors = [cmap((community_map.get(n, 0) % 20) / 20) for n in G.nodes()]

    # ── classify edges ────────────────────────────────────────────────────────
    intra_edges = [(u, v) for u, v in G.edges() if community_map.get(u) == community_map.get(v)]
    inter_edges = [(u, v) for u, v in G.edges() if community_map.get(u) != community_map.get(v)]

    # ── hub labels: top-N by degree in each community ────────────────────────
    label_nodes = {}
    for cid in comm_ids:
        members_in_G = [n for n in comm_members[cid] if n in G]
        hubs = sorted(members_in_G, key=lambda n: degrees.get(n, 0), reverse=True)
        for hub in hubs[:top_hubs_per_comm]:
            label_nodes[hub] = str(hub)

    # ── draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 14))

    # community halos
    for idx, cid in enumerate(comm_ids):
        members = [n for n in comm_members[cid] if n in pos]
        if not members:
            continue
        xs = [pos[n][0] for n in members]
        ys = [pos[n][1] for n in members]
        r  = max(math.dist([np.mean(xs), np.mean(ys)], [pos[n][0], pos[n][1]]) for n in members) + 0.6
        circle = Circle((float(np.mean(xs)), float(np.mean(ys))), r, color=cmap((idx % 20) / 20), alpha=0.07, zorder=0)
        ax.add_patch(circle)
        # community label at centroid
        ax.text(np.mean(xs), np.mean(ys) + r + 0.2, f"C{cid}(n={len(members)})", ha="center", va="bottom", fontsize=7, color=cmap((idx % 20) / 20), fontweight="bold")

    # edges
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inter_edges, alpha=0.06, edge_color="gray", width=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=intra_edges, alpha=0.25, edge_color="gray", width=0.6)

    # nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    # hub labels
    nx.draw_networkx_labels(G, pos, labels=label_nodes, ax=ax,
                            font_size=6, font_color="white",
                            font_weight="bold")

    # legend — one swatch per community (cap at 20)
    shown   = min(n_comms, 20)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap((comm_ids[i] % 20) / 20), markersize=8, label=f"C{comm_ids[i]} (n={len(comm_members[comm_ids[i]])})")
        for i in range(shown)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, title=f"Top-{shown} communities by size", title_fontsize=8, ncol=max(1, shown // 8), framealpha=0.6)

    ax.set_title(
        f"Greedy Modularity Communities — {n_comms} communities | "
        f"{G.number_of_nodes()} nodes shown\n"
        f"Node size ∝ degree  •  Labelled nodes = top hubs per community",
        fontsize=13
    )
    ax.axis("equal")
    ax.axis("off")
    plt.tight_layout()
    return _save(fig, filename)


# ── 9. Community size distribution ────────────────────────────────────────

def plot_community_sizes(community_map: dict, filename: str = "community_sizes.png") -> str:
    """
    Bar chart of community sizes sorted in descending order.

    Parameters
    ----------
    community_map : dict mapping node -> community_id.
    """
    from collections import Counter
    counts   = Counter(community_map.values())
    sorted_c = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    comm_ids = [str(c) for c, _ in sorted_c]
    sizes    = [s for _, s in sorted_c]

    fig, ax = plt.subplots(figsize=(max(10, len(sizes) * 0.4), 5))
    bars = ax.bar(comm_ids, sizes, color="steelblue", edgecolor="white", alpha=0.85)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(size), ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Community ID")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"Community Size Distribution " f"({len(counts)} communities, {sum(sizes)} nodes total)", fontsize=13)
    ax.set_yscale("log")
    if len(sizes) > 30:
        ax.set_xticks([])
        ax.set_xlabel("Communities (sorted by size, largest first)")
    plt.tight_layout()
    return _save(fig, filename)


# ── 10. Inter-community edge density heatmap ──────────────────────────────

def plot_community_matrix(G: nx.Graph, community_map: dict, top_k: int = 15, filename: str = "community_matrix.png") -> str:
    """
    Heatmap of edge density between the `top_k` largest communities.
    Cell (i, j) = edges(Ci, Cj) / (|Ci| * |Cj|)   for i != j
                = 2*intra_edges / (|Ci|*(|Ci|-1))  for i == j

    Parameters
    ----------
    G             : The full NetworkX graph.
    community_map : dict mapping node -> community_id.
    top_k         : Only show the top_k largest communities.
    """
    from collections import Counter, defaultdict

    size_by_comm = Counter(community_map.values())
    top_comms    = [c for c, _ in size_by_comm.most_common(top_k)]
    comm_index   = {c: i for i, c in enumerate(top_comms)}
    k            = len(top_comms)

    edge_counts = defaultdict(int)
    for u, v in G.edges():
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu in comm_index and cv in comm_index:
            i, j = comm_index[cu], comm_index[cv]
            edge_counts[(min(i, j), max(i, j))] += 1

    matrix = np.zeros((k, k))
    for (i, j), count in edge_counts.items():
        si = size_by_comm[top_comms[i]]
        sj = size_by_comm[top_comms[j]]
        denom = (si * (si - 1) / 2) if (i == j and si > 1) else (si * sj)
        matrix[i, j] = count / denom
        matrix[j, i] = count / denom

    labels = [f"C{c}\n(n={size_by_comm[c]})" for c in top_comms]
    fig, ax = plt.subplots(figsize=(max(8, k * 0.7), max(7, k * 0.65)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Edge density")

    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Inter-Community Edge Density (top-{k} communities)", fontsize=13)

    for i in range(k):
        for j in range(k):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if val > matrix.max() * 0.6 else "black")

    plt.tight_layout()
    return _save(fig, filename)


# ── 11. Baseline boxplots (degree & clustering across Real / BA / ER) ──────

def plot_baseline_boxplots(G_real: nx.Graph, G_ba:   nx.Graph, G_er:   nx.Graph, filename: str = "baseline_boxplots.png") -> str:
    """
    Side-by-side boxplots comparing degree distribution and local clustering
    coefficient across the real Facebook graph, a Barabasi-Albert model, and
    an Erdos-Renyi model.

    Panels
    ------
    Left  — degree distribution per network
    Right — local clustering coefficient per network

    Parameters
    ----------
    G_real, G_ba, G_er : NetworkX graphs (undirected, unweighted).
    filename            : Output PNG filename.
    """
    networks = [G_real, G_ba, G_er]
    labels   = ["Real\n(Facebook)", "Barabasi-\nAlbert", "Erdos-\nRenyi"]
    colors   = ["#4878CF", "#D65F5F", "#6ACC65"]

    degree_data     = [[d for _, d in G.degree()]      for G in networks]
    clustering_data = [list(nx.clustering(G).values()) for G in networks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, data, title, ylabel in zip(
        axes,
        [degree_data, clustering_data],
        ["Degree Distribution", "Local Clustering Coefficient"],
        ["Degree", "Clustering coefficient"],
    ):
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # annotate median value next to each box
        for i, d in enumerate(data):
            med = float(np.median(d))
            ax.text(i + 1, med, f" {med:.2f}",
                    va="center", ha="left", fontsize=8, color="black")

        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("Baseline Comparison — Real vs Synthetic Networks", fontsize=14, y=1.01)
    plt.tight_layout()
    return _save(fig, filename)


# ── 12. Similarity / structural metrics table ──────────────────────────────

def plot_similarity_table(G_real: nx.Graph, G_ba:   nx.Graph, G_er:   nx.Graph, filename: str = "similarity_table.png") -> str:
    """
    Render a formatted PNG table of structural similarity metrics comparing
    the real Facebook graph against two synthetic baselines.

    Metrics reported
    ----------------
    Nodes, Edges, Avg Degree, Density, Avg Clustering, Transitivity,
    Diameter (LCC), Avg Shortest Path (sampled), Degree Assortativity,
    Power-law gamma, Jaccard overlap of top-50 hubs (vs Real).

    Parameters
    ----------
    G_real, G_ba, G_er : NetworkX graphs.
    filename            : Output PNG filename.
    """
    from collections import Counter
    import random

    def _gamma(G):
        deg_seq = [d for _, d in G.degree() if d > 0]
        cnt     = Counter(deg_seq)
        xs      = np.log10(np.array(sorted(cnt.keys()), dtype=float))
        ys      = np.log10(np.array([cnt[k] for k in sorted(cnt.keys())],
                                    dtype=float))
        return round(float(-np.polyfit(xs, ys, 1)[0]), 3) if len(xs) >= 2 else float("nan")

    def _avg_sp(G, k=500):
        lcc  = max(nx.connected_components(G), key=len)
        Gsub = G.subgraph(lcc)
        if Gsub.number_of_nodes() <= 1:
            return float("nan")
        sample  = random.sample(list(Gsub.nodes()), min(k, Gsub.number_of_nodes()))
        lengths = []
        for src in sample:
            lengths.extend(nx.single_source_shortest_path_length(Gsub, src).values())
        return round(float(np.mean(lengths)), 4)

    def _jaccard_hubs(G_ref, G_other, top=50):
        hubs_ref   = set(sorted(G_ref.nodes(),   key=lambda n: G_ref.degree(n),   reverse=True)[:top])
        hubs_other = set(sorted(G_other.nodes(), key=lambda n: G_other.degree(n), reverse=True)[:top])
        inter = len(hubs_ref & hubs_other)
        union = len(hubs_ref | hubs_other)
        return round(inter / union, 4) if union else 0.0

    def _stats(G):
        n   = G.number_of_nodes()
        m   = G.number_of_edges()
        lcc = max(nx.connected_components(G), key=len)
        return {
            "Nodes":               n,
            "Edges":               m,
            "Avg Degree":          round(2 * m / n, 3) if n else 0,
            "Density":             round(nx.density(G), 6),
            "Avg Clustering":      round(nx.average_clustering(G), 4),
            "Transitivity":        round(nx.transitivity(G), 4),
            "Diameter (LCC)":      nx.diameter(G.subgraph(lcc)) if len(lcc) > 1 else 0,
            "Avg Shortest Path":   _avg_sp(G),
            "Assortativity":       round(nx.degree_assortativity_coefficient(G), 4),
            "Power-law gamma":     _gamma(G),
        }

    sr, sb, se = _stats(G_real), _stats(G_ba), _stats(G_er)
    metrics    = list(sr.keys()) + ["Jaccard Hub-50 (vs Real)"]
    col_real   = [str(sr[m]) for m in sr] + ["—"]
    col_ba     = [str(sb[m]) for m in sb] + [str(_jaccard_hubs(G_real, G_ba))]
    col_er     = [str(se[m]) for m in se] + [str(_jaccard_hubs(G_real, G_er))]

    n_rows = len(metrics)
    fig, ax = plt.subplots(figsize=(11, 0.48 * n_rows + 1.6))
    ax.axis("off")

    table = ax.table(
        cellText  = [[m, r, b, e] for m, r, b, e in zip(metrics, col_real, col_ba, col_er)],
        colLabels = ["Metric", "Real (Facebook)", "Barabasi-Albert", "Erdos-Renyi"],
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)

    header_color = "#2c3e50"
    for col_idx in range(4):
        cell = table[0, col_idx]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#f2f2f2", "#ffffff"]
    for row_idx in range(1, n_rows + 1):
        for col_idx in range(4):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(row_colors[(row_idx - 1) % 2])
            if col_idx == 0:
                cell.set_text_props(fontweight="bold")

    ax.set_title("Structural Similarity Metrics — Real vs Synthetic Networks", fontsize=13, pad=14)
    plt.tight_layout()
    return _save(fig, filename)
