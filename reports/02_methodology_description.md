# Methodology Description

## Facebook Ego-Network Analysis Pipeline

---

## Overview

The analysis is implemented as a five-phase Python pipeline (`main.py`) operating on the SNAP Facebook combined ego-network dataset. Each phase is modular and produces intermediate CSV outputs and PNG figures that feed into subsequent phases and the final report.

```text
Raw CSV  →  Phase 1: Clean  →  Phase 2: Global Metrics  →  Phase 3: Ego Analysis →  Phase 4: Synthetic Comparison →  Phase 5: Diffusion Simulation
```

---

## Dataset

**Source:** Stanford Network Analysis Project (SNAP)
**File:** `data/raw/facebook_combined.csv`
**Format:** Edge list — two tab/space-separated integer node IDs per row, representing undirected friendship connections.
**Scale:** 4,039 users (nodes), 88,234 friendship pairs (edges).
**Origin:** Collected from Facebook survey participants. Node IDs are anonymized. The combined file merges 10 individual ego networks into a single connected component.

---

## Phase 1 — Data Cleaning (`src/cleaning.py`)

**Goal:** Produce a clean, canonical edge list for graph construction.

**Steps:**

1. **Load** raw CSV with auto-format detection (`src/data_loading.py`): handles space-separated, comma-separated, and tab-separated variants; assigns columns `node1`, `node2`.
2. **Canonicalise direction:** For each edge (u, v), sort so that min(u,v) is always `node1`. This ensures (u,v) and (v,u) are treated as the same undirected edge.
3. **Drop duplicates:** After canonicalisation, remove identical rows.
4. **Remove self-loops:** Drop rows where `node1 == node2`.
5. **Save:** Write cleaned edges to `data/processed/edges_clean.csv`.

**Output artefacts:**

- `data/processed/edges_clean.csv` — canonical edge list
- `data/processed/degree_frequency.csv` — node degree counts sorted descending

---

## Phase 2 — Graph Construction & Global Metrics (`src/graph_builder.py`, `src/metrics.py`)

**Graph construction (`build_facebook_graph`):**

- Construct `nx.Graph` (undirected, unweighted) from the cleaned edge list.
- Add `degree` as a node attribute via `nx.set_node_attributes`.

**Global metrics (`global_summary`):**

| Metric | Computation |
| --- | --- |
| Nodes (n) | `G.number_of_nodes()` |
| Edges (m) | `G.number_of_edges()` |
| Average degree | 2m / n |
| Average clustering | `nx.average_clustering(G)` — mean of local clustering coefficients |
| Diameter | `nx.diameter(G)` if connected, else ∞ |

**Degree distribution:**

- Tabulate (degree, count, fraction) using `collections.Counter`.
- Plot linear-scale histogram (log y-axis) and log-log scatter with power-law fit.
- Power-law exponent γ estimated via log-log linear regression: fit a line to log(count) ~ log(degree), take −slope as γ.

**Centrality analysis (`top_nodes_by_centrality`):**

| Measure | Method | Complexity |
| --- | --- | --- |
| Degree centrality | `nx.degree_centrality` | O(n) |
| Betweenness centrality | `nx.betweenness_centrality(k=500)` | O(k·m) approx. |
| Closeness centrality | `nx.closeness_centrality` | O(nm) |
| Eigenvector centrality | `nx.eigenvector_centrality(max_iter=1000)` | O(m·iter) |

Top-10 nodes by degree are reported with all four centrality scores.

**Community detection:**

Two algorithms are implemented:

1. **Greedy Modularity (Clauset-Newman-Moore)** — `greedy_modularity_communities` from NetworkX. Agglomerative: starts with each node in its own community and merges pairs that maximise ΔQ. Returns a `frozenset` per community; converted to `dict[node → community_id]`.

2. **Asynchronous Label Propagation** — `asyn_lpa_communities` from NetworkX. Each node adopts the majority label among its neighbours, iterated until stable. Non-deterministic; seeded with `random_state=42` for reproducibility.

**Output artefacts:**

- `figures/degree_distribution.png`
- `figures/degree_dist_loglog.png`
- `data/processed/degree_distribution.csv`
- `data/processed/top_centrality_nodes.csv`
- `data/processed/communities.csv`

---

## Phase 3 — Ego Network Analysis (`src/ego_analysis.py`)

**Ego node selection:** Top-10 nodes by degree in the full graph are chosen as ego centres (`top_ego_candidates`).

**Ego network extraction:** `nx.ego_graph(G, ego_node, radius=1)` — returns the subgraph induced by the ego node and all nodes within 1 hop, including all edges among them.

**Metrics computed per ego node (`analyze_ego_network`):**

| Metric | Definition |
| --- | --- |
| n_nodes, n_edges | Size of the ego network subgraph |
| density | 2m / (n(n−1)) |
| avg_clustering | Mean local clustering coefficient over all nodes in the ego net |
| ego_clustering | Local clustering coefficient of the ego node itself |
| transitivity | Global clustering (ratio of triangles to triads) |
| avg_shortest_path | Mean over all pairs; ∞ if ego net is disconnected |
| diameter | Longest shortest path within ego net |
| ego_degree | Degree of ego in the full graph G |
| ego_betweenness | Betweenness of ego within its own ego network |
| ego_closeness | Closeness of ego within its own ego network |
| n_triangles | Number of triangles containing the ego node |
| effective_size | \|alters\| − (avg degree in alter subgraph) — Burt (1992) |
| efficiency | effective_size / \|alters\| |

**Structural holes (Burt 1992):**

- The **alter subgraph** is the induced subgraph on the ego's neighbours (excluding the ego itself).
- **Redundancy** = average degree of nodes in the alter subgraph = measure of how interconnected the alters are.
- **Effective size** = \|alters\| − redundancy: high value means the ego connects otherwise-unconnected groups (information broker); low value means the ego is embedded in a tightly-knit clique.

**Visualisation — Sunflower ego network layout:**

- Ego node at centre (0, 0).
- Community detection (greedy modularity) run on the ego network.
- Each community's centroid is placed on an orbit ring of radius R = 9.0, evenly spaced in angle.
- Nodes within each community are arranged in concentric rings around the centroid (ring spacing 0.65; capacity per ring = ⌊2πr / 0.6⌋).
- Node size encodes degree: size = 35 + 380 × (deg / max_deg)^0.6.
- Intra-community edges sampled to ≤400 per community; ego-spoke edges sampled to ≤300 total.
- Top-2 hubs per community labelled.

**Output artefacts:**

- `figures/ego_network_node{N}.png`
- `figures/ego_heatmap.png`
- `data/processed/ego_analysis.csv`

---

## Phase 4 — Synthetic Network Comparison (`src/synthetic_networks.py`)

**Purpose:** Establish null-model baselines by comparing the Facebook graph against two canonical random graph models matched to the same scale.

**Barabási-Albert (BA) model:**

- Parameters: n = 4,039; m = 4 (edges added per new node, `BA_M` in config).
- Generated via `nx.barabasi_albert_graph(n, m, seed=42)`.
- Expected degree distribution: power-law with γ ≈ 3.

**Erdős-Rényi (ER) model:**

- Parameters: n = 4,039; p = 2m_real / (n(n−1)) (density-matched to the real graph).
- Generated via `nx.erdos_renyi_graph(n, p, seed=42)`.
- Expected degree distribution: Poisson; low clustering.

**Comparison metrics (via `plot_similarity_table`):**

| Metric | Description |
| --- | --- |
| Nodes, Edges | Raw graph size |
| Avg Degree | 2m/n |
| Density | 2m / (n(n−1)) |
| Avg Clustering | Mean local clustering coefficient |
| Transitivity | Global clustering ratio |
| Diameter (LCC) | Diameter of largest connected component |
| Avg Shortest Path | Sampled over 500 random source nodes |
| Assortativity | `nx.degree_assortativity_coefficient` |
| Power-law gamma | Log-log linear regression slope |
| Jaccard Hub-50 | Jaccard overlap of top-50 degree nodes vs. Real |

**Output artefacts:**

- `figures/synthetic_comparison.png`
- `figures/baseline_boxplots.png`
- `figures/similarity_table.png`
- `data/processed/edges_ba.csv`
- `data/processed/edges_er.csv`

---

## Phase 5 — Information Diffusion (`src/diffusion.py`)

**Model:** Independent Cascade (IC) — Kempe et al. (2003).

**Parameters:**

- Seed nodes: Top-3 highest-degree nodes (top ego candidates).
- Propagation probability: p = 0.1 (each edge independently transmits with 10% chance per step).
- Maximum steps: 100.
- Random seed: 42 for reproducibility.

**Algorithm:**

```text
activated  ← {seed nodes}
new_active ← {seed nodes}

repeat:
    candidates ← neighbours of new_active not in activated
    new_active  ← {v in candidates : Uniform(0,1) < p}
    activated  ← activated ∪ new_active
until new_active is empty or max_steps reached
```

**Outputs:** Step-by-step count of activated nodes (cumulative); fraction of total nodes reached at termination.

**Output artefacts:**

- `figures/diffusion_spread.png`
- `data/processed/diffusion_results.csv`

---

## Implementation Notes

- **Backend:** `matplotlib.use("Agg")` — non-interactive PNG generation; safe for headless script execution.
- **Reproducibility:** All stochastic operations seeded with `RANDOM_SEED = 42`.
- **Python version:** 3.9+; key dependencies: NetworkX ≥ 3.0, matplotlib ≥ 3.5, pandas, numpy.
- **Large-graph handling:** Betweenness centrality uses k=500 random pivots for approximation (exact computation is O(nm) which is ~350 M operations on this graph). Community graph visualisation subsamples to top-1,200 highest-degree nodes when the full graph exceeds that count.

---

*Methodology document — MITS-AI Sem-2 Complex Networks project, 2026.*
