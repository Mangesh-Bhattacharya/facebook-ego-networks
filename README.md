# Facebook Ego-Network Analysis

A five-phase Python pipeline for analysing the SNAP Facebook combined ego-network dataset.
Covers global graph metrics, ego-network structural-hole analysis, synthetic null-model comparison,
and information-diffusion simulation.

---

## Dataset

| Property | Value |
| --- | --- |
| Source | Stanford Network Analysis Project (SNAP) |
| File | `data/raw/facebook_combined.csv` |
| Nodes | 4,039 |
| Edges | 88,234 |
| Type | Undirected, unweighted friendship graph |

Download from [SNAP](http://snap.stanford.edu/data/ego-Facebook.html) and place as
`data/raw/facebook_combined.csv` (space- or comma-separated edge list).

---

## Project Structure

```text
facebook-ego-networks/
├── main.py                        # Pipeline entry point
├── src/
│   ├── config.py                  # Paths and constants
│   ├── data_loading.py            # Auto-format CSV loader
│   ├── cleaning.py                # Canonicalise + deduplicate edges
│   ├── graph_builder.py           # Build nx.Graph; extract ego subgraphs
│   ├── metrics.py                 # Global metrics, centrality, community detection
│   ├── ego_analysis.py            # Per-ego structural hole metrics
│   ├── synthetic_networks.py      # BA and ER null-model generators
│   ├── diffusion.py               # Independent Cascade simulation
│   └── visualization.py           # All figures
├── data/
│   ├── raw/                       # facebook_combined.csv (not tracked)
│   └── processed/                 # Generated CSVs
├── figures/                       # Generated PNG figures
├── reports/
│   ├── 01_literature_review_notes.md
│   ├── 02_methodology_description.md
│   ├── 03_results_and_discussion.md
│   └── final_report.tex
└── environments/
    └── requirements.txt
```

---

## Setup

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r environments/requirements.txt
```

**Key dependencies** (pinned in `requirements.txt`):

| Package | Version |
| --- | --- |
| networkx | 3.3 |
| pandas | 2.2.2 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |

---

## Usage

```bash
python main.py
```

The pipeline runs all five phases sequentially and prints a progress log to stdout.
All outputs are written to `data/processed/` (CSV) and `figures/` (PNG).

---

## Pipeline Phases

### Phase 1 — Data Cleaning

- Canonicalises edge direction (min node ID first) to deduplicate undirected pairs.
- Removes self-loops.
- Saves `data/processed/edges_clean.csv` and `data/processed/degree_frequency.csv`.

### Phase 2 — Global Graph Metrics

| Metric | Value |
| --- | --- |
| Average degree | 43.69 |
| Average clustering coefficient | 0.6055 |
| Diameter | 8 |
| Density | 0.0108 |

- Computes degree, betweenness (k=500 pivot approximation), closeness, and eigenvector centrality.
- Detects communities via greedy modularity (CNM) and asynchronous label propagation.
- Produces `figures/degree_distribution.png` and `figures/degree_dist_loglog.png`.

### Phase 3 — Ego Network Analysis

- Selects the top-10 nodes by degree as ego centres.
- Computes per-ego: density, clustering, transitivity, diameter, effective size (Burt 1992), and efficiency.
- Produces a sunflower-layout visualisation for the highest-degree ego (`figures/ego_network_node107.png`)
  and a comparison heatmap (`figures/ego_heatmap.png`).

### Phase 4 — Synthetic Network Comparison

Generates two null-model graphs at the same scale and compares 10 structural metrics:

| Model | Parameters |
| --- | --- |
| Barabási-Albert (BA) | n=4,039, m=4 edges per new node |
| Erdős-Rényi (ER) | n=4,039, p matched to real density |

Key finding: real clustering (0.606) is ~55× higher than either null model, confirming the
small-world property absent from BA and ER.

Produces `figures/synthetic_comparison.png`.

### Phase 5 — Information Diffusion (Independent Cascade)

- Seeds: top-3 degree hubs (nodes 107, 1684, 3437).
- Propagation probability: p = 0.1 per edge per step.
- Typical outcome: ~80–95% of the network activated within 10–15 steps.

Produces `figures/diffusion_spread.png` and `data/processed/diffusion_results.csv`.

---

## Key Findings

| Finding | Evidence |
| --- | --- |
| Scale-free degree distribution | Power-law fit γ ≈ 1.8–2.5 on log-log degree plot |
| Small-world structure | Clustering 0.606, diameter 8 |
| Strong community structure | Modularity Q > 0.4 (CNM) |
| Information broker hubs | Node 107: 1,045 connections, large effective size |
| BA model misses clustering | Average clustering 55× too low |
| ER model misses hubs | Poisson degree distribution, no community structure |
| Efficient hub-seeded diffusion | ~80–95% reach at p=0.1 from top-3 hubs |

---

## Configuration

All paths and constants are in `src/config.py`:

| Constant | Default | Description |
| --- | --- | --- |
| `EGO_NODE` | `0` | Default ego centre for single-ego extraction |
| `BA_M` | `4` | Edges per new node in the BA model |
| `RANDOM_SEED` | `42` | Seed for all stochastic operations |
| `DIFFUSION_P` | `0.1` | Per-edge transmission probability (IC model) |

---

## Reports

| File | Contents |
| --- | --- |
| `reports/01_literature_review_notes.md` | Annotated bibliography (McAuley & Leskovec 2012, Barabási & Albert 1999, Watts & Strogatz 1998, Burt 1992, Kempe et al. 2003, and others) |
| `reports/02_methodology_description.md` | Full pipeline methodology with algorithm pseudocode, parameter tables, and output artefact lists |
| `reports/03_results_and_discussion.md` | Phase-by-phase results and discussion grounded in the computed metrics |
| `reports/final_report.tex` | LaTeX source for the final project report |

---

## References

- McAuley, J. & Leskovec, J. (2012). *Learning to Discover Social Circles in Ego Networks.* NIPS.
- Barabási, A.-L. & Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science.
- Watts, D.J. & Strogatz, S.H. (1998). *Collective Dynamics of 'Small-World' Networks.* Nature.
- Burt, R.S. (1992). *Structural Holes: The Social Structure of Competition.* Harvard University Press.
- Kempe, D., Kleinberg, J. & Tardos, É. (2003). *Maximizing the Spread of Influence through a Social Network.* KDD.
- Clauset, A., Newman, M.E.J. & Moore, C. (2004). *Finding Community Structure in Very Large Networks.* Physical Review E.

---

*MITS-AI Sem-2 Complex Networks project, 2025.*
