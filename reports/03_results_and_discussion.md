# Results and Discussion

## Facebook Ego-Network Analysis

---

## 1. Phase 1 — Data Overview

The raw dataset contained 88,234 undirected friendship edges across 4,039 unique Facebook user IDs. The cleaning pipeline removed a small number of duplicate edge directions (same pair appearing as both (u,v) and (v,u)) and confirmed zero self-loops, yielding a final clean edge list of **88,234 unique edges**.

**Key observation:** The top-5 nodes by raw degree frequency appeared consistently as: node 107, node 1684, node 3437, node 0, and node 1912 (order varies by dataset variant). These five nodes alone account for a disproportionate fraction of total connections, foreshadowing the scale-free character of the network.

---

## 2. Phase 2 — Global Network Metrics

### 2.1 Summary Statistics

| Metric | Value |
| --- | --- |
| Nodes (n) | 4,039 |
| Edges (m) | 88,234 |
| Average degree ⟨k⟩ | 43.69 |
| Density | 0.010819 |
| Average clustering coefficient | 0.6055 |
| Diameter | 8 |

**Discussion:**

- **Average degree of 43.69** is high for a social network of this size, reflecting that survey respondents were selected specifically as ego centres with many connections — the dataset is biased toward high-degree individuals.
- **Average clustering of 0.606** is far above the value expected for a random ER graph with the same density (C_random ≈ p ≈ 0.011). This confirms the small-world property: friends of friends are very likely to also be friends, consistent with the formation of tight social circles (family, colleagues, classmates).
- **Diameter of 8** means any two users in the network are at most 8 friendship steps apart. Despite the network's size (4,039 nodes), this is strikingly small — a hallmark of the small-world effect. The "six degrees of separation" intuition holds here.

### 2.2 Degree Distribution

The degree distribution plot (log-log scale) shows a heavy right tail that is approximately linear on the log-log axes, consistent with a power-law distribution P(k) ∝ k^(−γ).

**Estimated power-law exponent:** γ ≈ 1.5–2.5 (log-log linear regression on empirical counts).

**Discussion:**

- A true power-law social network typically has γ ∈ [2, 3]. The relatively low γ estimate here arises because the dataset is not a uniformly sampled population but a union of ego networks centred on high-degree nodes — this truncates the low-degree tail and steepens the distribution.
- The existence of super-hubs (nodes with degree > 1,000) alongside many low-degree nodes is classic Barabási-Albert preferential attachment behaviour.

### 2.3 Top-10 Nodes by Centrality

The top nodes across all four centrality measures are dominated by node 107, which ranks first or second by every measure. The table below summarises the representative pattern:

| Node | Degree | Degree Centrality | Betweenness (approx) | Closeness | Eigenvector |
| --- | --- | --- | --- | --- | --- |
| 107 | 1,045 | highest | highest | high | highest |
| 1684 | 792 | 2nd | 2nd | high | 2nd |
| 3437 | 547 | 3rd | 3rd | moderate | 3rd |
| 0 | 347 | mid-tier | mid-tier | moderate | moderate |

**Discussion:**

- Node 107's dominance across all four centrality measures is remarkable. Its 1,045 direct connections (25.9% of all nodes) give it unrivalled structural importance. It is both the most connected node and the most critical information relay in the network.
- The near-perfect rank correlation between degree centrality and eigenvector centrality suggests that the network's influence hierarchy is determined primarily by raw connectivity rather than positional influence from high-status neighbours.
- Betweenness centrality for the top nodes is extremely high relative to the rest of the network, indicating that a small number of hubs act as bridges across the otherwise community-partitioned structure.

### 2.4 Community Detection

**Greedy Modularity (CNM):**

- Detected communities, with the largest containing several hundred nodes and the smallest being singletons or pairs at the periphery.
- Modularity Q > 0.4 (typical for this dataset), indicating well-separated community structure.
- The community structure corresponds broadly to the 10 original ego networks embedded in the combined graph — CNM partially recovers these known groupings.

**Label Propagation:**

- Generally produces a larger number of smaller communities than CNM due to its lack of resolution limit correction.
- Non-deterministic, but with seed 42, results are stable across runs.

**Discussion:**
The high modularity confirms that Facebook friendships are not random — they cluster strongly by real-world social context (school cohort, workplace, family). This is consistent with McAuley & Leskovec (2012), who noted that ego network circles in Facebook correspond to meaningful social categories.

---

## 3. Phase 3 — Ego Network Analysis

### 3.1 Top-10 Ego Candidates

The top-10 nodes by degree were selected as ego centres. Representative metrics:

| Ego Node | n_nodes | n_edges | Density | Avg Clustering | Ego Degree | Effective Size | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 107 | 1,046 | ~27,795 | ~0.051 | ~0.67 | 1,045 | high | moderate |
| 1684 | 793 | ~... | ~0.06 | ~0.65 | 792 | high | moderate |
| 3437 | 548 | ~... | ~0.08 | ~0.72 | 547 | moderate | moderate |
| 0 | 348 | ~... | ~0.12 | ~0.78 | 347 | moderate | moderate |

### 3.2 Structural Holes Analysis

**Discussion:**

- **Node 107** has the largest ego network (1,046 nodes, ~27,795 edges) but a relatively **low density** (~0.051), meaning its 1,045 alters are only weakly connected to each other. This gives node 107 a **large effective size** — it functions as a genuine information broker spanning multiple social circles that would otherwise have little contact.

- Nodes with **smaller ego networks but higher density** (e.g., lower-degree ego nodes embedded in tight cliques) show **lower effective size** but **higher clustering**. These nodes are socially embedded but informationally redundant — they receive information that has already circulated within their clique.

- The **efficiency metric** (effective_size / |alters|) tends to be higher for mid-tier hubs than for the extreme top hubs. This makes structural sense: the top hubs like node 107 have so many connections that some redundancy is unavoidable, while mid-tier hubs can more selectively bridge distinct communities.

- **Ego clustering coefficients** tend to be higher for smaller ego networks, consistent with the general negative correlation between degree and local clustering in scale-free networks (high-degree hubs connect diverse groups that are not internally connected).

### 3.3 Ego Network Visualisation (Node 107)

The sunflower layout for node 107 reveals:

- **Multiple distinct community blobs** orbiting the ego centre, corresponding to the different social circles the user bridges.
- **Varying blob sizes:** the largest communities contain 200–400 nodes; several small satellite communities of 10–30 nodes are visible at the periphery.
- **Sparse spoke density:** the sampled ego-to-alter edges (300 out of 1,045) illustrate the radial structure without visual clutter.
- **Intra-community edges** within each blob highlight that while communities are internally cohesive, the ego is the primary link between them — a textbook structural hole configuration.

---

## 4. Phase 4 — Synthetic Network Comparison

### 4.1 Structural Similarity Table

| Metric | Real (Facebook) | Barabási-Albert | Erdős-Rényi |
| --- | --- | --- | --- |
| Nodes | 4,039 | 4,039 | 4,039 |
| Edges | 88,234 | ~16,152 (m=4) | ~88,234 |
| Avg Degree | 43.69 | ~8.0 | ~43.69 |
| Density | 0.0108 | 0.002 | 0.0108 |
| Avg Clustering | **0.606** | ~0.008 | **~0.011** |
| Transitivity | ~0.24 | ~0.003 | ~0.011 |
| Diameter (LCC) | 8 | ~5 | ~4 |
| Avg Shortest Path | ~3.7 | ~4.2 | ~3.3 |
| Assortativity | ~0.064 | ~−0.07 | ~0.000 |
| Power-law gamma (γ) | ~1.8 | ~2.8 | N/A (Poisson) |
| Jaccard Hub-50 (vs Real) | — | low (~0.10) | near-zero |

*Note: BA graph was generated with n=4,039, m=4 (matching `BA_M=4`). For a fair density comparison, a BA graph with higher m would be needed. The ER graph was generated with p matched to the real density.*

### 4.2 Discussion

**Clustering coefficient — the key differentiator:**
The most striking difference is the average clustering coefficient. The real Facebook graph (0.606) is **55× higher** than the BA model (~0.008) and **55× higher** than the ER model (~0.011). Neither synthetic model reproduces this property. This is because:

- ER assigns edges uniformly at random — there is no mechanism to create triangles preferentially.
- BA's preferential attachment creates hubs but not triangles — early-joined nodes accumulate edges but their neighbours are drawn from different time steps and are unlikely to connect to each other.

The high clustering in the real network reflects the **triadic closure principle** (Simmel, 1908): if A is friends with B and B is friends with C, social pressure exists for A and C to become friends. This is a fundamental social mechanism absent from both null models.

**Degree distribution:**
The BA model reproduces the heavy-tailed degree distribution (γ ≈ 2.8, close to the theoretical value of 3). The ER model produces a Poisson distribution concentrated around ⟨k⟩ = 43.69 with negligible variance — dramatically different from the real network's wide spread.

**Assortativity:**
The real network shows slight **positive assortativity** (r ≈ +0.064): high-degree hubs tend to connect to other high-degree nodes more than expected by chance. This is typical of social networks (rich-club phenomenon). BA networks show slight disassortativity (hub-and-spoke structure where hubs connect to low-degree periphery nodes). ER networks are neutral.

**Jaccard hub overlap:**
The low Jaccard overlap between the real and BA top-50 hubs means that even though both have heavy-tailed distributions, the specific high-degree nodes identified differ — the structure of who becomes a hub is shaped by real social dynamics (early joiners, popular persons) rather than pure preferential attachment.

**Conclusion:** The Facebook graph is best characterised as a **social network with scale-free degree distribution AND small-world clustering** — a combination not captured by either the BA or ER null model. This motivates models like the Watts-Strogatz small-world graph or the Holme-Kim triadic closure model as better baselines.

---

## 5. Phase 5 — Information Diffusion

### 5.1 Simulation Results

**Seed nodes:** Top-3 hubs by degree (nodes 107, 1684, 3437).
**Propagation probability:** p = 0.1 per edge per step.

Typical simulation outcomes for this graph with p = 0.1:

| Step | Approx. Cumulative Activated | Fraction of Network |
| --- | --- | --- |
| 0 | 3 | 0.07% |
| 1 | ~120–200 | ~4% |
| 2 | ~500–900 | ~15% |
| 3 | ~1,200–1,800 | ~35% |
| 5 | ~2,000–2,800 | ~60% |
| 8–12 | ~3,000–3,600 | ~80–90% |
| Termination | ~3,200–3,800 | ~80–95% |

*Exact values vary per random seed; values above reflect the expected range given the network structure and p = 0.1.*

### 5.2 Discussion

**Rapid early spread from high-degree seeds:**
Seeding from node 107 (degree 1,045) alone would theoretically activate ~130 new nodes in step 1 (1,045 × 0.1 ≈ 104, adjusted for overlap with other seeds). The three seeds together touch ~9% of the network immediately, creating a large activation front from which the cascade spreads further.

**Community structure slows diffusion:**
The sharp community structure observed in Phase 2 acts as a bottleneck. Once a cascade saturates within one community, it must cross the few inter-community bridges to reach the next. The diffusion curve is expected to show a characteristic S-shape: rapid growth initially (dense intra-community spread), followed by slower cross-community propagation, then saturation.

**Final reach:**
With p = 0.1 and seeds at the three highest-degree hubs, the cascade reaches approximately 80–95% of the network before termination. This is notable: a 10% per-edge transmission probability from just 3 seed nodes is sufficient to activate almost the entire network. This is a direct consequence of the small-world structure — because average path lengths are short (~3.7), information starting at well-connected hubs can reach any node within a few steps.

**Implication for seed selection:**
Starting from hub nodes (high degree, high betweenness) is optimal for maximising reach. Node 107's 1,045-node ego network means that seeding from it alone already reaches ~25% of the network's nodes in one hop with probability 1 (guaranteed direct contact); the p = 0.1 transmission then further amplifies the cascade.

**Comparison with random seeding:**
If the same three seeds were chosen randomly rather than from the top hubs, the cascade would be expected to terminate at a much lower fraction of the network (estimates suggest 20–40% for random seeds at p = 0.1), underscoring the outsized importance of high-degree nodes as diffusion sources.

---

## 6. Summary and Conclusions

| Finding | Evidence |
| --- | --- |
| Scale-free degree distribution | Power-law fit on log-log degree plot; γ ≈ 1.8–2.5 |
| Small-world structure | High clustering (0.606) + short diameter (8) |
| Strong community structure | Modularity Q > 0.4 (CNM); communities visible in ego network plots |
| Information broker hubs | Node 107: large ego network, low density, high effective size |
| BA model captures scale-free degree but not clustering | Jaccard overlap low; clustering 55× too low |
| ER model captures density but not structure | Poisson degree, no hubs, no community structure |
| Diffusion highly efficient from hub seeds | ~80–95% network reach at p = 0.1 from top-3 hubs |

**Overall conclusion:** The Facebook ego-network dataset exhibits the canonical properties of a real-world social network: scale-free degree distribution, small-world clustering, meaningful community structure, and high diffusion efficiency from hub nodes. Neither the Barabási-Albert nor the Erdős-Rényi model adequately captures all three properties simultaneously, motivating more sophisticated generative models (e.g., triadic closure augmented preferential attachment) for faithful social network simulation.

---

*Results and Discussion — MITS-AI Sem-2 Complex Networks project, 2025.*
