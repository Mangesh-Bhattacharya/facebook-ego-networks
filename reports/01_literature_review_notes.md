# Literature Review Notes

## Facebook Ego-Network Analysis

---

## 1. Foundational Dataset

**McAuley, J. & Leskovec, J. (2012). Learning to Discover Social Circles in Ego Networks. NIPS.**

- Introduced the SNAP Facebook ego-network dataset used in this project.
- The dataset consists of 10 ego networks collected from Facebook survey participants, merged into a single undirected graph of 4,039 nodes and 88,234 edges.
- Node features (anonymised) and ground-truth social circle labels are provided per ego network.
- Key finding: social circles in ego networks are not disjoint — individuals can simultaneously belong to family, college, and work circles.
- Relevance: Our project uses the combined (`facebook_combined.txt`) graph, treating it as a single social network for macro-level and ego-level analysis.

---

## 2. Scale-Free Networks and Power-Law Degree Distributions

**Barabási, A.-L. & Albert, R. (1999). Emergence of Scaling in Random Networks. Science, 286(5439), 509–512.**

- Proposed the Barabási-Albert (BA) preferential attachment model.
- Networks grown by preferential attachment produce power-law degree distributions: P(k) ∝ k^(−γ), γ ≈ 3.
- Hubs — nodes with exceptionally high degree — arise naturally and dominate connectivity.
- Relevance: The Facebook graph is expected to be approximately scale-free. We fit a power-law exponent γ to the empirical degree distribution (log-log linear regression) and compare against a synthetic BA graph (m = 4 edges per new node).

**Clauset, A., Shalizi, C.R., & Newman, M.E.J. (2009). Power-Law Distributions in Empirical Data. SIAM Review, 51(4), 661–703.**

- Establishes rigorous statistical methods (maximum likelihood estimation, Kolmogorov-Smirnov tests) for fitting power laws.
- Notes that many claimed power laws are actually log-normal or stretched-exponential distributions.
- Relevance: We use log-log linear regression as a first-order estimate of γ; the limitations of this approach are acknowledged.

---

## 3. Small-World Networks

**Watts, D.J. & Strogatz, S.H. (1998). Collective Dynamics of 'Small-World' Networks. Nature, 393, 440–442.**

- Defined small-world networks: high clustering coefficient (like regular lattices) combined with short average path lengths (like random graphs).
- Introduced the clustering coefficient C and average shortest path length L as key diagnostics.
- Small-world criterion: C ≫ C_random and L ≈ L_random.
- Relevance: The Facebook graph exhibits small-world properties (diameter 8, high average clustering ~0.606). We compare these metrics against the Erdős-Rényi baseline.

---

## 4. Ego Networks and Structural Holes

**Burt, R.S. (1992). Structural Holes: The Social Structure of Competition. Harvard University Press.**

- Defined the ego network as the subgraph of a focal node (ego) and all its direct contacts (alters), including edges among alters.
- Introduced **effective size**: the number of non-redundant contacts (alters not connected to each other), measuring the ego's access to structurally diverse information.
  - Effective size = |alters| − avg redundancy of alters
  - Higher effective size → ego bridges more structural holes → greater information brokerage advantage.
- Defined **efficiency** = effective size / |alters| (normalised to [0,1]).
- Relevance: We compute effective size and efficiency for the top-10 ego nodes to identify which high-degree nodes are true brokers vs. densely embedded community members.

**Everett, M. & Borgatti, S.P. (2005). Ego Network Betweenness. Social Networks, 27(1), 31–38.**

- Extended betweenness centrality to the ego network context, showing that betweenness and structural hole metrics are complementary rather than redundant.
- Relevance: We report both betweenness centrality (global) and effective size (local ego-level) for a richer characterisation of hub nodes.

---

## 5. Community Detection

**Newman, M.E.J. & Girvan, M. (2004). Finding and Evaluating Community Structure in Networks. Physical Review E, 69, 026113.**

- Proposed modularity Q as an objective function for community detection.
  - Q = fraction of intra-community edges − expected fraction under a null model.
  - Q ∈ [−1, 1]; values > 0.3 indicate meaningful community structure.
- Introduced edge-betweenness divisive method (Girvan-Newman algorithm).

**Clauset, A., Newman, M.E.J., & Moore, C. (2004). Finding Community Structure in Very Large Networks. Physical Review E, 70, 066111.**

- Proposed the greedy modularity maximisation algorithm (Clauset-Newman-Moore, CNM).
- Merges communities greedily to maximise ΔQ at each step; runs in O(m d log n) time.
- This is the primary community detection algorithm used in this project (`greedy_modularity_communities` in NetworkX).

**Raghavan, U.N., Albert, R., & Kumara, S. (2007). Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks. Physical Review E, 76, 036106.**

- Proposed asynchronous label propagation (LPA) for community detection.
- Each node adopts the label held by the majority of its neighbours, iterated until convergence.
- Near-linear time complexity: O(m); non-deterministic.
- Relevance: We implement LPA as a second community detection method (`asyn_lpa_communities`) to compare against CNM.

---

## 6. Centrality Measures

**Freeman, L.C. (1977). A Set of Measures of Centrality Based on Betweenness. Sociometry, 40(1), 35–41.**

- Formalised betweenness centrality: fraction of shortest paths in the network that pass through a given node.
- High betweenness nodes are critical bridges; their removal can disconnect communities.

**Bonacich, P. (1987). Power and Centrality: A Family of Measures. American Journal of Sociology, 92(5), 1170–1182.**

- Introduced eigenvector centrality: a node's influence is proportional to the sum of its neighbours' centralities.
- Related to Google PageRank.

**Sabidussi, G. (1966). The Centrality Index of a Graph. Psychometrika, 31(4), 581–603.**

- Defined closeness centrality as the reciprocal of the average shortest path to all other nodes.
- Nodes with high closeness can quickly reach all others.

Relevance: We compute degree, betweenness, closeness, and eigenvector centrality for all nodes, identifying top-10 hubs across all four measures.

---

## 7. Information Diffusion

**Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the Spread of Influence through a Social Network. KDD.**

- Formalised the Independent Cascade (IC) model:
  - Each newly activated node independently attempts to activate each of its inactive neighbours with probability p.
  - The process terminates when no further activations occur.
- Proved that the influence spread function is submodular, enabling greedy seed selection with (1 − 1/e) approximation guarantee.
- Relevance: We implement IC with p = 0.1, seeding from the top-3 highest-degree nodes, to simulate information spread across the Facebook graph.

**Watts, D.J. & Dodds, P.S. (2007). Influentials, Networks, and Public Opinion Formation. Journal of Consumer Research, 34(4), 441–458.**

- Challenged the "influentials hypothesis": large cascades are driven more by the susceptibility of the population than by the degree of seed nodes.
- Relevance: Comparing diffusion reach from high-degree seeds versus random seeds would be an interesting extension.

---

## 8. Random Graph Baselines

**Erdős, P. & Rényi, A. (1959). On Random Graphs. Publicationes Mathematicae, 6, 290–297.**

- Defined the G(n, p) random graph model: n nodes, each pair connected independently with probability p.
- Properties: Poisson degree distribution, low clustering, short paths.
- Relevance: We generate an ER baseline matching the Facebook graph's n and edge density p, establishing what a structureless random network would look like.

---

## 9. Summary of Key Metrics Expected

| Property | Expected in Facebook Graph |
| --- | --- |
| Degree distribution | Heavy-tailed (power-law) |
| Average clustering | High (≈ 0.60) |
| Diameter | Small (≈ 8) |
| Modularity | High Q (≥ 0.4) |
| Community structure | Clearly present |
| Assortativity | Slightly positive |

---

*Notes compiled for MITS-AI Sem-2 Complex Networks project, 2025.*
