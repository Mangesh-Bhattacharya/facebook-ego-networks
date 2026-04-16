import networkx as nx
import pandas as pd
import os

def generate_ba_network(n, m, seed=None):
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G

def generate_er_network(n, p, seed=None):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G

def save_synthetic_edges(G, filename):
    edges_df = nx.to_pandas_edgelist(G)
    edges_df.to_csv(filename, index=False)
