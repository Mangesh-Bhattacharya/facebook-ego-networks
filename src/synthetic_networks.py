import networkx as nx
import pandas as pd
import os

def generate_ba_network(n, m, seed=None):
    """Generate Barabasi-Albert network with n nodes, m edges per new node"""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G

def generate_er_network(n, p, seed=None):
    """Generate Erdos-Renyi network with n nodes, probability p"""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G

def save_synthetic_edges(G, filename):
    """Save graph edges to CSV file"""
    edges_df = nx.to_pandas_edgelist(G)
    edges_df.to_csv(filename, index=False)
