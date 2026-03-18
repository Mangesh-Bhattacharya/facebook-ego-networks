"""
graph_builder.py
Builds the NetworkX graph from a clean edge-list DataFrame.

Graph Schema
------------
Type       : nx.Graph (undirected, unweighted)
Nodes      : Facebook user IDs (int)
Edges      : Friendship connections
Node attrs : degree (int)
"""
import sys
import os
import networkx as nx
import pandas as pd

try:
    from .config import EGO_NODE
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import EGO_NODE


def build_facebook_graph(edges_df: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(edges_df, source="node1", target="node2", create_using=nx.Graph())
    nx.set_node_attributes(G, dict(G.degree()), name="degree")
    print(f"  Graph built : {G.number_of_nodes():,} nodes | " f"{G.number_of_edges():,} edges")
    return G


def extract_ego_network(G: nx.Graph, ego_node: int = EGO_NODE,
                        radius: int = 1) -> nx.Graph:
    if ego_node not in G:
        raise ValueError(f"Node {ego_node} is not in the graph.")
    ego_net = nx.ego_graph(G, ego_node, radius=radius)
    print(f"  Ego-net node {ego_node} : {ego_net.number_of_nodes()} nodes | "f"{ego_net.number_of_edges()} edges")
    return ego_net
