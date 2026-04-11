import networkx as nx   
from netpcf.partition.helpers.edge_len import edge_len

def total_edge_length(G: nx.Graph, dist: str) -> float:
    return sum(edge_len(G, u, v, dist) for u, v in G.edges())