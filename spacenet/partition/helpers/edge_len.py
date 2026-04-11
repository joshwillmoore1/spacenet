import networkx as nx

def edge_len(G: nx.Graph, u, v, dist: str) -> float:
    return float(G[u][v].get(dist, 1.0))