import numpy as np

def prepare_reachable_edge_arrays(node_list: np.ndarray, node_distances: np.ndarray, node_to_edges: dict):
    """Build per-source edge arrays once so each radius only needs boolean masking."""
    node_position = {node: i for i, node in enumerate(node_list)}
    edges_seen = set()
    edge_u_idx = []
    edge_v_idx = []
    edge_weights = []

    for node in node_list:
        for edge, weight in node_to_edges.get(node, []):
            if edge in edges_seen:
                continue
            if edge[0] not in node_position or edge[1] not in node_position:
                continue
            edge_u_idx.append(node_position[edge[0]])
            edge_v_idx.append(node_position[edge[1]])
            edge_weights.append(weight)
            edges_seen.add(edge)

    if not edge_weights:
        empty = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.intp)
        return empty, empty, empty_int, empty_int

    edge_weights = np.asarray(edge_weights, dtype=np.float64)
    edge_u_idx = np.asarray(edge_u_idx, dtype=np.intp)
    edge_v_idx = np.asarray(edge_v_idx, dtype=np.intp)

    return edge_weights, node_distances[edge_u_idx], node_distances[edge_v_idx], edge_u_idx, edge_v_idx
