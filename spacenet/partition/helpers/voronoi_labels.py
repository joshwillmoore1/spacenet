from typing import Any, List, Optional,Dict
from spacenet.partition.helpers.DistanceBackend import DistanceBackend
import numpy as np
import networkx as nx   


def voronoi_labels(
    G: nx.Graph,
    seeds: List[Any],
    dist_attr: str,
    backend: Optional[DistanceBackend] = None,
) -> Dict[Any, int]:
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)
    if not seeds:
        return {}

    seed_indices = [backend.node_to_idx[seed] for seed in seeds]
    dist = backend.distances_from_indices(seed_indices)
    assignments = np.argmin(dist, axis=0)
    best_dist = dist[assignments, np.arange(dist.shape[1])]
    finite = np.isfinite(best_dist)

    return {
        backend.idx_to_node[idx]: int(assignments[idx])
        for idx in np.flatnonzero(finite)
    }