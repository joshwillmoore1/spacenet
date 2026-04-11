import random
from typing import Any, List, Optional
from spacenet.partition.helpers.DistanceBackend import DistanceBackend
import numpy as np
import networkx as nx   

def seed_farthest_first(
    G: nx.Graph,
    k: int,
    dist_attr: str,
    rng: random.Random,
    backend: Optional[DistanceBackend] = None,
) -> List[Any]:
    del rng
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)

    if not backend.idx_to_node:
        return []

    incidence = backend.weighted_degree.copy()
    incidence[backend.degree == 0] = np.inf
    first_idx = int(np.argmin(incidence))
    seed_indices = [first_idx]
    min_dist = backend.distances_from_index(first_idx)

    while len(seed_indices) < k:
        candidate_idx = int(np.argmax(min_dist))
        seed_indices.append(candidate_idx)
        min_dist = np.minimum(min_dist, backend.distances_from_index(candidate_idx))

    return [backend.idx_to_node[idx] for idx in seed_indices]