from typing import Any, List, Dict
import random
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
from netpcf.partition.helpers.community_ids import community_ids
import numpy as np



def approx_update_medoids_from_array(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    samples_per_comm: int,
    rng: random.Random,
) -> tuple[Dict[int, Any], Dict[int, np.ndarray]]:
    medoids: Dict[int, Any] = {}
    distmaps: Dict[int, np.ndarray] = {}

    for community in community_ids(label_arr):
        node_indices = np.flatnonzero(label_arr == community)
        if node_indices.size == 0:
            continue

        if node_indices.size <= samples_per_comm:
            sample_indices = node_indices.tolist()
        else:
            sample_indices = rng.sample(node_indices.tolist(), samples_per_comm)

        dist = backend.distances_from_indices(sample_indices)
        costs = np.sum(dist[:, node_indices], axis=1, dtype=np.float64)
        best_pos = int(np.argmin(costs))
        best_idx = int(sample_indices[best_pos])

        medoids[int(community)] = backend.idx_to_node[best_idx]
        distmaps[int(community)] = dist[best_pos]

    return medoids, distmaps