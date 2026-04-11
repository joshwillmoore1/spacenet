
from typing import Any, List, Dict
from spacenet.partition.helpers.DistanceBackend import DistanceBackend
from spacenet.partition.helpers.community_volume_array import community_volume_array
from spacenet.partition.helpers.is_removal_disconnecting import is_removal_disconnecting
from spacenet.partition.helpers.aggregate_neighbor_communities_numba import aggregate_neighbor_communities_numba
import numpy as np
import random
import math


def local_move_pass_from_array(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    medoids: Dict[int, Any],
    distmaps: Dict[int, np.ndarray],
    T: float,
    alpha: float,
    p: int,
    rng: random.Random,
) -> int:
    node_indices = list(range(len(backend.idx_to_node)))
    rng.shuffle(node_indices)

    volumes = community_volume_array(backend, label_arr)
    community_size = np.bincount(label_arr[label_arr >= 0], minlength=max(1, len(volumes))).astype(np.int64)
    moved = 0

    for node_idx in node_indices:
        community_a = int(label_arr[node_idx])
        if community_a < 0:
            continue

        start_ptr = backend.indptr[node_idx]
        end_ptr = backend.indptr[node_idx + 1]
        neighbors = backend.indices[start_ptr:end_ptr]
        weights = backend.data[start_ptr:end_ptr]
        neighbor_labels = label_arr[neighbors]
        sum_to_a, candidate_labels, candidate_weights = aggregate_neighbor_communities_numba(
            neighbor_labels,
            weights,
            int(community_a),
        )
        if candidate_labels.size == 0:
            continue

        if is_removal_disconnecting(backend, label_arr, community_a, node_idx, community_size):
            continue

        dist_a = distmaps.get(community_a)
        if dist_a is None:
            medoid_idx = backend.node_to_idx[medoids[community_a]]
            dist_a = backend.distances_from_index(medoid_idx)
            distmaps[community_a] = dist_a
        da = float(dist_a[node_idx])
        if not math.isfinite(da):
            medoid_idx = backend.node_to_idx[medoids[community_a]]
            dist_a = backend.distances_from_index(medoid_idx)
            distmaps[community_a] = dist_a
            da = float(dist_a[node_idx])

        best_gain = 0.0
        best_comm = None
        best_to_b = 0.0

        volume_a = float(volumes[community_a])
        for idx in range(candidate_labels.shape[0]):
            community_b = int(candidate_labels[idx])
            to_b = float(candidate_weights[idx])
            if to_b <= 0.0:
                continue

            volume_b = float(volumes[community_b])
            d_vol = float(alpha) * (
                (volume_a - to_b - T) ** 2
                - (volume_a - T) ** 2
                + (volume_b + sum_to_a - T) ** 2
                - (volume_b - T) ** 2
            )

            dist_b = distmaps.get(community_b)
            if dist_b is None:
                medoid_idx = backend.node_to_idx[medoids[community_b]]
                dist_b = backend.distances_from_index(medoid_idx)
                distmaps[community_b] = dist_b
            db = float(dist_b[node_idx])

            if p == 1:
                d_comp = float(1 - alpha) * (db - da)
            else:
                d_comp = float(1 - alpha) * (db**p - da**p)

            d_obj = d_vol + d_comp
            if d_obj < best_gain:
                best_gain = d_obj
                best_comm = community_b
                best_to_b = to_b

        if best_comm is None:
            continue

        label_arr[node_idx] = best_comm
        volumes[community_a] -= best_to_b
        volumes[best_comm] += sum_to_a
        community_size[community_a] -= 1
        community_size[best_comm] += 1
        moved += 1

    return moved