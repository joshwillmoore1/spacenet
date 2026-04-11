from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from numba import njit

@njit
def aggregate_neighbor_communities_numba(
    neighbor_labels: np.ndarray,
    weights: np.ndarray,
    community_a: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    same_sum = 0.0
    unique_labels = np.empty(neighbor_labels.shape[0], dtype=np.int64)
    unique_weights = np.empty(weights.shape[0], dtype=np.float64)
    n_unique = 0

    for idx in range(neighbor_labels.shape[0]):
        label = int(neighbor_labels[idx])
        weight = float(weights[idx])

        if label < 0:
            continue
        if label == community_a:
            same_sum += weight
            continue

        found = False
        for j in range(n_unique):
            if unique_labels[j] == label:
                unique_weights[j] += weight
                found = True
                break
        if not found:
            unique_labels[n_unique] = label
            unique_weights[n_unique] = weight
            n_unique += 1

    return same_sum, unique_labels[:n_unique], unique_weights[:n_unique]
