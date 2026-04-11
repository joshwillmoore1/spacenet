from __future__ import annotations
import numpy as np
from numba import njit

@njit
def is_removal_disconnecting_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    label_arr: np.ndarray,
    community: int,
    node_idx: int,
    community_size: np.ndarray,
    marks: np.ndarray,
    token: int,
) -> bool:
    if community >= community_size.shape[0] or community_size[community] <= 2:
        return False

    start_ptr = indptr[node_idx]
    end_ptr = indptr[node_idx + 1]

    same_neighbor_count = 0
    for pos in range(start_ptr, end_ptr):
        nbr = indices[pos]
        if label_arr[nbr] == community:
            same_neighbor_count += 1

    if same_neighbor_count < 2:
        return False

    same_neighbors = np.empty(same_neighbor_count, dtype=np.int64)
    fill = 0
    for pos in range(start_ptr, end_ptr):
        nbr = indices[pos]
        if label_arr[nbr] == community:
            same_neighbors[fill] = nbr
            fill += 1

    first_neighbor = same_neighbors[0]
    remaining = same_neighbor_count - 1
    stack = np.empty(label_arr.shape[0], dtype=np.int64)
    stack_size = 1
    stack[0] = first_neighbor
    marks[node_idx] = token
    marks[first_neighbor] = token

    while stack_size > 0:
        stack_size -= 1
        node = stack[stack_size]
        row_start = indptr[node]
        row_end = indptr[node + 1]

        for pos in range(row_start, row_end):
            nbr = indices[pos]
            if marks[nbr] == token or label_arr[nbr] != community:
                continue

            marks[nbr] = token

            for idx in range(1, same_neighbor_count):
                if same_neighbors[idx] == nbr:
                    same_neighbors[idx] = -1
                    remaining -= 1
                    break

            if remaining == 0:
                return False

            stack[stack_size] = nbr
            stack_size += 1

    return True