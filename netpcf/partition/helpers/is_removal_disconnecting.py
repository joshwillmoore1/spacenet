
from typing import Any, List, Dict
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
from netpcf.partition.helpers.is_removal_disconnecting_numba import is_removal_disconnecting_numba
import numpy as np



def is_removal_disconnecting(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    community: int,
    node_idx: int,
    community_size: np.ndarray,
) -> bool:
    if backend.visit_token == np.iinfo(backend.visit_marks.dtype).max:
        backend.visit_marks.fill(0)
        backend.visit_token = 0
    backend.visit_token += 1
    return bool(
        is_removal_disconnecting_numba(
            backend.indptr,
            backend.indices,
            label_arr,
            int(community),
            int(node_idx),
            community_size,
            backend.visit_marks,
            int(backend.visit_token),
        )
    )