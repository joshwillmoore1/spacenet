from typing import Dict
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
from netpcf.partition.helpers.community_ids import community_ids

import numpy as np



def per_community_W_B_V_from_array(
    backend: DistanceBackend,
    label_arr: np.ndarray,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float], float]:
    communities = community_ids(label_arr)
    if communities.size == 0:
        return {}, {}, {}, 0.0

    lu = label_arr[backend.edge_u]
    lv = label_arr[backend.edge_v]
    valid = (lu >= 0) & (lv >= 0)
    same = valid & (lu == lv)
    diff = valid & (lu != lv)

    max_label = int(communities.max())
    w_arr = np.zeros(max_label + 1, dtype=np.float64)
    b_arr = np.zeros(max_label + 1, dtype=np.float64)

    if np.any(same):
        np.add.at(w_arr, lu[same], backend.edge_w[same])
    if np.any(diff):
        diff_w = backend.edge_w[diff]
        np.add.at(b_arr, lu[diff], diff_w)
        np.add.at(b_arr, lv[diff], diff_w)

    volumes = w_arr + b_arr
    W = {int(c): float(w_arr[c]) for c in communities}
    B = {int(c): float(b_arr[c]) for c in communities}
    V = {int(c): float(volumes[c]) for c in communities}
    total_cut = float(np.sum(backend.edge_w[diff], dtype=np.float64))
    return W, B, V, total_cut