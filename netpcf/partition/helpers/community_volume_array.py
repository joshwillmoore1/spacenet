import numpy as np  
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
from netpcf.partition.helpers.community_ids import community_ids



def community_volume_array(backend: DistanceBackend, label_arr: np.ndarray) -> np.ndarray:
    communities = community_ids(label_arr)
    if communities.size == 0:
        return np.zeros(0, dtype=np.float64)

    lu = label_arr[backend.edge_u]
    lv = label_arr[backend.edge_v]
    valid = (lu >= 0) & (lv >= 0)
    same = valid & (lu == lv)
    diff = valid & (lu != lv)
    max_label = int(communities.max())
    volumes = np.zeros(max_label + 1, dtype=np.float64)

    if np.any(same):
        np.add.at(volumes, lu[same], backend.edge_w[same])
    if np.any(diff):
        diff_w = backend.edge_w[diff]
        np.add.at(volumes, lu[diff], diff_w)
        np.add.at(volumes, lv[diff], diff_w)

    return volumes