
from typing import Any, List, Dict
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
from netpcf.partition.helpers.community_ids import community_ids
from netpcf.partition.helpers.community_volume_array import community_volume_array
import numpy as np


def compute_objective_from_arrays(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    distmaps: Dict[int, np.ndarray],
    T: float,
    alpha: float,
    p: int,
) -> tuple[float, float, float]:
    communities = community_ids(label_arr)
    volumes = community_volume_array(backend, label_arr)
    vol_pen = sum((float(volumes[int(c)]) - T) ** p for c in communities)

    comp = 0.0
    for community in communities:
        dist = distmaps.get(int(community))
        if dist is None:
            continue
        members = label_arr == community
        vals = dist[members]
        if p == 1:
            comp += float(np.sum(vals, dtype=np.float64))
        else:
            comp += float(np.sum(vals**p, dtype=np.float64))

    return alpha * vol_pen + (1 - alpha) * comp, comp, vol_pen