from typing import Any, Dict
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
import numpy as np

def labels_to_array(labels: Dict[Any, int], backend: DistanceBackend) -> np.ndarray:
    label_arr = np.full(len(backend.idx_to_node), -1, dtype=np.int64)
    for node, community in labels.items():
        label_arr[backend.node_to_idx[node]] = int(community)
    return label_arr