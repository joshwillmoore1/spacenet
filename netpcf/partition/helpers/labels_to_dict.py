from typing import Any, Dict
from netpcf.partition.helpers.DistanceBackend import DistanceBackend
import numpy as np

def labels_to_dict(label_arr: np.ndarray, backend: DistanceBackend) -> Dict[Any, int]:
    return {
        backend.idx_to_node[idx]: int(label_arr[idx])
        for idx in range(len(label_arr))
        if label_arr[idx] >= 0
    }