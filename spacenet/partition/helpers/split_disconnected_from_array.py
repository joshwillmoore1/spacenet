from typing import Any, List
from spacenet.partition.helpers.DistanceBackend import DistanceBackend
from spacenet.partition.helpers.community_ids import community_ids
import numpy as np



def split_disconnected_from_array(backend: DistanceBackend, label_arr: np.ndarray) -> np.ndarray:
    if label_arr.size == 0:
        return label_arr

    current_max = int(label_arr.max(initial=-1))
    for community in sorted(int(c) for c in community_ids(label_arr)):
        member_mask = label_arr == community
        members = np.flatnonzero(member_mask)
        if members.size <= 1:
            continue

        visited = np.zeros(label_arr.shape[0], dtype=bool)
        components: List[np.ndarray] = []

        for start in members:
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            component: List[int] = []

            while stack:
                node = stack.pop()
                component.append(node)
                start_ptr = backend.indptr[node]
                end_ptr = backend.indptr[node + 1]
                for nbr in backend.indices[start_ptr:end_ptr]:
                    nbr = int(nbr)
                    if member_mask[nbr] and not visited[nbr]:
                        visited[nbr] = True
                        stack.append(nbr)

            components.append(np.asarray(component, dtype=np.int64))

        if len(components) <= 1:
            continue

        for component in components[1:]:
            current_max += 1
            label_arr[component] = current_max

    return label_arr