from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from numba import njit
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


@dataclass
class DistanceBackend:
    node_to_idx: Dict[Any, int]
    idx_to_node: List[Any]
    csr: csr_array
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_w: np.ndarray
    weighted_degree: np.ndarray
    degree: np.ndarray
    visit_marks: np.ndarray
    visit_token: int

    @classmethod
    def from_networkx(cls, G: nx.Graph, weight_attr: str = "Distance") -> "DistanceBackend":
        idx_to_node = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(idx_to_node)}
        n_nodes = len(idx_to_node)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        edge_u: List[int] = []
        edge_v: List[int] = []
        edge_w: List[float] = []

        for u, v, edge_data in G.edges(data=True):
            i = node_to_idx[u]
            j = node_to_idx[v]
            w = float(edge_data.get(weight_attr, 1.0))
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((w, w))
            edge_u.append(i)
            edge_v.append(j)
            edge_w.append(w)

        csr = csr_array((data, (rows, cols)), shape=(n_nodes, n_nodes))
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int64)
        weights = np.asarray(csr.data, dtype=np.float64)
        weighted_degree = np.add.reduceat(weights, indptr[:-1], dtype=np.float64)
        degree = np.diff(indptr)

        return cls(
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            csr=csr,
            indptr=indptr,
            indices=indices,
            data=weights,
            edge_u=np.asarray(edge_u, dtype=np.int64),
            edge_v=np.asarray(edge_v, dtype=np.int64),
            edge_w=np.asarray(edge_w, dtype=np.float64),
            weighted_degree=weighted_degree,
            degree=degree,
            visit_marks=np.zeros(n_nodes, dtype=np.int32),
            visit_token=0,
        )

    def distances_from_index(self, source_idx: int) -> np.ndarray:
        return np.asarray(
            dijkstra(self.csr, directed=False, indices=source_idx, return_predecessors=False),
            dtype=np.float64,
        )

    def distances_from_indices(self, source_indices: List[int]) -> np.ndarray:
        if not source_indices:
            return np.empty((0, len(self.idx_to_node)), dtype=np.float64)
        dist = dijkstra(self.csr, directed=False, indices=source_indices, return_predecessors=False)
        dist = np.asarray(dist, dtype=np.float64)
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
        return dist

    def distmap_from_array(self, dist: np.ndarray) -> Dict[Any, float]:
        finite = np.isfinite(dist)
        return {
            self.idx_to_node[idx]: float(dist[idx])
            for idx in np.flatnonzero(finite)
        }


@njit
def _is_removal_disconnecting_numba(
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


@njit
def _aggregate_neighbor_communities_numba(
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


@dataclass
class LeidenCompactResult:
    labels: Dict
    community_lengths: Dict[int, float]
    community_cut_lengths: Dict[int, float]
    community_volumes: Dict[int, float]
    medoids: Dict[int, object]
    T: float
    k: int
    objective: float
    compactness: float
    volume_penalty: float
    total_cut_length: float
    moves: int
    iterations: int


def _edge_len(G: nx.Graph, u, v, dist: str) -> float:
    return float(G[u][v].get(dist, 1.0))


def _total_edge_length(G: nx.Graph, dist: str) -> float:
    return sum(_edge_len(G, u, v, dist) for u, v in G.edges())


def _labels_to_array(labels: Dict[Any, int], backend: DistanceBackend) -> np.ndarray:
    label_arr = np.full(len(backend.idx_to_node), -1, dtype=np.int64)
    for node, community in labels.items():
        label_arr[backend.node_to_idx[node]] = int(community)
    return label_arr


def _labels_to_dict(label_arr: np.ndarray, backend: DistanceBackend) -> Dict[Any, int]:
    return {
        backend.idx_to_node[idx]: int(label_arr[idx])
        for idx in range(len(label_arr))
        if label_arr[idx] >= 0
    }


def _community_ids(label_arr: np.ndarray) -> np.ndarray:
    return np.unique(label_arr[label_arr >= 0])


def _per_community_W_B_V_from_array(
    backend: DistanceBackend,
    label_arr: np.ndarray,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float], float]:
    communities = _community_ids(label_arr)
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


def _community_volume_array(backend: DistanceBackend, label_arr: np.ndarray) -> np.ndarray:
    communities = _community_ids(label_arr)
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


def _seed_farthest_first(
    G: nx.Graph,
    k: int,
    dist_attr: str,
    rng: random.Random,
    backend: Optional[DistanceBackend] = None,
) -> List[Any]:
    del rng
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)

    if not backend.idx_to_node:
        return []

    incidence = backend.weighted_degree.copy()
    incidence[backend.degree == 0] = np.inf
    first_idx = int(np.argmin(incidence))
    seed_indices = [first_idx]
    min_dist = backend.distances_from_index(first_idx)

    while len(seed_indices) < k:
        candidate_idx = int(np.argmax(min_dist))
        seed_indices.append(candidate_idx)
        min_dist = np.minimum(min_dist, backend.distances_from_index(candidate_idx))

    return [backend.idx_to_node[idx] for idx in seed_indices]


def _voronoi_labels(
    G: nx.Graph,
    seeds: List[Any],
    dist_attr: str,
    backend: Optional[DistanceBackend] = None,
) -> Dict[Any, int]:
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)
    if not seeds:
        return {}

    seed_indices = [backend.node_to_idx[seed] for seed in seeds]
    dist = backend.distances_from_indices(seed_indices)
    assignments = np.argmin(dist, axis=0)
    best_dist = dist[assignments, np.arange(dist.shape[1])]
    finite = np.isfinite(best_dist)

    return {
        backend.idx_to_node[idx]: int(assignments[idx])
        for idx in np.flatnonzero(finite)
    }


def _approx_update_medoids(
    G: nx.Graph,
    labels: Dict[Any, int],
    dist_attr: str,
    samples_per_comm: int = 10,
    rng: Optional[random.Random] = None,
    backend: Optional[DistanceBackend] = None,
):
    if rng is None:
        rng = random.Random(0)
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)

    label_arr = _labels_to_array(labels, backend)
    communities = _community_ids(label_arr)
    medoids: Dict[int, Any] = {}
    distmaps: Dict[int, Dict[Any, float]] = {}

    for community in communities:
        node_indices = np.flatnonzero(label_arr == community)
        if node_indices.size == 0:
            continue

        if node_indices.size <= samples_per_comm:
            sample_indices = node_indices.tolist()
        else:
            sample_indices = rng.sample(node_indices.tolist(), samples_per_comm)

        dist = backend.distances_from_indices(sample_indices)
        costs = np.sum(dist[:, node_indices], axis=1, dtype=np.float64)
        best_pos = int(np.argmin(costs))
        best_idx = int(sample_indices[best_pos])
        best_dist = dist[best_pos]

        medoids[int(community)] = backend.idx_to_node[best_idx]
        distmaps[int(community)] = backend.distmap_from_array(best_dist)

    return medoids, distmaps


def _split_disconnected_from_array(backend: DistanceBackend, label_arr: np.ndarray) -> np.ndarray:
    if label_arr.size == 0:
        return label_arr

    current_max = int(label_arr.max(initial=-1))
    for community in sorted(int(c) for c in _community_ids(label_arr)):
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


def _split_disconnected(
    G: nx.Graph,
    labels: Dict[Any, int],
    backend: Optional[DistanceBackend] = None,
) -> Dict[Any, int]:
    if backend is None:
        backend = DistanceBackend.from_networkx(G)
    label_arr = _labels_to_array(labels, backend)
    label_arr = _split_disconnected_from_array(backend, label_arr)
    return _labels_to_dict(label_arr, backend)


def _compute_objective(
    G: nx.Graph,
    labels: Dict[Any, int],
    distmaps: Dict[int, Dict[Any, float]],
    T: float,
    alpha: float,
    p: int,
    dist_attr: str,
    backend: Optional[DistanceBackend] = None,
):
    if backend is None:
        backend = DistanceBackend.from_networkx(G, dist_attr)

    label_arr = _labels_to_array(labels, backend)
    _, _, volumes, _ = _per_community_W_B_V_from_array(backend, label_arr)
    communities = _community_ids(label_arr)
    vol_pen = sum((volumes.get(int(c), 0.0) - T) ** p for c in communities)

    comp = 0.0
    for community in communities:
        dmap = distmaps.get(int(community), {})
        for node_idx in np.flatnonzero(label_arr == community):
            node = backend.idx_to_node[int(node_idx)]
            du = dmap.get(node, 0.0)
            comp += du if p == 1 else du**p

    return alpha * vol_pen + (1 - alpha) * comp, comp, vol_pen


def _compute_objective_from_arrays(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    distmaps: Dict[int, np.ndarray],
    T: float,
    alpha: float,
    p: int,
) -> tuple[float, float, float]:
    communities = _community_ids(label_arr)
    volumes = _community_volume_array(backend, label_arr)
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


def _is_removal_disconnecting(
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
        _is_removal_disconnecting_numba(
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


def _approx_update_medoids_from_array(
    backend: DistanceBackend,
    label_arr: np.ndarray,
    samples_per_comm: int,
    rng: random.Random,
) -> tuple[Dict[int, Any], Dict[int, np.ndarray]]:
    medoids: Dict[int, Any] = {}
    distmaps: Dict[int, np.ndarray] = {}

    for community in _community_ids(label_arr):
        node_indices = np.flatnonzero(label_arr == community)
        if node_indices.size == 0:
            continue

        if node_indices.size <= samples_per_comm:
            sample_indices = node_indices.tolist()
        else:
            sample_indices = rng.sample(node_indices.tolist(), samples_per_comm)

        dist = backend.distances_from_indices(sample_indices)
        costs = np.sum(dist[:, node_indices], axis=1, dtype=np.float64)
        best_pos = int(np.argmin(costs))
        best_idx = int(sample_indices[best_pos])

        medoids[int(community)] = backend.idx_to_node[best_idx]
        distmaps[int(community)] = dist[best_pos]

    return medoids, distmaps


def _local_move_pass(
    G: nx.Graph,
    labels: Dict[Any, int],
    medoids: Dict[int, object],
    distmaps: Dict[int, Dict[Any, float]],
    T: float,
    alpha: float,
    p: int,
    dist_attr: str,
    rng: random.Random,
) -> int:
    backend = DistanceBackend.from_networkx(G, dist_attr)
    label_arr = _labels_to_array(labels, backend)
    dist_arrays: Dict[int, np.ndarray] = {}
    for community, dmap in distmaps.items():
        dist = np.full(len(label_arr), np.inf, dtype=np.float64)
        for node, value in dmap.items():
            dist[backend.node_to_idx[node]] = float(value)
        dist_arrays[int(community)] = dist

    moved = _local_move_pass_from_array(
        backend=backend,
        label_arr=label_arr,
        medoids=medoids,
        distmaps=dist_arrays,
        T=T,
        alpha=alpha,
        p=p,
        rng=rng,
    )
    labels.clear()
    labels.update(_labels_to_dict(label_arr, backend))
    for community, dist in dist_arrays.items():
        distmaps[community] = backend.distmap_from_array(dist)
    return moved


def _local_move_pass_from_array(
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

    volumes = _community_volume_array(backend, label_arr)
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
        sum_to_a, candidate_labels, candidate_weights = _aggregate_neighbor_communities_numba(
            neighbor_labels,
            weights,
            int(community_a),
        )
        if candidate_labels.size == 0:
            continue

        if _is_removal_disconnecting(backend, label_arr, community_a, node_idx, community_size):
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


def leiden_compact_volume_partition(
    G: nx.Graph,
    k: Optional[int] = None,
    T: Optional[float] = None,
    distance_attr: str = "Distance",
    alpha: float = 0.5,
    p: int = 2,
    max_iter: int = 10,
    seed_samples: int = 10,
    random_seed: Optional[int] = 0,
) -> LeidenCompactResult:
    if G.is_directed():
        raise ValueError("Expected an undirected graph.")

    rng = random.Random(random_seed)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")

    backend = DistanceBackend.from_networkx(G, distance_attr)

    if k is None and T is None:
        k = max(1, min(16, int(round(math.sqrt(max(1, G.number_of_nodes())) / 2))))

    if k is None and T is not None:
        k = max(1, int(round((2.0 * _total_edge_length(G, distance_attr)) / T)))

    if k is not None and T is None:
        seeds = _seed_farthest_first(G, k, distance_attr, rng, backend=backend)
        labels = _voronoi_labels(G, seeds, distance_attr, backend=backend)
        label_arr = _labels_to_array(labels, backend)
        _, _, volumes, _ = _per_community_W_B_V_from_array(backend, label_arr)
        T = sum(volumes.values()) / float(k) if k > 0 else 0.0

    seeds = _seed_farthest_first(G, k, distance_attr, rng, backend=backend)
    labels = _voronoi_labels(G, seeds, distance_attr, backend=backend)
    label_arr = _labels_to_array(labels, backend)
    label_arr = _split_disconnected_from_array(backend, label_arr)

    communities = sorted(int(c) for c in _community_ids(label_arr))
    mapping = {community: idx for idx, community in enumerate(communities)}
    for old, new in mapping.items():
        label_arr[label_arr == old] = new

    medoids, distmaps = _approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)

    total_moves = 0
    iterations = 0
    last_obj, comp, vpen = _compute_objective_from_arrays(backend, label_arr, distmaps, T, alpha, p)

    while iterations < max_iter:
        iterations += 1
        moved = _local_move_pass_from_array(backend, label_arr, medoids, distmaps, T, alpha, p, rng)
        total_moves += moved

        medoids, distmaps = _approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)
        moved += _local_move_pass_from_array(backend, label_arr, medoids, distmaps, T, alpha, p, rng)

        label_arr = _split_disconnected_from_array(backend, label_arr)

        medoids, distmaps = _approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)
        obj, comp, vpen = _compute_objective_from_arrays(backend, label_arr, distmaps, T, alpha, p)

        if moved == 0 or abs(last_obj - obj) < 1e-9:
            last_obj = obj
            break
        last_obj = obj

    labels = _labels_to_dict(label_arr, backend)
    Wfin, Bfin, Vfin, total_cut = _per_community_W_B_V_from_array(backend, label_arr)
    k_final = len(set(labels.values()))

    return LeidenCompactResult(
        labels=labels,
        community_lengths=Wfin,
        community_cut_lengths=Bfin,
        community_volumes=Vfin,
        medoids=medoids,
        T=T,
        k=k_final,
        objective=last_obj,
        compactness=comp,
        volume_penalty=vpen,
        total_cut_length=total_cut,
        moves=total_moves,
        iterations=iterations,
    )
