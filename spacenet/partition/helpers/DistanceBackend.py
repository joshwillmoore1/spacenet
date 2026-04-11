from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
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