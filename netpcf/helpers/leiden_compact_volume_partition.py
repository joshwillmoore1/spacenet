from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx


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


def _per_community_W_B_V(G: nx.Graph, labels: Dict, dist_attr: str):
    W: Dict[int,float] = {}
    B: Dict[int,float] = {}
    total_cut = 0.0
    for u, v in G.edges():
        w = _edge_len(G, u, v, dist_attr)
        cu, cv = labels.get(u), labels.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            W[cu] = W.get(cu, 0.0) + w
        else:
            B[cu] = B.get(cu, 0.0) + w
            B[cv] = B.get(cv, 0.0) + w
            total_cut += w
    V: Dict[int,float] = {}
    for c in set(labels.values()):
        W.setdefault(c, 0.0); B.setdefault(c, 0.0)
        V[c] = W[c] + B[c]
    return W, B, V, total_cut


def _seed_farthest_first(G: nx.Graph, k: int, dist_attr: str, rng: random.Random) -> List:
    nodes = list(G.nodes())
    if not nodes:
        return []
    def inc(n):
        if G.degree(n) == 0: return float("inf")
        return sum(_edge_len(G, n, v, dist_attr) for v in G.neighbors(n))
    first = min(nodes, key=inc)
    seeds = [first]
    min_dist = {n: float("inf") for n in nodes}
    d0 = nx.single_source_dijkstra_path_length(G, first, weight=dist_attr)
    for n, d in d0.items():
        min_dist[n] = min(min_dist[n], d)
    while len(seeds) < k:
        candidate = max(min_dist.items(), key=lambda kv: kv[1])[0]
        seeds.append(candidate)
        d_new = nx.single_source_dijkstra_path_length(G, candidate, weight=dist_attr)
        for n, d in d_new.items():
            if d < min_dist[n]:
                min_dist[n] = d
    return seeds


def _voronoi_labels(G: nx.Graph, seeds: List, dist_attr: str) -> Dict:
    labels = {}
    pq = []
    for cid, s in enumerate(seeds):
        heapq.heappush(pq, (0.0, cid, s))
    seen_best = {}
    while pq:
        d, c, u = heapq.heappop(pq)
        if u in seen_best:
            continue
        seen_best[u] = (d, c)
        for v in G.neighbors(u):
            if v in seen_best: continue
            heapq.heappush(pq, (d + _edge_len(G, u, v, dist_attr), c, v))
    for n, (_, c) in seen_best.items():
        labels[n] = c
    return labels


def _approx_update_medoids(G: nx.Graph, labels: Dict, dist_attr: str, samples_per_comm: int = 10, rng: Optional[random.Random] = None):
    if rng is None: rng = random.Random(0)
    comm_nodes: Dict[int, List] = {}
    for n, c in labels.items():
        comm_nodes.setdefault(c, []).append(n)
    medoids = {}; distmaps = {}
    for c, nodes in comm_nodes.items():
        if not nodes: continue
        S = nodes[:] if len(nodes) <= samples_per_comm else rng.sample(nodes, samples_per_comm)
        best_m=None; best_cost=float('inf'); best_distmap=None
        for cand in S:
            dmap = nx.single_source_dijkstra_path_length(G, cand, weight=dist_attr)
            cost = sum(dmap.get(u, float('inf')) for u in nodes)
            if cost < best_cost:
                best_cost = cost; best_m = cand; best_distmap = dmap
        if best_distmap is None:
            best_m = nodes[0]
            best_distmap = nx.single_source_dijkstra_path_length(G, best_m, weight=dist_attr)
        medoids[c]=best_m; distmaps[c]=best_distmap
    return medoids, distmaps


def _community_components(G: nx.Graph, labels: Dict, c: int):
    nodes_c = [u for u, lbl in labels.items() if lbl == c]
    if not nodes_c: return []
    H = G.subgraph(nodes_c)
    return [set(comp) for comp in nx.connected_components(H)]


def _split_disconnected(G: nx.Graph, labels: Dict) -> Dict:
    current_max = max(labels.values()) if labels else -1
    communities = sorted(set(labels.values()))
    for c in communities:
        comps = _community_components(G, labels, c)
        if len(comps) <= 1: continue
        keep = comps[0]
        for u in list(labels):
            if labels[u] == c and u not in keep:
                current_max += 1
                new_id = current_max
                stack = [u]; visited = set([u]); labels[u] = new_id
                while stack:
                    x = stack.pop()
                    for v in G.neighbors(x):
                        if v not in visited and labels.get(v) == c:
                            visited.add(v); labels[v] = new_id; stack.append(v)
    return labels


def _compute_objective(G: nx.Graph, labels: Dict, distmaps: Dict[int, Dict],
                       T: float, alpha: float, p: int, dist_attr: str):
    W, B, V, _ = _per_community_W_B_V(G, labels, dist_attr)
    vol_pen = sum((V.get(c,0.0) - T)**2 for c in set(labels.values()))

    comp = 0.0
    for c in set(labels.values()):
        dmap = distmaps.get(c, {})
        for u, lbl in labels.items():
            if lbl == c:
                du = dmap.get(u, 0.0)
                comp += (du**p if p!=1 else du)
                
    return alpha*vol_pen +(1-alpha)*comp, comp, vol_pen


def _local_move_pass(G: nx.Graph, labels: Dict, medoids: Dict[int, object], distmaps: Dict[int, Dict],
                     T: float, alpha: float, p: int, dist_attr: str, rng: random.Random) -> int:
    nodes = list(G.nodes()); rng.shuffle(nodes)
    moved = 0
    for u in nodes:
        a = labels[u]
        neigh_cs = {labels[v] for v in G.neighbors(u) if labels[v] != a}
        if not neigh_cs: continue

        deg_a = sum(1 for v in G.neighbors(u) if labels[v] == a)
        if deg_a >= 2:
            nodes_a = [x for x in G.nodes() if labels.get(x) == a and x != u]
            if len(nodes_a) > 1 and not nx.is_connected(G.subgraph(nodes_a)):
                continue

        sum_to_a = sum(_edge_len(G, u, v, dist_attr) for v in G.neighbors(u) if labels.get(v) == a)
        sum_to_b_map = {b: sum(_edge_len(G, u, v, dist_attr) for v in G.neighbors(u) if labels.get(v) == b) for b in neigh_cs}
        sum_to_other = sum(_edge_len(G, u, v, dist_attr) for v in G.neighbors(u) if labels.get(v) not in (a,) and labels.get(v) not in neigh_cs)

        W, B, V, _ = _per_community_W_B_V(G, labels, dist_attr)
        Va = V.get(a, 0.0)

        da = distmaps.get(a, {}).get(u, float('inf'))
        if not math.isfinite(da):
            distmaps[a] = nx.single_source_dijkstra_path_length(G, medoids[a], weight=dist_attr)
            da = distmaps[a].get(u, float('inf'))

        best_gain = 0.0; best_b = None

        for b in neigh_cs:
            if sum_to_b_map[b] <= 0.0: continue
            Vb = V.get(b, 0.0)
            to_b = sum_to_b_map[b]
            A = to_b + sum_to_other
            Bv = sum_to_a + sum_to_other
            d_vol = float(alpha) * ( (Va - A - T)**2 - (Va - T)**2 + (Vb + Bv - T)**2 - (Vb - T)**2 )
            db = distmaps.get(b, {}).get(u)
            if db is None:
                distmaps[b] = nx.single_source_dijkstra_path_length(G, medoids[b], weight=dist_attr)
                db = distmaps[b].get(u, float('inf'))
            d_comp = float(1 - alpha)*( (db**p if p!=1 else db) - (da**p if p!=1 else da) )
            d_obj = d_vol + d_comp
            if d_obj < best_gain:
                best_gain = d_obj; best_b = b

        if best_b is not None:
            labels[u] = best_b
            moved += 1

    return moved


def leiden_compact_volume_partition(G: nx.Graph, k: Optional[int] = None, T: Optional[float] = None,
                          distance_attr: str = "Distance", alpha: float = 0.5, p: int = 2,
                          max_iter: int = 10, seed_samples: int = 10,
                          random_seed: Optional[int] = 0) -> LeidenCompactResult:
    if G.is_directed(): raise ValueError("Expected an undirected graph.")
    rng = random.Random(random_seed)

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0,1]")
    
    nodes = list(G.nodes())
    if not nodes:
        return LeidenCompactResult({}, {}, {}, {}, {}, T if T is not None else 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    if k is None and T is None:
        k = max(1, min(16, int(round(math.sqrt(max(1, G.number_of_nodes())) / 2))))

    if k is None and T is not None:
        k = max(1, int(round((2.0 * _total_edge_length(G, distance_attr)) / T)))

    if k is not None and T is None:
        seeds = _seed_farthest_first(G, k, distance_attr, rng)
        labels = _voronoi_labels(G, seeds, distance_attr)
        W0, B0, V0, _ = _per_community_W_B_V(G, labels, distance_attr)
        T = sum(V0.values()) / float(k) if k > 0 else 0.0

    seeds = _seed_farthest_first(G, k, distance_attr, rng)
    labels = _voronoi_labels(G, seeds, distance_attr)
    labels = _split_disconnected(G, labels)
    mapping = {c:i for i,c in enumerate(sorted(set(labels.values())))}
    labels = {n: mapping[c] for n,c in labels.items()}

    medoids, distmaps = _approx_update_medoids(G, labels, distance_attr, samples_per_comm=seed_samples, rng=rng)

    total_moves = 0; it = 0
    last_obj, comp, vpen = _compute_objective(G, labels, distmaps, T, alpha, p, distance_attr)

    while it < max_iter:
        it += 1
        moved = _local_move_pass(G, labels, medoids, distmaps, T, alpha, p, distance_attr, rng)
        total_moves += moved

        medoids, distmaps = _approx_update_medoids(G, labels, distance_attr, samples_per_comm=seed_samples, rng=rng)
        moved += _local_move_pass(G, labels, medoids, distmaps, T, alpha, p, distance_attr, rng)

        labels = _split_disconnected(G, labels)

        medoids, distmaps = _approx_update_medoids(G, labels, distance_attr, samples_per_comm=seed_samples, rng=rng)
        obj, comp, vpen = _compute_objective(G, labels, distmaps, T, alpha, p, distance_attr)

        if moved == 0 or abs(last_obj - obj) < 1e-9:
            last_obj = obj
            break
        last_obj = obj

    Wfin, Bfin, Vfin, total_cut = _per_community_W_B_V(G, labels, distance_attr)
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
        iterations=it,
    )