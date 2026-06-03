from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from spacenet.partition.helpers.DistanceBackend import DistanceBackend
from spacenet.partition.helpers.total_edge_length import total_edge_length
from spacenet.partition.helpers.seed_farthest_first import seed_farthest_first
from spacenet.partition.helpers.voronoi_labels import voronoi_labels
from spacenet.partition.helpers.labels_to_array import labels_to_array
from spacenet.partition.helpers.split_disconnected_from_array import split_disconnected_from_array
from spacenet.partition.helpers.community_ids import community_ids
from spacenet.partition.helpers.approx_update_medoids_from_array import approx_update_medoids_from_array
from spacenet.partition.helpers.local_move_pass_from_array import local_move_pass_from_array
from spacenet.partition.helpers.compute_objective_from_arrays import compute_objective_from_arrays
from spacenet.partition.helpers.labels_to_dict import labels_to_dict
from spacenet.partition.helpers.per_community_W_B_V_from_array import per_community_W_B_V_from_array

import random
import math
import networkx as nx   



@dataclass
class PartitionResult:
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
    objective_values: List[float]
    
    
    
def compact_volume_partition(
    G: nx.Graph,
    k: Optional[int] = None,
    T: Optional[float] = None,
    distance_attr: str = "Distance",
    alpha: float = 0.5,
    p: int = 2,
    max_iter: int = 10,
    seed_samples: int = 10,
    random_seed: Optional[int] = 0,
) -> PartitionResult:
    """
    
    A compact and volume-balanced partitioning algorithm for spatial networks with edge distances.
    The algorithm seeks to partition the graph into k communities that are spatially contiguous, have similar volumes (total edge lengths), and minimize the distance from nodes to their community medoids.
    Convergence is achieved when no single node can be moved to a different community to improve the combined objective of compactness and volume balance, or when the maximum number of iterations is reached.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to partition. Must be undirected.
    k : int, optional
        The desired number of communities. If None, a heuristic based on the number of nodes will be used.
    T : float, optional
        The target volume for each community. If None, it will be estimated based on the total edge length and k.
    distance_attr : str, optional
        The edge attribute to use as distance for the partitioning. Default is "Distance".
    alpha : float, optional
        The weight of the volume penalty in the objective function. Must be in [0, 1]. Default is 0.5.
    p : int, optional
        The exponent for the volume penalty. Default is 2 (squared penalty).
    max_iter : int, optional
        The maximum number of iterations for the local moving phase. Default is 10.
    seed_samples : int, optional
        The number of samples to use when approximating medoids. Default is 10.
    random_seed : int or None, optional
        The random seed for reproducibility. If None, a random seed will be used. Default is 0.
    
    
    Returns
    -------
    PartitionResult
        A dataclass containing the partitioning results, including:
        
        - labels: A dictionary mapping each node to its assigned community.
        
        - community_lengths: A dictionary mapping each community to the total length of internal edges (W).
        
        - community_cut_lengths: A dictionary mapping each community to the total length of cut edges (B).
        
        - community_volumes: A dictionary mapping each community to its volume (V = W + B).
        
        - medoids: A dictionary mapping each community to its medoid node.
        
        - T: The target volume used in the partitioning.
        
        - k: The number of communities in the final partitioning.
        
        - objective: The final value of the objective function.
        
        - compactness: The final compactness component of the objective function.
        
        - volume_penalty: The final volume penalty component of the objective function.
        
        - total_cut_length: The total length of cut edges across all communities.
        
        - moves: The total number of node moves made during the local moving phase.
        
        - iterations: The number of iterations taken in the local moving phase until convergence or reaching max_iter.
    
    
    Notes
    -----
    
    TODO: add reference to paper
    
    
    Examples
    --------
    
    Compact volume partitioning can be used to create spatially contiguous and volume-balanced communities in a spatial network.
    Below is an example of how to use the `compact_volume_partition` function with a spatial network generated from the Spiral dataset. 
    The resulting partition labels are added to the spatial network and visualised.
    
    .. code-block:: python
    
        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.generate_spatial_network(points,network_type='delaunay',max_edge_distance=75)

        # compute a compact volume partition for the spatial network with 5 partitions
        cv_partition = sn.partition.compact_volume_partition(G,k=5)

        #get the labels and nodes associated with each partition
        partition_label_dict = cv_partition.labels
        nodes_partition,labels_partition = list(partition_label_dict.keys()),list(partition_label_dict.values())

        # add the partition labels to the spatial network
        sn.utils.add_node_labels(G,
                                labels=labels_partition,
                                nodes=nodes_partition,
                                node_label_name='volume_partition (5)')

        # plot the spatial network with nodes coloured by their partition label
        sn.utils.plot_spatial_network(G,node_label_name='volume_partition (5)')
    
    
    """
    
    if G.is_directed():
        raise ValueError("Expected an undirected graph.")

    rng = random.Random(random_seed)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")

    backend = DistanceBackend.from_networkx(G, distance_attr)

    if k is None and T is None:
        k = max(1, min(16, int(round(math.sqrt(max(1, G.number_of_nodes())) / 2))))

    if k is None and T is not None:
        k = max(1, int(round((2.0 * total_edge_length(G, distance_attr)) / T)))

    if k is not None and T is None:
        seeds = seed_farthest_first(G, k, distance_attr, rng, backend=backend)
        labels = voronoi_labels(G, seeds, distance_attr, backend=backend)
        label_arr = labels_to_array(labels, backend)
        _, _, volumes, _ = per_community_W_B_V_from_array(backend, label_arr)
        T = sum(volumes.values()) / float(k) if k > 0 else 0.0

    seeds = seed_farthest_first(G, k, distance_attr, rng, backend=backend)
    labels = voronoi_labels(G, seeds, distance_attr, backend=backend)
    label_arr = labels_to_array(labels, backend)
    label_arr = split_disconnected_from_array(backend, label_arr)

    communities = sorted(int(c) for c in community_ids(label_arr))
    mapping = {community: idx for idx, community in enumerate(communities)}
    for old, new in mapping.items():
        label_arr[label_arr == old] = new

    medoids, distmaps = approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)

    total_moves = 0
    iterations = 0
    last_obj, comp, vpen = compute_objective_from_arrays(backend, label_arr, distmaps, T, alpha, p)
    objective_values = [last_obj]   
    while iterations < max_iter:
        iterations += 1
        moved = local_move_pass_from_array(backend, label_arr, medoids, distmaps, T, alpha, p, rng)
        total_moves += moved

        medoids, distmaps = approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)
        moved += local_move_pass_from_array(backend, label_arr, medoids, distmaps, T, alpha, p, rng)

        label_arr = split_disconnected_from_array(backend, label_arr)

        medoids, distmaps = approx_update_medoids_from_array(backend, label_arr, seed_samples, rng)
        obj, comp, vpen = compute_objective_from_arrays(backend, label_arr, distmaps, T, alpha, p)
        objective_values.append(obj)
        if moved == 0 or abs(last_obj - obj) < 1e-9:
            last_obj = obj
            break
        last_obj = obj

    labels = labels_to_dict(label_arr, backend)
    Wfin, Bfin, Vfin, total_cut = per_community_W_B_V_from_array(backend, label_arr)
    k_final = len(set(labels.values()))

    return PartitionResult(
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
        objective_values=objective_values
    )
