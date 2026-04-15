import numpy as np
from spacenet.pcf.helpers.polynomial_kernel import polynomial_kernel
from spacenet.pcf.helpers.integrated_poly_finite_kernel import integrated_poly_finite_kernel
from spacenet.pcf.helpers.prepare_reachable_edge_arrays import prepare_reachable_edge_arrays   

def compute_contributions(object_id_A: np.array, object_indices_B: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,node_to_edges: dict) -> np.array:
    """
    
    Computes the local contributions to the pair correlation function for a given reference node (object_id_A) and a set of target nodes (object_indices_B) at specified radii (r). 
    The contributions are computed using a polynomial kernel function that weights the distances of nodes in the network relative to the reference node, and an integrated kernel function that accounts for the edge lengths in the network. 
    The contributions are normalized by the total density of target nodes and the total length of the network to yield local contributions to the pair correlation function.
    
    Parameters
    ----------
    
    object_id_A : np.array, int
        The ID of the reference node for which local contributions are being computed.
    object_indices_B : np.array
        An array of node indices corresponding to the population B for which the pair correlation function is being computed.
    r : np.array
        An array of radii at which to compute the contributions.
    spatial_kernel_bandwidth : float
        The bandwidth parameter for the spatial kernel function.
    spatial_kernel_n : float
        The exponent parameter for the spatial kernel function.
    total_length : float
        The total length of the network, used for density normalization.
    this_node_shortest_distance : dict
        A dictionary mapping node indices to their shortest distance from the reference node (object_id_A). This should be precomputed for efficiency.
    node_to_edges : dict
        A dictionary mapping node indices to a list of edges (and their weights) that are connected to that node. This should be precomputed for efficiency.
    
    Returns 
    -------
    
    local_contributions : np.array
        An array of local contributions to the pair correlation function for the reference node at each radius in r.  
    
    
    """
    # Convert node distances and list to NumPy arrays for vectorized operations
    node_list = np.array(list(this_node_shortest_distance.keys()))
    node_distances = np.array(list(this_node_shortest_distance.values()))

    # Precompute total density
    total_density = len(object_indices_B) / total_length

    # Precompute kernel membership and the subset of nodes that can contribute.
    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    object_population_mask = np.isin(node_list, object_indices_B, assume_unique=True) & (node_list != object_id_A)
    edge_weights, edge_d_1, edge_d_2, edge_u_idx, edge_v_idx = prepare_reachable_edge_arrays(
        node_list, node_distances, node_to_edges
    )

    # Initialize lists for numerator contributions and total kernel lengths
    numerator_contributions = np.zeros(len(r), dtype=np.float64)
    total_kernel_lengths = np.ones(len(r), dtype=np.float64)
    

    for r_index, r_value in enumerate(r):
        kernel_node_mask = kernel_r_indicators[:, r_index]
        population_in_kernel_mask = kernel_node_mask & object_population_mask

        if np.any(population_in_kernel_mask):
            # Compute numerator contributions
            distances = np.abs(node_distances[population_in_kernel_mask] - r_value)
            
            numerator = np.sum(
                polynomial_kernel(
                    distances.astype(np.float64), 
                    n=np.float64(spatial_kernel_n), 
                    delta_r=np.float64(spatial_kernel_bandwidth)
                )
            )
            numerator_contributions[r_index]=numerator

            edge_mask = kernel_node_mask[edge_u_idx] | kernel_node_mask[edge_v_idx]
            
            total_length_in_kernel = np.sum(integrated_poly_finite_kernel(
                r=r_value,
                w=edge_weights[edge_mask],
                d_1=edge_d_1[edge_mask],
                d_2=edge_d_2[edge_mask],
                delta_r=spatial_kernel_bandwidth, n=spatial_kernel_n
            ))
            total_kernel_lengths[r_index]=total_length_in_kernel
            
            
    # Compute local contributions
    local_contributions = numerator_contributions / total_kernel_lengths
    local_contributions = local_contributions / total_density

    return local_contributions
