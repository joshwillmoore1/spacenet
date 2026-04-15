import numpy as np
from spacenet.pcf.helpers.polynomial_kernel import polynomial_kernel
from spacenet.pcf.helpers.integrated_poly_finite_kernel import integrated_poly_finite_kernel
from spacenet.pcf.helpers.prepare_reachable_edge_arrays import prepare_reachable_edge_arrays


def compute_weighted_contributions(object_id_A: np.array, object_indices_B: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,these_marker_contributions_weighting: np.array,node_to_edges: dict):
    """
    
    Compute the local contributions to the pair correlation function for a given reference node (object_id_A) and a set of target nodes (object_indices_B) at specified radii (r), weighted by marker contributions.
    
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
    these_marker_contributions_weighting : np.array
        An array of shape (num_objects_B, num_markers) containing the contributions of each object in population B to each marker. This should be precomputed based on the marker values and the weighting scheme for the contributions.
    node_to_edges : dict
        A dictionary mapping node indices to a list of edges (and their weights) that are connected to that node. This should be precomputed for efficiency.
    
    Returns
    -------
    
    local_contributions : np.array  
        An array of local contributions to the pair correlation function for the reference node at each radius in r, weighted by the marker contributions. The shape of this array will be (num_markers, len(r)).
    
    """
    
    valid_indices = np.isin(object_indices_B, object_id_A, invert=True)
    object_indices_B = object_indices_B[valid_indices]
    these_marker_contributions_weighting = these_marker_contributions_weighting[valid_indices,:]
    
    # Convert node distances and list to NumPy arrays for vectorized operations
    node_list = np.array(list(this_node_shortest_distance.keys()))
    node_distances = np.array(list(this_node_shortest_distance.values()))

    # Precompute total density
    total_density = (np.sum(these_marker_contributions_weighting,axis=0,keepdims=True) / total_length)
    # Precompute kernel indicators and the subset of nodes that can contribute.
    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    object_population_mask = np.isin(node_list, object_indices_B, assume_unique=True)
    object_indices_B_position = {node: i for i, node in enumerate(object_indices_B)}
    edge_weights, edge_d_1, edge_d_2, edge_u_idx, edge_v_idx = prepare_reachable_edge_arrays(
        node_list, node_distances, node_to_edges
    )
    
    # Initialize lists for numerator contributions and total kernel lengths
    numerator_contributions = np.zeros((these_marker_contributions_weighting.shape[1],len(r)), dtype=np.float64)
    total_kernel_lengths = np.ones((these_marker_contributions_weighting.shape[1],len(r)), dtype=np.float64)
    
    for r_index, r_value in enumerate(r):
        kernel_node_mask = kernel_r_indicators[:, r_index]
        population_in_kernel_mask = kernel_node_mask & object_population_mask

        if np.any(population_in_kernel_mask):
            # Compute numerator contributions
            selected_nodes = node_list[population_in_kernel_mask]
            distances = np.abs(node_distances[population_in_kernel_mask] - r_value)
            
            # get the marker contributions for the nodes in the kernel
            selected_object_positions = [object_indices_B_position[node] for node in selected_nodes]
            these_marker_contributions_weighting_in_kernel = these_marker_contributions_weighting[selected_object_positions, :].copy()
            
            edge_mask = kernel_node_mask[edge_u_idx] | kernel_node_mask[edge_v_idx]
            
            total_length = np.sum(integrated_poly_finite_kernel(
                r=r_value,
                w=edge_weights[edge_mask],
                d_1=edge_d_1[edge_mask],
                d_2=edge_d_2[edge_mask],
                delta_r=spatial_kernel_bandwidth, n=spatial_kernel_n
            ))
            
            # Compute the kernel for the marker contributions - this should be each column of the marker contributions weighting matrix multiplied by the  spatial kernel contribution
            # Compute the kernel for the marker contributions
            spatial_kernel_contributions = np.array(polynomial_kernel(distances, n=spatial_kernel_n, delta_r=spatial_kernel_bandwidth), ndmin=1)
            
            # multply each of column of the marker contributions weighting matrix by the kernel for the distances
            these_marker_contributions_weighting_in_kernel *= spatial_kernel_contributions[:, np.newaxis]
                        
            # taking the sum of the kernel contributions for each object contributing 
            numerator = np.sum(these_marker_contributions_weighting_in_kernel,axis=0)
            
            numerator_contributions[:,r_index]=numerator
            total_kernel_lengths[:,r_index]=total_length*np.ones(these_marker_contributions_weighting_in_kernel.shape[1])
       

     # Transpose to marke sure the output is r x tau   
    numerator_contributions = np.vstack(numerator_contributions)
    total_kernel_lengths = np.vstack(total_kernel_lengths)  
    
    # Compute local contributions
    local_contributions = numerator_contributions / total_kernel_lengths
    # for each column of local contributions divide by the total density
    local_contributions /= total_density.T

    
    return local_contributions
