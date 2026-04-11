import numpy as np
from spacenet.pcf.helpers.polynomial_kernel import polynomial_kernel
from spacenet.pcf.helpers.integrated_poly_finite_kernel import integrated_poly_finite_kernel


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
    node_distance_dict = dict(zip(node_list, node_distances))

    # Precompute total density
    total_density = (np.sum(these_marker_contributions_weighting,axis=0,keepdims=True) / total_length)
    # Precompute kernel indicators and nodes in kernels for all r values
    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    which_nodes_in_kernels = [node_list[kernel_r_indicators[:, i]] for i in range(len(r))]

    # Precompute nodes in kernels and in population for all r values -- this can be pre filtered?
    object_indices_B_set = set(object_indices_B)
    which_nodes_in_kernels_and_in_pop = [
        np.array([node for node in which_nodes_in_kernel if node != object_id_A and node in object_indices_B_set])
        for which_nodes_in_kernel in which_nodes_in_kernels
    ]
    
    # Initialize lists for numerator contributions and total kernel lengths
    numerator_contributions = np.zeros((these_marker_contributions_weighting.shape[1],len(r)), dtype=np.float64)
    total_kernel_lengths = np.ones((these_marker_contributions_weighting.shape[1],len(r)), dtype=np.float64)
    
    for r_index,(r_value, which_nodes_in_kernel_and_in_pop_local) in enumerate(zip(r, which_nodes_in_kernels_and_in_pop)):
        if which_nodes_in_kernel_and_in_pop_local.size > 0:
            # Compute numerator contributions
            node_indices = np.isin(node_list, which_nodes_in_kernel_and_in_pop_local)
            distances = np.abs(node_distances[node_indices] - r_value)
            
            # get the marker contributions for the nodes in the kernel
            these_marker_contributions_weighting_in_kernel = these_marker_contributions_weighting[np.isin(object_indices_B, which_nodes_in_kernel_and_in_pop_local,assume_unique=True),:]
            
            # Compute total kernel lengths
            edges_seen = set()
            filtered_edges_with_data = {}
            nodes_in_kernel = set(which_nodes_in_kernels[r_index])

            for node in nodes_in_kernel:
                for edge, weight in node_to_edges.get(node, []):
                    if edge not in edges_seen:
                        filtered_edges_with_data[edge] = weight
                        edges_seen.add(edge)
            weights, d_1, d_2 = zip(*((data, node_distance_dict[this_edge[0]], node_distance_dict[this_edge[1]]) for this_edge, data in filtered_edges_with_data.items()))

            weights = np.array(weights,dtype=np.float64)
            d_1 = np.array(d_1, dtype=np.float64)
            d_2 = np.array(d_2, dtype=np.float64)
            
            total_length = np.sum(integrated_poly_finite_kernel(
                r=r_value, w=weights, d_1=d_1, d_2=d_2, 
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