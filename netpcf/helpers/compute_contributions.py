import numpy as np
from polynomial_kernel import polynomial_kernel
from integrated_poly_finite_kernel import integrated_poly_finite_kernel


def compute_contributions(object_id_A: np.array, object_indices_B: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,node_to_edges: dict) -> np.array:

    # Convert node distances and list to NumPy arrays for vectorized operations
    node_list = np.array(list(this_node_shortest_distance.keys()))
    node_distances = np.array(list(this_node_shortest_distance.values()))
    node_distance_dict = dict(zip(node_list, node_distances))

    # Precompute total density
    total_density = len(object_indices_B) / total_length

    # Precompute kernel indicators and nodes in kernels for all r values
    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    which_nodes_in_kernels = [node_list[kernel_r_indicators[:, i]] for i in range(len(r))]

    # Precompute nodes in kernels and in population for all r values - pretty sure this is redundant - look at removing
    object_indices_B_set = set(object_indices_B)
    which_nodes_in_kernels_and_in_pop = [
        np.array([node for node in which_nodes_in_kernel if node != object_id_A and node in object_indices_B_set])
        for which_nodes_in_kernel in which_nodes_in_kernels
    ]
    #which_nodes_in_kernels_and_in_pop = [
    #    nodes[(nodes != object_id_A) & np.isin(nodes, object_indices_B)]
    #    for nodes in which_nodes_in_kernels]

    # Initialize lists for numerator contributions and total kernel lengths
    numerator_contributions = np.zeros(len(r), dtype=np.float64)
    total_kernel_lengths = np.ones(len(r), dtype=np.float64)
    

    for r_index,(r_value, which_nodes_in_kernel_and_in_pop_local) in enumerate(zip(r, which_nodes_in_kernels_and_in_pop)):
        if which_nodes_in_kernel_and_in_pop_local.size > 0:
            # Compute numerator contributions
            node_indices = np.isin(node_list, which_nodes_in_kernel_and_in_pop_local)
            distances = np.abs(node_distances[node_indices] - r_value)
            
            numerator = np.sum(
                polynomial_kernel(
                    distances.astype(np.float64), 
                    n=np.float64(spatial_kernel_n), 
                    delta_r=np.float64(spatial_kernel_bandwidth)
                )
            )
            numerator_contributions[r_index]=numerator

            # Compute total kernel lengths -OLD METHOD - too much overhead with networkx
            #edges = this_network.edges(which_nodes_in_kernels[r_index], data=edge_weight_name)
            #weights, d_1, d_2 = zip(*((data, node_distance_dict[u], node_distance_dict[v]) for u, v, data in edges))
            
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
            
            total_length_in_kernel = np.sum(integrated_poly_finite_kernel(
                r=r_value, w=weights, d_1=d_1, d_2=d_2, 
                delta_r=spatial_kernel_bandwidth, n=spatial_kernel_n
            ))
            total_kernel_lengths[r_index]=total_length_in_kernel
            
            
    # Compute local contributions
    local_contributions = numerator_contributions / total_kernel_lengths
    local_contributions = local_contributions / total_density

    return local_contributions