import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

# bespoke imports
from netpcf.helpers.compute_weighted_contributions import compute_weighted_contributions
from netpcf.helpers.compute_weighted_contributions_parallel import compute_weighted_contributions_parallel   
from netpcf.helpers.batched_dijkstra import batched_dijkstra
from netpcf.helpers.spatial_bootstrap import spatial_bootstrap
from netpcf.helpers.polynomial_kernel import polynomial_kernel

def weighted_pair_correlation_function(network,labels_for_objects_B, object_indices_A=None,object_indices_B=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10,marker_kernel_bandwidth=0.2,marker_kernel_n=1,marker_min=0, marker_max=1, marker_step=0.1, edge_weight_name='Distance', save_temp_network=False, return_confidence_interval=False,confidence_interval_kwargs={},low_memory=False,verbose=True,n_jobs=1):
    
    # we make a copy of the networks as we may removed edges if a region is specified
    this_network = network
    all_node_ids = np.asarray(list(this_network.nodes))
    
    # check object indices are in the domain
    object_indices_A=object_indices_A[np.isin(object_indices_A,all_node_ids,assume_unique=True)]
    object_indices_B=object_indices_B[np.isin(object_indices_B,all_node_ids,assume_unique=True)]

    # check if the populations are empty
    number_of_objects_A = len(object_indices_A)
    number_of_objects_B = len(object_indices_B)
    
    if number_of_objects_A == 0:
        raise RuntimeError(f'The object_indices_A is empty following filteration of node check.')
    
    if number_of_objects_B == 0:
        raise RuntimeError(f'The object_indices_A is empty following filteration of node check.')    

    # check if the populations are empty
    number_of_objects_A_first = len(object_indices_A)
    number_of_objects_B_first = len(object_indices_B)
        
    if number_of_objects_A_first == 0:
        raise RuntimeError(f'The population_A is empty following filteration of boundaries and query.')
    
    if number_of_objects_B_first == 0:
        raise RuntimeError(f'The population_A is empty following filteration of boundaries and query.')
    

    these_continuous_labels=labels_for_objects_B
    
    # pre compute the kernel of marker contributions for all target markers
    tau = np.arange(marker_min, marker_max + marker_step, marker_step)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting= np.empty((len(object_indices_B), len(tau)))
    for i, t in enumerate(tau):
        marker_contributions_weighting[:,i] = polynomial_kernel(np.abs(these_continuous_labels - t), n=marker_kernel_n, delta_r=marker_kernel_bandwidth)
        
    # total length of the network
    total_length = this_network.size(weight=edge_weight_name)   
    largest_edge_length = max(data for _, _, data in this_network.edges(data=edge_weight_name))
    distance_upper_bound = r_max + r_step + spatial_kernel_bandwidth + largest_edge_length + 1e-6
   
   
   # Create a dictionary of all edges with their data for static pass
    edges_with_data = { (u, v): data for u, v, data in this_network.edges(data=edge_weight_name)}
    
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))
       
    all_network_distances=dict()
    

    
    if verbose:
        print("Computing node-node distances...")
    if low_memory:
        all_network_distances = batched_dijkstra(this_network, object_indices_A, batch_size=1000, weight=edge_weight_name, limit=distance_upper_bound,verbose=verbose)
    else:
        
        nodes = list(this_network.nodes())
        node_idx = {node: i for i, node in enumerate(nodes)}

        sparse_adj_mat = nx.to_scipy_sparse_array(this_network, weight=edge_weight_name, nodelist=nodes, format='csr')
        
        # Get indices of sources
        sources_idx = [node_idx[s] for s in object_indices_A]
        
        # Run Dijkstra from multiple sources independently
        dist_matrix = dijkstra(sparse_adj_mat, directed=False, unweighted=False, indices=sources_idx, limit=distance_upper_bound, min_only=False)
        
        # Convert back to dict form if needed
        all_network_distances = {object_indices_A[i]: {nodes[j]: dist for j, dist in zip(np.flatnonzero(~np.isinf(row)), row[~np.isinf(row)]) } for i, row in enumerate(dist_matrix)}
        
        
    r = np.arange(r_min, r_max + r_step, r_step)
    
    # contributions are 
    contributions= np.empty((len(object_indices_A),len(tau), len(r)))
    if n_jobs == 1:
        for i, obj_a in enumerate(tqdm(object_indices_A, desc="Computing contributions", unit="contributions", disable=not verbose)):
   
            contributions[i,:,:] = compute_weighted_contributions(obj_a, object_indices_B, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances[obj_a],marker_contributions_weighting,node_to_edges)
            #compute_weighted_contributions(object_id_A: np.array, object_indices_B: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,these_marker_contributions_weighting: np.array,node_to_edges: dict):
    else:
        contributions = np.array(compute_weighted_contributions_parallel(object_indices_A, object_indices_B, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,marker_contributions_weighting,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    # Compute the g function
    g = np.mean(contributions, axis=0)
    
    if return_confidence_interval:
        
        # TODO: Update for weighted PCF
        print("Warning: Ensure that the spatial_bootstrap function is updated for weighted PCF.")
        
        if verbose:
            print("Computing confidence intervals via spatial bootstrap...")
            
        confidence_interval = spatial_bootstrap(object_indices_A,contributions,all_network_distances,total_length,confidence_interval_kwargs)
        
        return r, tau, g, confidence_interval
    
    else:
        return r, tau, g





