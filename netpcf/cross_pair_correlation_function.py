import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

# bespoke imports
from netpcf.helpers.compute_contributions import compute_contributions
from netpcf.helpers.compute_contributions_parallel import compute_contributions_parallel   
from netpcf.helpers.batched_dijkstra import batched_dijkstra
from netpcf.helpers.spatial_bootstrap import spatial_bootstrap

def cross_pair_correlation_function(network, object_indices_A=None, object_indices_B=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10, edge_weight_name='Distance', return_confidence_interval=False,confidence_interval_kwargs={},low_memory=False,verbose=True,n_jobs=1):
    
    # we make a copy of the networks as we may removed edges if a region is specified
    this_network = network
    all_node_ids = np.asarray(list(this_network.nodes))
    
    if object_indices_A is None:
        object_indices_A = all_node_ids
    if object_indices_B is None:
        object_indices_B = all_node_ids
    
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

    # total length of the network
    total_length = this_network.size(weight=edge_weight_name)   
    # Find the largest edge length value in the network
    largest_edge_length = max(data for _, _, data in this_network.edges(data=edge_weight_name))

    # Create a dictionary of all edges with their data for static pass
    edges_with_data = { (u, v): data for u, v, data in this_network.edges(data=edge_weight_name)}
    
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))
        
    all_network_distances=dict()
    distance_upper_bound = r_max + r_step + spatial_kernel_bandwidth + largest_edge_length + 1e-6
    
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
    contributions= np.empty((len(object_indices_A), len(r)))
    
    if n_jobs == 1:             
        for i, obj_a in enumerate(tqdm(object_indices_A, desc="Computing contributions", unit="contributions", disable=not verbose)):   
            contributions[i,:] = compute_contributions(obj_a, object_indices_B, r, spatial_kernel_bandwidth,spatial_kernel_n, total_length,all_network_distances[obj_a],node_to_edges)
    else:
        contributions = np.array(compute_contributions_parallel(object_indices_A, object_indices_B, r, spatial_kernel_bandwidth,
                                    spatial_kernel_n,total_length, all_network_distances,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    # Compute the g function
    g = np.mean(contributions, axis=0)
    
    
    if return_confidence_interval:
        if verbose:
            print("Computing confidence intervals via spatial bootstrap...")
        confidence_interval=spatial_bootstrap(this_network,edge_weight_name,object_indices_A,contributions,all_network_distances,weight_matrix=None,**confidence_interval_kwargs)
        
        return r ,g , confidence_interval

    else:
        return r, g





