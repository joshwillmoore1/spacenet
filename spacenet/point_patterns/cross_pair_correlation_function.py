import numpy as np
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

# bespoke imports
from spacenet.point_patterns.helpers.compute_contributions import compute_contributions
from spacenet.point_patterns.helpers.compute_contributions_parallel import compute_contributions_parallel   
from spacenet.point_patterns.helpers.spatial_bootstrap import spatial_bootstrap
from spacenet.point_patterns.helpers.is_connected_filter import is_connected_filter
from spacenet.helpers.node_node_distance import node_node_distance  
from spacenet.point_patterns.helpers.polynomial_kernel_bandwidth_scale import polynomial_kernel_bandwidth_scale

def cross_pair_correlation_function(spatial_network, nodes_a=None, nodes_b=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10, edge_weight_name='Distance', return_confidence_interval=False,low_memory=False,verbose=True,n_jobs=1):
    """
    
    Computes the pair correlation function (PCF) and cross-PCF between on nodes in a spatial network.
    The PCF is computed when nodes_a and nodes_b are the same set of node indices.     
    The cross-PCF is computed when nodes_a and nodes_b are different sets of node indices. The cross-PCF measures the spatial correlation between two different populations of nodes in the network, while the PCF measures the spatial correlation within a single population of nodes.
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network on which to compute the pair correlation function. Edges should have a weight attribute corresponding to the distance between nodes.
    nodes_a : array-like, optional
        The indices of the nodes in population A. If None, all nodes in the network will be considered as part of population A. Default is None.
    nodes_b : array-like, optional
        The indices of the nodes in population B. If None, all nodes in the network will be considered as part of population B. Default is None.
    spatial_kernel_bandwidth : float, optional
        The bandwidth parameter for the spatial kernel function. This controls the smoothness of the pair correlation function. Default is 10.
    spatial_kernel_n : int, optional
        The exponent parameter for the spatial kernel function. This controls the shape of the kernel. Default is 2 (which corresponds to Epanechnikov).
    r_min : float, optional
        The minimum distance to consider when computing the pair correlation function. Default is 0.
    r_max : float, optional
        The maximum distance to consider when computing the pair correlation function. Default is 100.
    r_step : float, optional
        The step size for the distance bins when computing the pair correlation function. Default is 10.
    edge_weight_name : str, optional
        The name of the edge attribute in the network that corresponds to the distance between nodes. Default is 'Distance'.
    low_memory : bool, optional
        Whether to use a low-memory implementation of Dijkstra's algorithm that computes distances in batches. This can be useful for large networks that do not fit in memory. Default is False.
    verbose : bool, optional
        Whether to print progress messages during computation. Default is True.
    n_jobs : int, optional
        The number of parallel jobs to run when computing contributions. If n_jobs > 1, the contributions will be computed in parallel across multiple CPU cores. Default is 1 (no parallelization).
    

    Returns
    -------
    
    radii : numpy.ndarray
        The radii at which the pair correlation function was computed.
    
    g : numpy.ndarray
        The values of the pair correlation function at the corresponding radii.
    
    confidence_interval : numpy.ndarray (if return_confidence_interval is True)   
        If return_confidence_interval is True, this will be a numpy array (2,n) containing the confidence intervals for the pair correlation function at each radius. The first row corresponds to the lower bounds of the confidence intervals, and the second row corresponds to the upper bounds. If return_confidence_interval is False, this will not be returned.
    
    Notes
    -----
    For a pair correlation function (not cross) on a spatial network set nodes_a and nodes_b to be the same set of node indices. 
    For details, see the reference paper...

    
    Examples
    --------
    
    
    
    
    """
    
    # we make a copy of the networks as we may removed edges if a region is specified
    this_network = spatial_network
    all_node_ids = np.asarray(list(this_network.nodes))
    
    if nodes_a is None:
        nodes_a = all_node_ids
    if nodes_b is None:
        nodes_b = all_node_ids
    
    # check object indices are in the domain
    nodes_a=nodes_a[np.isin(nodes_a,all_node_ids,assume_unique=True)]
    nodes_b=nodes_b[np.isin(nodes_b,all_node_ids,assume_unique=True)]

    # filter to the largest connected component if the network is not connected
    this_network,nodes_a,nodes_b = is_connected_filter(this_network,nodes_a,nodes_b,filter_largest_connected=True)
    
    # check if the populations are empty
    number_of_objects_A = len(nodes_a)
    number_of_objects_B = len(nodes_b)
    
    if number_of_objects_A == 0:
        raise RuntimeError(f'The nodes_a is empty following filteration of node check.')
    
    if number_of_objects_B == 0:
        raise RuntimeError(f'The nodes_b is empty following filteration of node check.')    
    

    # total length of the network
    total_length = this_network.size(weight=edge_weight_name)   
    # Find the largest edge length value in the network

    # Create a dictionary of all edges with their data for static pass
    edges_with_data = { (u, v): data for u, v, data in this_network.edges(data=edge_weight_name)}
    
    # get the edge values as a numpy array for efficient computation of statistics
    edge_values = np.array(list(edges_with_data.values()))
    largest_edge_length = np.max(edge_values)
    total_length = np.sum(edge_values)
    mean_edge_length = np.mean(edge_values) 
    
    discrete_check_lower_bound = polynomial_kernel_bandwidth_scale(spatial_kernel_n,proportion_kernel_mass=0.75)*mean_edge_length
    
    if spatial_kernel_bandwidth < discrete_check_lower_bound:
        print(f"Warning: The spatial kernel bandwidth is set to {spatial_kernel_bandwidth}, which may be too small relative to the mean edge length of the network ({mean_edge_length:.2f}). This could lead to a pair correlation function that is dominated by discrete effects. Consider increasing the spatial kernel bandwidth to at least {discrete_check_lower_bound:.2f} to mitigate this issue.")

    
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))
        
    all_network_distances=dict()
    distance_upper_bound = r_max + r_step + spatial_kernel_bandwidth + largest_edge_length + 1e-6
    
    # compute the node-node distances for all nodes in population A to all other nodes in the network, with a distance limit to reduce computational cost. This will be used to compute the contributions of each node in population A to the pair correlation function.
    all_network_distances = node_node_distance(this_network,nodes_a, weight=edge_weight_name,limit=distance_upper_bound,low_memory=low_memory,verbose=verbose)
    
    r = np.arange(r_min, r_max + r_step, r_step)
    contributions= np.empty((len(nodes_a), len(r)))
    
    if n_jobs == 1:             
        for i, obj_a in enumerate(tqdm(nodes_a, desc="Computing contributions", unit="contributions", disable=not verbose)):   
            contributions[i,:] = compute_contributions(obj_a, nodes_b, r, spatial_kernel_bandwidth,spatial_kernel_n, total_length,all_network_distances[obj_a],node_to_edges)
    else:
        contributions = np.array(compute_contributions_parallel(nodes_a, nodes_b, r, spatial_kernel_bandwidth,
                                    spatial_kernel_n,total_length, all_network_distances,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    # Compute the g function
    g = np.mean(contributions, axis=0)
    
    if return_confidence_interval:
        if verbose:
            print("Computing confidence intervals via spatial bootstrap...")
        confidence_interval=spatial_bootstrap(this_network,edge_weight_name,nodes_a,contributions)
        
        return r ,g , confidence_interval

    else:
        return r, g





