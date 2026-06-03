import numpy as np
from collections import defaultdict
from tqdm import tqdm  # optional progress bar
from copy import deepcopy   

# bespoke imports
from spacenet.point_patterns.helpers.compute_weighted_contributions import compute_weighted_contributions
from spacenet.point_patterns.helpers.compute_weighted_contributions_parallel import compute_weighted_contributions_parallel   
from spacenet.point_patterns.helpers.spatial_bootstrap import spatial_bootstrap
from spacenet.point_patterns.helpers.polynomial_kernel import polynomial_kernel
from spacenet.point_patterns.helpers.is_connected_filter import is_connected_filter
from spacenet.helpers.node_node_distance import node_node_distance
from spacenet.point_patterns.helpers.polynomial_kernel_bandwidth_scale import polynomial_kernel_bandwidth_scale


def cross_weighted_pair_correlation_function(spatial_network,node_label_name_a,node_label_name_b, nodes_a=None,nodes_b=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10,marker_kernel_bandwidth_A=0.2,marker_kernel_n_A=1,marker_min_A=0, marker_max_A=1, marker_step_A=0.1,marker_kernel_bandwidth_B=0.2,marker_kernel_n_B=1,marker_min_B=0, marker_max_B=1, marker_step_B=0.1, edge_weight_name='Distance', return_confidence_interval=False,low_memory=False,verbose=True,n_jobs=1):
    """
    
    Compute the cross weighted pair correlation function between two populations of objects (A and B) on a spatial network, where the contributions of each object to the pair correlation function are weighted by a kernel function based on their marker levels.
    Polynomial kernels are used to weight contributions based on marked levels, and the pair correlation function is computed across a range of distances (r) and target marker values for both populations. 
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network on which to compute the pair correlation function. Edges should have a weight attribute corresponding to the distance between nodes.
    node_label_name_a : str or array-like
        A continuous label for each object in population A. This should be the name of labels associated with a continous label on the name. Alternatively, this can be an array of continous values of the same length as nodes_b, where each entry corresponds to the label of the respective object in population A.
    node_label_name_b : str or array-like
        A continuous label for each object in population B. This should be the name of labels associated with a continous label on the name. Alternatively, this can be an array of continous values of the same length as nodes_b, where each entry corresponds to the label of the respective object in population B.
    nodes_a : array-like, optional
        The indices of the nodes in population A. If None, all nodes in the network will be considered as part of population A. Default is None.
    nodes_b : array-like, optional
        The indices of the nodes in population B. If None, all nodes in the network will be considered as part of population B. Default is None.
    spatial_kernel_bandwidth : float, optional
        The bandwidth parameter for the spatial kernel function. This controls the smoothness of the pair correlation function. Default is 10.
    spatial_kernel_n : int, optional
        The exponent parameter for the spatial kernel function. This controls the shape of the kernel. Default is 2 (which corresponds to a Gaussian-like kernel).
    r_min : float, optional
        The minimum distance to consider when computing the pair correlation function. Default is 0.
    r_max : float, optional
        The maximum distance to consider when computing the pair correlation function. Default is 100.
    r_step : float, optional
        The step size for the distance bins when computing the pair correlation function. Default is 10.
    marker_kernel_bandwidth_A : float, optional
        The bandwidth parameter for the marker kernel function for population A. This controls the smoothness of the weighting based on marker levels. Default is 0.2.
    marker_kernel_n_A : int, optional
        The exponent parameter for the marker kernel function for population A. This controls the shape of the kernel for marker weighting. Default is 1 (which corresponds to a triangular kernel).
    marker_min_A : float, optional
        The minimum marker value to consider for population A when computing the marker kernel. Default is 0.
    marker_max_A : float, optional
        The maximum marker value to consider for population A when computing the marker kernel. Default is 1.
    marker_step_A : float, optional
        The step size for the marker values when computing the marker kernel for population A. Default is 0.1.
    marker_kernel_bandwidth_B : float, optional
        The bandwidth parameter for the marker kernel function for population B. This controls the smoothness of the weighting based on marker levels for population B. Default is 0.2.
    marker_kernel_n_B : int, optional
        The exponent parameter for the marker kernel function for population B. This controls the shape of the kernel for marker weighting for population B. Default is 1 (which corresponds to a triangular kernel).
    marker_min_B : float, optional
        The minimum marker value to consider for population B when computing the marker kernel. Default is 0.
    marker_max_B : float, optional
        The maximum marker value to consider for population B when computing the marker kernel. Default is 1.
    marker_step_B : float, optional
        The step size for the marker values when computing the marker kernel for population B. Default is 0.1.
    edge_weight_name : str, optional
        The name of the edge attribute in the network that corresponds to the distance between nodes. Default is 'Distance'.
    return_confidence_interval : bool, optional
        Whether to compute and return 95% confidence intervals for the pair correlation function using spatial bootstrapping. Default is False.
    low_memory : bool, optional
        Whether to use a low-memory implementation of Dijkstra's algorithm that computes distances in batches. This can be useful for large networks that do not fit in memory. Default is False.
    verbose : bool, optional
        Whether to print progress messages during computation. Default is True.
    
    Returns
    -------
    
    tau_A : numpy.ndarray
        The target marker values for label A at which the pair correlation function was computed.
    tau_B : numpy.ndarray
        The target marker values for label B at which the pair correlation function was computed.
    radii : numpy.ndarray   
        The radii at which the pair correlation function was computed.
    g : numpy.ndarray
        The values of the pair correlation function at the corresponding target marker values and radii. This will be a 3D array with dimensions corresponding to (number of target marker values for A, number of target marker values for B, number of radii).
    confidence_interval : numpy.ndarray (if return_confidence_interval is True)
        If return_confidence_interval is True, this will be a numpy array (2, number of target marker values for A, number of target marker values for B, number of radii) containing the confidence intervals for the pair correlation function at each combination of target marker values and radius. The CI[0,:,:,:] corresponds to the lower bounds of the confidence intervals, and CI[1,:,:,:] corresponds to the upper bounds. If return_confidence_interval is False, this will not be returned.
    
    Notes
    -----
    
    For details, see the reference paper...
    
    Examples
    --------
    
    """
    
    all_node_ids = np.asarray(list(spatial_network.nodes))
    
    if nodes_a is None:
        nodes_a = all_node_ids
    if nodes_b is None:
        nodes_b = all_node_ids
    
    orginal_nodes_a = deepcopy(nodes_a)
    orginal_nodes_b = deepcopy(nodes_b)

    # filter to the largest connected component if the network is not connected
    this_network,nodes_a,nodes_b = is_connected_filter(spatial_network,nodes_a,nodes_b,filter_largest_connected=True)

    # check if the populations are empty
    number_of_objects_A = len(nodes_a)
    number_of_objects_B = len(nodes_b)
    
    if number_of_objects_A == 0:
        raise RuntimeError(f'The nodes_a is empty following filteration of node check.')
    
    if number_of_objects_B == 0:
        raise RuntimeError(f'The nodes_b is empty following filteration of node check.')  
    
    # grab the continuous labels for objects A
    if isinstance(node_label_name_a, str):
        labels_for_objects_A = np.array([this_network.nodes[node][node_label_name_a] for node in orginal_nodes_a]) 
    elif isinstance(node_label_name_a, (list, np.ndarray)):
        if len(node_label_name_a) != len(orginal_nodes_a):
            raise ValueError(f'If node_label_name_b is an array, it must be the same length as nodes_b. Got length {len(node_label_name_b)} for node_label_name_b and length {len(orginal_nodes_b)} for nodes_b.')
        labels_for_objects_A = np.array(node_label_name_a)
    else:
        raise ValueError(f'node_label_name_b must be either a string (the name of the node attribute corresponding to the continuous labels for objects B) or an array of continuous values for objects B. Got type {type(node_label_name_b)} for node_label_name_b.')  

    
    # grab the continuous labels for objects B
    if isinstance(node_label_name_b, str):
        labels_for_objects_B = np.array([this_network.nodes[node][node_label_name_b] for node in orginal_nodes_b]) 
    elif isinstance(node_label_name_b, (list, np.ndarray)):
        if len(node_label_name_b) != len(orginal_nodes_b):
            raise ValueError(f'If node_label_name_b is an array, it must be the same length as nodes_b. Got length {len(node_label_name_b)} for node_label_name_b and length {len(orginal_nodes_b)} for nodes_b.')
        labels_for_objects_B = np.array(node_label_name_b)
    else:
        raise ValueError(f'node_label_name_b must be either a string (the name of the node attribute corresponding to the continuous labels for objects B) or an array of continuous values for objects B. Got type {type(node_label_name_b)} for node_label_name_b.')  
  

    # check that the labels are continuous
    if not np.issubdtype(labels_for_objects_A.dtype, np.number):
        raise ValueError(f'The labels for objects A must be continuous (numeric). Got dtype {labels_for_objects_A.dtype} for labels_for_objects_A.')
    if not np.issubdtype(labels_for_objects_B.dtype, np.number):
        raise ValueError(f'The labels for objects B must be continuous (numeric). Got dtype {labels_for_objects_B.dtype} for labels_for_objects_B.')
    
    
    # filter labels if pop has been filtered
    these_continuous_labels_A=labels_for_objects_A[np.isin(orginal_nodes_a,nodes_a,assume_unique=True)]
    these_continuous_labels=labels_for_objects_B[np.isin(orginal_nodes_b,nodes_b,assume_unique=True)]
    
    # pre compute the kernel of marker contributions for all target markers
    tau_B = np.arange(marker_min_B, marker_max_B + marker_step_B, marker_step_B)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting= np.empty((len(nodes_b), len(tau_B)))
    for i, t in enumerate(tau_B):
        marker_contributions_weighting[:,i] = polynomial_kernel(np.abs(these_continuous_labels - t), n=marker_kernel_n_B, delta_r=marker_kernel_bandwidth_B)
        
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

    distance_upper_bound = r_max + r_step + spatial_kernel_bandwidth + largest_edge_length + 1e-6

    
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))
           
    all_network_distances = node_node_distance(this_network,nodes_a, weight=edge_weight_name,limit=distance_upper_bound,low_memory=low_memory,verbose=verbose)

    r = np.arange(r_min, r_max + r_step, r_step)
    
    # contributions are 
    contributions= np.empty((len(nodes_a),len(tau_B), len(r)))    
    if n_jobs == 1:
        for i, obj_a in enumerate(tqdm(nodes_a, desc="Computing contributions", unit="contributions", disable=not verbose)):
   
            contributions[i,:,:] = compute_weighted_contributions(obj_a, nodes_b, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances[obj_a],marker_contributions_weighting,node_to_edges)
            #compute_weighted_contributions(object_id_A: np.array, nodes_b: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,these_marker_contributions_weighting: np.array,node_to_edges: dict):
    else:
        contributions = np.array(compute_weighted_contributions_parallel(nodes_a, nodes_b, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,marker_contributions_weighting,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    
    # now compute the weighting for objects A
    tau_A = np.arange(marker_min_A, marker_max_A + marker_step_A, marker_step_A)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting_A= np.empty((len(nodes_a), len(tau_A)))
    weighted_contributions = np.empty((len(nodes_a),  len(tau_A)  ,len(tau_B), len(r)))

    # compute the marker contributions and apply to the spatial contributions
    for i, t in enumerate(tau_A):
        marker_contributions_weighting_A[:,i] = polynomial_kernel(np.abs(these_continuous_labels_A - t), n=marker_kernel_n_A, delta_r=marker_kernel_bandwidth_A)
        weighted_contributions[:,i,:,:] = contributions * marker_contributions_weighting_A[:,i][:,np.newaxis,np.newaxis]
        
        
    # comput the weighted average of the contributions
    total_marker_contribution_weighting_A = np.sum(marker_contributions_weighting_A, axis=0)
    g = (1/total_marker_contribution_weighting_A)[:,np.newaxis,np.newaxis]*np.sum(weighted_contributions, axis=0)
    
    # so we want to have g (r, m1,m2) so let's changes the order of the dimensions
    #g = np.transpose(g, (2,0,1))
    
    # add a method for spatial contributions
    if return_confidence_interval:
        # spatial bootstrap needs to accomodate the tensor
        confidence_interval=spatial_bootstrap(this_network,edge_weight_name,nodes_a,weighted_contributions,weight_matrix=marker_contributions_weighting_A)
        return tau_A, tau_B, r, g , confidence_interval

    else:
        return tau_A, tau_B, r,  g





