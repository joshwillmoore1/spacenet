import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
from tqdm import tqdm  # optional progress bar
from copy import deepcopy   

# bespoke imports
from spacenet.point_patterns.helpers.compute_weighted_contributions import compute_weighted_contributions
from spacenet.point_patterns.helpers.compute_weighted_contributions_parallel import compute_weighted_contributions_parallel   
from spacenet.helpers.batched_dijkstra import batched_dijkstra
from spacenet.point_patterns.helpers.spatial_bootstrap import spatial_bootstrap
from spacenet.point_patterns.helpers.polynomial_kernel import polynomial_kernel
from spacenet.point_patterns.helpers.is_connected_filter import is_connected_filter
from spacenet.helpers.node_node_distance import node_node_distance

def cross_weighted_pair_correlation_function(spatial_network,labels_for_objects_A,labels_for_objects_B, object_indices_A=None,object_indices_B=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10,marker_kernel_bandwidth_A=0.2,marker_kernel_n_A=1,marker_min_A=0, marker_max_A=1, marker_step_A=0.1,marker_kernel_bandwidth_B=0.2,marker_kernel_n_B=1,marker_min_B=0, marker_max_B=1, marker_step_B=0.1, edge_weight_name='Distance', return_confidence_interval=False,low_memory=False,verbose=True,n_jobs=1):
    """
    
    Compute the cross weighted pair correlation function between two populations of objects (A and B) on a spatial network, where the contributions of each object to the pair correlation function are weighted by a kernel function based on their marker levels.
    Polynomial kernels are used to weight contributions based on marker levels, and the pair correlation function is computed across a range of distances (r) and target marker values for both populations. 
    The function can also compute confidence intervals using spatial bootstrapping.
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network on which to compute the pair correlation function. Edges should have a weight attribute corresponding to the distance between nodes.
    labels_for_objects_A : np.array
        An array of continuous marker values for the objects in population A. The length of this array should match the number of nodes in the network, and the values should correspond to the marker levels for each node.
    labels_for_objects_B : np.array
        An array of continuous marker values for the objects in population B. The length of this array should match the number of nodes in the network, and the values should correspond to the marker levels for each node.
    object_indices_A : array-like, optional
        The indices of the nodes in population A. If None, all nodes in the network will be considered as part of population A. Default is None.
    object_indices_B : array-like, optional
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
    
    orginal_object_indices_A = deepcopy(object_indices_A)
    orginal_object_indices_B = deepcopy(object_indices_B)

    # filter to the largest connected component if the network is not connected
    this_network,object_indices_A,object_indices_B = is_connected_filter(spatial_network,object_indices_A,object_indices_B,filter_largest_connected=True)

    # check if the populations are empty
    number_of_objects_A = len(object_indices_A)
    number_of_objects_B = len(object_indices_B)
    
    if number_of_objects_A == 0:
        raise RuntimeError(f'The object_indices_A is empty following filteration of node check.')
    
    if number_of_objects_B == 0:
        raise RuntimeError(f'The object_indices_B is empty following filteration of node check.')    

    # filter labels if pop has been filtered
    these_continuous_labels_A=labels_for_objects_A[np.isin(orginal_object_indices_A,object_indices_A,assume_unique=True)]
    these_continuous_labels=labels_for_objects_B[np.isin(orginal_object_indices_B,object_indices_B,assume_unique=True)]
    
    # pre compute the kernel of marker contributions for all target markers
    tau_B = np.arange(marker_min_B, marker_max_B + marker_step_B, marker_step_B)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting= np.empty((len(object_indices_B), len(tau_B)))
    for i, t in enumerate(tau_B):
        marker_contributions_weighting[:,i] = polynomial_kernel(np.abs(these_continuous_labels - t), n=marker_kernel_n_B, delta_r=marker_kernel_bandwidth_B)
        
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
       
    #all_network_distances=dict()
    
    all_network_distances = node_node_distance(this_network,object_indices_A, weight=edge_weight_name,limit=distance_upper_bound,low_memory=low_memory,verbose=verbose)

    r = np.arange(r_min, r_max + r_step, r_step)
    
    # contributions are 
    contributions= np.empty((len(object_indices_A),len(tau_B), len(r)))    
    if n_jobs == 1:
        for i, obj_a in enumerate(tqdm(object_indices_A, desc="Computing contributions", unit="contributions", disable=not verbose)):
   
            contributions[i,:,:] = compute_weighted_contributions(obj_a, object_indices_B, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances[obj_a],marker_contributions_weighting,node_to_edges)
            #compute_weighted_contributions(object_id_A: np.array, object_indices_B: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,these_marker_contributions_weighting: np.array,node_to_edges: dict):
    else:
        contributions = np.array(compute_weighted_contributions_parallel(object_indices_A, object_indices_B, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,marker_contributions_weighting,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    
    # now compute the weighting for objects A
    tau_A = np.arange(marker_min_A, marker_max_A + marker_step_A, marker_step_A)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting_A= np.empty((len(object_indices_A), len(tau_A)))
    weighted_contributions = np.empty((len(object_indices_A),  len(tau_A)  ,len(tau_B), len(r)))

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
        confidence_interval=spatial_bootstrap(this_network,edge_weight_name,object_indices_A,weighted_contributions,weight_matrix=marker_contributions_weighting_A)
        return tau_A, tau_B, r, g , confidence_interval

    else:
        return tau_A, tau_B, r,  g





