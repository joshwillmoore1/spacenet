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



def weighted_pair_correlation_function(spatial_network, node_label_name_b=None, nodes_a=None,nodes_b=None, spatial_kernel_bandwidth=10,spatial_kernel_n=2, r_min=0, r_max=100, r_step=10,marker_kernel_bandwidth=0.2,marker_kernel_n=1,marker_min=0, marker_max=1, marker_step=0.1, edge_weight_name='Distance',return_confidence_interval=False,low_memory=False,verbose=True,n_jobs=1):
    """
    
    Computes the weighted pair correlation function between two populations of nodes. Computes the spatial correlation between two populations of nodes where the second population (B) has a continuous label (marker) associated with it, and the pair correlation function is weighted by the similarity in the continuous label to a target value. 
    
        
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network on which to compute the pair correlation function. Edges should have a weight attribute corresponding to the distance between nodes.
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
    marker_kernel_bandwidth : float, optional
        The bandwidth parameter for the marker kernel function. This controls the smoothness of the weighting based on the continuous labels of objects B. Default is 0.2.
    marker_kernel_n : int, optional
        The exponent parameter for the marker kernel function. This controls the shape of the kernel for weighting based on the continuous labels of objects B. Default is 1 (which corresponds to a Laplacian-like kernel).
    marker_min : float, optional
        The minimum value of the continuous label for objects B to consider when computing the weighted pair correlation function. Default is 0.
    marker_max : float, optional
        The maximum value of the continuous label for objects B to consider when computing the weighted pair correlation function. Default is 1.    
    edge_weight_name : str, optional
        The name of the edge attribute in the network that corresponds to the distance between nodes. Default is 'Distance'.
    return_confidence_interval : bool, optional
        Whether to compute and return confidence intervals for the pair correlation function using spatial bootstrapping. Default is False.
    low_memory : bool, optional
        Whether to use a low-memory implementation of Dijkstra's algorithm that computes distances in batches. This can be useful for large networks that do not fit in memory. Default is False.
    verbose : bool, optional
        Whether to print progress messages during computation. Default is True.
    n_jobs : int, optional
        The number of parallel jobs to run when computing contributions. If n_jobs > 1, the contributions will be computed in parallel across multiple CPU cores. Default is 1 (no parallelization).
    

    Returns
    -------
        
    tau : numpy.ndarray 
        The target marker values at which the pair correlation function was computed.
    
    radii : numpy.ndarray
        The radii at which the pair correlation function was computed.
    
    g : numpy.ndarray
        The values of the pair correlation function at the corresponding radii and target marker values. Shape is (len(tau), len(radius)).
    
    confidence_interval : numpy.ndarray (if return_confidence_interval is True)   
        If return_confidence_interval is True, this will be a numpy array (2,num_mark_targets,num_radii) containing the confidence intervals for the pair correlation function at each mark target and radii. The CI[0,:,:] corresponds to the lower bounds of the confidence intervals, and CI[1,:,:] corresponds to the upper bounds. If return_confidence_interval is False, this will not be returned.
    
    
    Notes
    -----
    
    TODO: add reference to paper
    
    Examples
    --------
    
    Computing the weighted pair correlation function for a spatial network with a continuous node label (marker) and plotting the results:
    
    .. code-block:: python
    
        import spacenet as sn
        import numpy as np
        import matplotlib.pyplot as plt 

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values
        categorical_labels = sprial_df['Marker (categorical)'].values
        continuous_labels = sprial_df['Marker (continuous)'].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)
        sn.utils.add_node_labels(G,categorical_labels,node_label_name='Marker (categorical)')
        sn.utils.add_node_labels(G,continuous_labels,node_label_name='Marker (continuous)')

        # get the node ids for the nodes with categorical label A
        nodes_a = sn.utils.query_nodes(G,node_label_name='Marker (categorical)',relation='is',node_label_value='A')

        # compute the weighted-PCF for the spatial network between nodes_a and all nodes with continuous label 'Marker (continuous)'
        tau,radius,pcf_values,con_interval = sn.point_patterns.weighted_pair_correlation_function(G,
                                                                                                node_label_name_b='Marker (continuous)',
                                                                                                nodes_a=nodes_a,
                                                                                                spatial_kernel_bandwidth=80,
                                                                                                r_max=1000,
                                                                                                return_confidence_interval=True)

        # plot the PCF for the weighted pair correlation function at target mark of 0 and 1
        tau_index_0=np.where(tau==0)[0][0]
        tau_index_1=np.where(tau==1)[0][0]

        fig,ax=plt.subplots()
        ax.axhline(1,linestyle='dashed',color='grey')
        ax.plot(radius,pcf_values[tau_index_0,:],label='Target mark, $\\tau=0$',color='tab:blue')
        ax.fill_between(radius,con_interval[0,tau_index_0,:],con_interval[1,tau_index_0,:],alpha=0.2,color='tab:blue')

        ax.plot(radius,pcf_values[tau_index_1,:],label='Target mark, $\\tau=1$',color='tab:orange')
        ax.fill_between(radius,con_interval[0,tau_index_1,:],con_interval[1,tau_index_1,:],alpha=0.2,color='tab:orange')

        ax.set_xlabel('Radius')
        ax.set_ylabel('Weighted Pair Correlation')
        ax.set_ylim(0,3)
        ax.set_xlim(0,1000)   
        ax.legend()
    
    
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
    
    orginal_nodes_b = deepcopy(nodes_b)
    
    # filter to the largest connected component if the network is not connected
    this_network,nodes_a,nodes_b = is_connected_filter(this_network,nodes_a,nodes_b,filter_largest_connected=True)

    # check if the populations are empty
    number_of_objects_A = len(nodes_a)
    number_of_objects_B = len(nodes_b)
    
    if number_of_objects_A == 0:
        raise RuntimeError(f'The nodes_a is empty following filteration of node check.')
    
    if number_of_objects_B == 0:
        raise RuntimeError(f'The nodes_b is empty following filteration of node check.')   
    
    
    # grab the continuous labels for objects B
    if isinstance(node_label_name_b, str):
        labels_for_objects_B = np.array([this_network.nodes[node][node_label_name_b] for node in orginal_nodes_b]) 
    elif isinstance(node_label_name_b, (list, np.ndarray)):
        if len(node_label_name_b) != len(orginal_nodes_b):
            raise ValueError(f'If node_label_name_b is an array, it must be the same length as nodes_b. Got length {len(node_label_name_b)} for node_label_name_b and length {len(orginal_nodes_b)} for nodes_b.')
        labels_for_objects_B = np.array(node_label_name_b)
    else:
        raise ValueError(f'node_label_name_b must be either a string (the name of the node attribute corresponding to the continuous labels for objects B) or an array of continuous values for objects B. Got type {type(node_label_name_b)} for node_label_name_b.')  

    # check that the labels for objects B are continuous (numeric)
    if not np.issubdtype(labels_for_objects_B.dtype, np.number):
        raise ValueError(f'The labels for objects B must be continuous (numeric). Got dtype {labels_for_objects_B.dtype} for labels_for_objects_B.')
    
    # filter labels if pop has been filtered
    these_continuous_labels=labels_for_objects_B[np.isin(orginal_nodes_b,nodes_b,assume_unique=True)]
    
    # pre compute the kernel of marker contributions for all target markers
    tau = np.arange(marker_min, marker_max + marker_step, marker_step)
    
    # each column of the marker contrubutions weighting matrix is the kernel for a different target marker
    marker_contributions_weighting= np.empty((len(nodes_b), len(tau)))
    for i, t in enumerate(tau):
        marker_contributions_weighting[:,i] = polynomial_kernel(np.abs(these_continuous_labels - t), n=marker_kernel_n, delta_r=marker_kernel_bandwidth)
   
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
    contributions= np.empty((len(nodes_a),len(tau), len(r)))
    if n_jobs == 1:
        for i, obj_a in enumerate(tqdm(nodes_a, desc="Computing contributions", unit="contributions", disable=not verbose)):
   
            contributions[i,:,:] = compute_weighted_contributions(obj_a, nodes_b, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances[obj_a],marker_contributions_weighting,node_to_edges)
            #compute_weighted_contributions(object_id_A: np.array, nodes_b: np.array, r: np.array, spatial_kernel_bandwidth: float,spatial_kernel_n: float, total_length: float ,this_node_shortest_distance: dict,these_marker_contributions_weighting: np.array,node_to_edges: dict):
    else:
        contributions = np.array(compute_weighted_contributions_parallel(nodes_a, nodes_b, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,marker_contributions_weighting,node_to_edges, n_jobs=n_jobs,verbose=verbose))
    # Compute the g function
    g = np.mean(contributions, axis=0)
    
    # transpose g so that the first dimension corresponds to the marker targets and the second dimension corresponds to the radii, 
    # for consistency with other functions and ease of plotting
    
    if return_confidence_interval:
        
        if verbose:
            print("Computing confidence intervals via spatial bootstrap...")
        # contributions (number of objects A, number of target markers, number of radii)
        confidence_interval=spatial_bootstrap(this_network,edge_weight_name,nodes_a,contributions)
        return tau, r, g, confidence_interval
    
    else:
        return tau, r, g





