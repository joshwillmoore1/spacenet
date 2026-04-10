
import numpy as np
import networkx as nx
from netpcf.helpers.leiden_compact_volume_partition import leiden_compact_volume_partition

def spatial_bootstrap(spatial_network,edge_weight_name,object_indices_A,contributions,weight_matrix=None):
    """
    
    Compute the 95% confidence interval for the pair correlation function at each radius (and target marker, if applicable) using method of spatial bootstrap on spatial networks.
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network on which to compute the confidence intervals. Edges should have a weight attribute corresponding to the distance between nodes.
    edge_weight_name : str
        The name of the edge attribute in the network that corresponds to the distance between nodes. Default is 'Distance'.
    object_indices_A : array-like
        The indices of the nodes in population A for which to compute confidence intervals. This should correspond to the first dimension of the contributions array.
    contributions : numpy.ndarray
        An array of contributions to the pair correlation function for each node in object_indices_A. This should be the output of the compute_contributions_parallel function, and can be either 2D (n_objects_A, number_of_radii) or 3D (n_objects_A, number_of_target_markers, number_of_radii) depending on whether weights are used.
    weight_matrix : numpy.ndarray, optional
        An optional weight matrix to apply to the contributions when computing confidence intervals, required for computing confidence intervals for cross weighted pair correlation functions. This should be of shape (n_objects_A, number_of_target_markers_A). If None, no weights will be applied. Default is None. 
    
    Returns
    -------
    
    confidence_interval : numpy.ndarray 
        An array of confidence intervals for the pair correlation function at each radius (and target marker, if applicable). 
        The shape of this array will depend on the shape of the contributions array and whether weights are used. 
        
        - For 2D contributions, the output will be of shape (2,number_of_radii) where the first dimension corresponds to the lower and upper bounds of the confidence interval. 
        
        - For 3D contributions with weights, the output will be of shape (2, number_of_target_markers, number_of_radii) where the first dimension corresponds to the lower and upper bounds of the confidence interval for each target marker and radius.
        
        - For 4D contributions (cross-weighted case), the output will be of shape (2, number_of_target_markers_A, number_of_target_markers_B, number_of_radii) where the first dimension corresponds to the lower and upper bounds of the confidence interval for each combination of target markers and radius. 
    
    Notes
    -----
    This function extends Loh's method of spatial bootstrap for estimating confidence intervals of the pair correlation function for spatial networks.
    For details, see the reference paper: 
    
    """
    
    
    # Loh's method for interval estimation - tile the domain - sample tiles and compute g - CI of samples of g 
    
    # JWM
    # TODO: redo this - we need to sample blocks of the network, say 1000 times  - then compute g  - repeat  X times and get the CI from that
    # main challenge is sampling 'regions of the spatial network' - the domain is defined by the edges of the network - 
    # therefore we need a computationally efficient way to compute X spatial contiguous regions of the network all with equal sum of edge weights

    # critically, we have bootstrapped g is:  1/(total samples) * sum of contributions in the kernel - this is the pcf but the repeats 
    
    # now could we approximate the tiling using line graphs and communuity detection where the louvain method can be employed to find communities of edges with equal node sums
    
    # this needs to be updated as all_network_distances only computes from population A
    # need to ramdonly sample any node from the network and if the distance isn't in the dictionary, them compute it.
    
    
    
    # First we need to parition our spatial network in to subgraphs of approximately equal size and contiguous with minimal boundaries
    all_node_list = list(spatial_network.nodes())
    has_partition_label = '__partition_label' in spatial_network.nodes[all_node_list[0]]
    number_of_partitions = 9
    if has_partition_label:
        node_to_partition_dict = {node: spatial_network.nodes[node]['__partition_label'] for node in all_node_list}
    else:
                
        partition_paramters = dict(distance_attr=edge_weight_name,
        alpha=0.9,  # large gamma to hit equal-volume target harder , smaller gamma for more compact
        p=2,        # squared distances for tighter clusters
        max_iter=5,
        random_seed=0)
        
        network_partitions = leiden_compact_volume_partition(spatial_network, k=number_of_partitions, **partition_paramters)
        

        node_to_partition_dict = network_partitions.labels  # node -> community id, each community is spatially contiguous
        
        # add the partition labels to the graph
        for node in all_node_list:
            spatial_network.nodes[node]['__partition_label'] = node_to_partition_dict[node]
    
    region_ids = np.unique(list(node_to_partition_dict.values()))
    labels_of_indices_a = np.array([node_to_partition_dict[idx] for idx in object_indices_A])

    # sample regions with replacement - this is equivalent to sampling tiles in loh's method
    # get the contribution for that node 
    number_of_bootstrap_samples = 1000  
    
    # get 1
    toSampleStr = np.random.choice(region_ids,size=(number_of_partitions,number_of_bootstrap_samples))
    toSample = np.zeros(shape=(number_of_partitions,number_of_bootstrap_samples),dtype=int)
    
    if len(contributions.shape) == 2:
        # this is the cross pcf where the contributions are of the form (n_objects_A, number_of_radii)
        samplePCFs = np.zeros(shape=(number_of_bootstrap_samples, np.shape(contributions)[1]))
        quad_contributions = np.zeros((number_of_partitions,np.shape(contributions)[1]))
        quadNs = np.zeros(number_of_partitions)
        for j in range(len(region_ids)):
            quadID = region_ids[j]
            
            sampleMask = toSampleStr == quadID
            
            toSample[sampleMask] = j
            
            accept = labels_of_indices_a==quadID
            if sum(accept) > 0:
                quad_contributions[j,:] = np.sum(contributions[accept,:],axis=0)
                quadNs[j] = sum(accept)
                
                
        sample = np.sum(quad_contributions[toSample,:],axis=0)
        Ns = np.sum(quadNs[toSample],axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            samplePCFs = sample / Ns[:,np.newaxis]

        # Get 95% CI ----- here we need to be use the true PCF values - not the sample mean
        PCF_min = 2*np.mean(contributions,axis=0) - np.nanpercentile(samplePCFs, 97.5, axis=0)
        PCF_max = 2*np.mean(contributions,axis=0) - np.nanpercentile(samplePCFs, 2.5, axis=0)
        PCF_min[PCF_min<0] = 0
        confidence_interval = np.array([PCF_min, PCF_max])
        
    elif len(contributions.shape) == 3:
        # this is the weight pcf where the contributions are of the form (n_objects_A, number_of_target_markers, number_of_radii)
        standard_wpcf = np.mean(contributions, axis=0)  
        samplePCFs = np.zeros(shape=(number_of_bootstrap_samples, contributions.shape[1], contributions.shape[2]))
        quad_contributions = np.zeros((number_of_partitions, contributions.shape[1], contributions.shape[2]))
        quadNs = np.zeros(number_of_partitions)

        for j in range(len(region_ids)):
            quadID = region_ids[j]
            sampleMask = toSampleStr == quadID
            toSample[sampleMask] = j
            accept = labels_of_indices_a == quadID
            
            if sum(accept) > 0:
                quad_contributions[j, :, :] = np.sum(contributions[accept, :, :], axis=0)
                quadNs[j] = sum(accept)

        for b in range(number_of_bootstrap_samples):
            sampled_indices = np.random.choice(range(number_of_partitions), size=number_of_partitions, replace=True)
            sample = np.sum(quad_contributions[sampled_indices, :, :], axis=0)
            Ns = np.sum(quadNs[sampled_indices], axis=0)

            with np.errstate(divide='ignore', invalid='ignore'):
                
                samplePCFs[b, :, :] = sample / Ns   
        
        PCF_min = 2*standard_wpcf - np.nanpercentile(samplePCFs, 97.5, axis=0)
        PCF_max = 2*standard_wpcf - np.nanpercentile(samplePCFs, 2.5, axis=0)
        PCF_min[PCF_min<0] = 0
        
        confidence_interval = np.zeros((2, PCF_min.shape[0], PCF_max.shape[1]))
        confidence_interval[0,:,:] = PCF_min
        confidence_interval[1,:,:] = PCF_max
        
    elif len(contributions.shape) == 4:
        
        # this is the cross-weighted case
        total_marker_contribution_weighting_A = np.sum(weight_matrix, axis=0)
        standard_cross_wpcf = (1/total_marker_contribution_weighting_A)[:,np.newaxis,np.newaxis]*np.sum(contributions, axis=0)
    
        samplePCFs = np.zeros(shape=(number_of_bootstrap_samples, contributions.shape[1], contributions.shape[2],contributions.shape[3]))
        quad_contributions = np.zeros((number_of_partitions, contributions.shape[1], contributions.shape[2],contributions.shape[3]))
        quadNs = np.zeros((number_of_partitions, contributions.shape[1]))
        
        for j in range(len(region_ids)):
            quadID = region_ids[j]
            sampleMask = toSampleStr == quadID
            toSample[sampleMask] = j
            accept = labels_of_indices_a == quadID
            
            if sum(accept) > 0:
                for tau_a_indx in range(contributions.shape[1]):
                    quad_contributions[j, tau_a_indx, :,:] = np.sum(contributions[accept,tau_a_indx , :,:], axis=0)
                    quadNs[j,tau_a_indx] = np.sum(weight_matrix[accept,tau_a_indx])
        
        for b in range(number_of_bootstrap_samples):
            for tau_a_indx in range(contributions.shape[1]):
                sampled_indices = np.random.choice(range(number_of_partitions), size=number_of_partitions, replace=True)
                sample = np.sum(quad_contributions[sampled_indices, tau_a_indx, :,:], axis=0)
                Ns = np.sum(quadNs[sampled_indices,tau_a_indx], axis=0)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                
                    samplePCFs[b, tau_a_indx, :,:] = sample / Ns   
        
        PCF_min = 2*standard_cross_wpcf - np.nanpercentile(samplePCFs, 97.5, axis=0)
        PCF_max = 2*standard_cross_wpcf - np.nanpercentile(samplePCFs, 2.5, axis=0)
        PCF_min[PCF_min<0] = 0
        
        confidence_interval = np.zeros((2, PCF_min.shape[0], PCF_max.shape[1],PCF_max.shape[2]))
        confidence_interval[0,:,:,:] = PCF_min
        confidence_interval[1,:,:,:] = PCF_max
        
    else:
        raise ValueError("The contributions array must be 2D or 3D.")
    
    return confidence_interval
    
  