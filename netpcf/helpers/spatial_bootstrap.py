
import numpy as np
import networkx as nx
from netpcf.helpers.leiden_compact_volume_partition import leiden_compact_volume_partition

def spatial_bootstrap(this_network,edge_weight_name,object_indices_A,contributions,all_network_distances,number_sample_windows=100,percentile=0.95,sample_radius=200,weight_matrix=None):
    
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
    all_node_list = list(this_network.nodes())
    has_partition_label = '__partition_label' in this_network.nodes[all_node_list[0]]
    number_of_partitions = 9
    if has_partition_label:
        node_to_partition_dict = {node: this_network.nodes[node]['__partition_label'] for node in all_node_list}
    else:
                
        partition_paramters = dict(distance_attr=edge_weight_name,
        alpha=0.9,  # large gamma to hit equal-volume target harder , smaller gamma for more compact
        p=2,        # squared distances for tighter clusters
        max_iter=5,
        random_seed=0)
        
        network_partitions = leiden_compact_volume_partition(this_network, k=number_of_partitions, **partition_paramters)
        

        node_to_partition_dict = network_partitions.labels  # node -> community id, each community is spatially contiguous
        
        # add the partition labels to the graph
        for node in all_node_list:
            this_network.nodes[node]['__partition_label'] = node_to_partition_dict[node]
    
    region_ids = np.unique(list(node_to_partition_dict.values()))
    labels_of_indices_a = np.array([node_to_partition_dict[idx] for idx in object_indices_A])

    # sample regions with replacement - this is equivalent to sampling tiles in loh's method
    # get the contribution for that node 
    number_of_bootstrap_samples = 1000  
    
    samplePCFs = np.zeros(shape=(number_of_bootstrap_samples, np.shape(contributions)[1]))
    toSampleStr = np.random.choice(region_ids,size=(number_of_partitions,number_of_bootstrap_samples))
    toSample = np.zeros(shape=(number_of_partitions,number_of_bootstrap_samples),dtype=int)
    
    
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
    
    samplePCFs = sample / Ns[:,np.newaxis]

    # Get 95% CI ----- here we need to be use the true PCF values - not the sample mean
    PCF_min = 2*np.mean(contributions,axis=0) - np.percentile(samplePCFs, 97.5, axis=0)
    PCF_max = 2*np.mean(contributions,axis=0) - np.percentile(samplePCFs, 2.5, axis=0)
    PCF_min[PCF_min<0] = 0
    confidence_interval = np.array([PCF_min, PCF_max])
    
    
    # TODO: UPDATED FOR weighted and weight-weight PCF
    
    # OLD METHOD - kept for reference
    
    if False:    
        all_node_list = list(this_network.nodes())
        # randomly choose a source node and find all other source nodes within a certain distance
        sample_source_nodes = np.random.choice(all_node_list, size=number_sample_windows,replace=True)
            
        node_idx = {node: i for i, node in enumerate(all_node_list)}

        sparse_adj_mat = nx.to_scipy_sparse_array(this_network, weight=edge_weight_name, nodelist=all_node_list, format='csr')

        # Get indices of sources
        sources_idx = [node_idx[s] for s in sample_source_nodes]

        # Run Dijkstra from multiple sources independently
        dist_matrix = dijkstra(sparse_adj_mat, directed=False, unweighted=False, indices=sources_idx, limit=sample_radius, min_only=False)

        # Convert back to dict form if needed
        all_network_distances = {sample_source_nodes[i]: {all_node_list[j]: dist for j, dist in zip(np.flatnonzero(~np.isinf(row)), row[~np.isinf(row)]) } for i, row in enumerate(dist_matrix)}
            

        if len(contributions.shape) == 2:
            distribution_of_g_estimates = np.zeros((len(sample_source_nodes), contributions.shape[1]))
        elif len(contributions.shape) == 3:
            distribution_of_g_estimates = np.zeros((len(sample_source_nodes), contributions.shape[1], contributions.shape[2]))
        elif len(contributions.shape) == 4:
            distribution_of_g_estimates = np.zeros((len(sample_source_nodes), contributions.shape[1], contributions.shape[2],contributions.shape[3]))
            
        else:
            raise ValueError("The contributions array must be 2D or 3D.")

        for sample_index,sample in enumerate(sample_source_nodes):
            
            this_distances=all_network_distances[sample]
            
            # Convert node distances and list to NumPy arrays for vectorized operations
            node_list = np.array(list(this_distances.keys()))
            node_distances = np.array(list(this_distances.values()))

            # Precompute kernel indicators and filter nodes in kernels
            kernel_r_indicators = (node_distances <= (sample_radius))
            
            # Filter nodes based on kernel_r_indicators
            nodes_in_kernel = node_list[kernel_r_indicators]
            
            # now only get those in the object A population
            _, indices_in_objects_A,_ = np.intersect1d(object_indices_A,nodes_in_kernel, return_indices=True)
            
            if len(indices_in_objects_A) == 0:
                # If no nodes in the kernel, skip this sample
                distribution_of_g_estimates[sample_index,:] = np.zeros(contributions.shape[1])
                continue
            else:
                distribution_of_g_estimates[sample_index,:] = np.mean(contributions[indices_in_objects_A,:], axis=0)
            
            # this would need to be the weighted mean if we are using a weight matrix - then the percentile can be taken as normal - weights are happening locally
        
        if weight_matrix is None:
        
            # Calculate the x percent confidence interval
            #lower_bound = np.percentile(distribution_of_g_estimates, 100*(1-percentile), axis=0)
            #upper_bound = np.percentile(distribution_of_g_estimates, 100*percentile, axis=0)
            
            z_value = NormalDist().inv_cdf((1 + percentile) / 2.)
            lower_bound = np.mean(distribution_of_g_estimates,axis=0) - z_value * np.std(distribution_of_g_estimates, axis=0) / np.sqrt(distribution_of_g_estimates.shape[0])
            upper_bound = np.mean(distribution_of_g_estimates,axis=0) + z_value * np.std(distribution_of_g_estimates, axis=0) / np.sqrt(distribution_of_g_estimates.shape[0])
            lower_bound[lower_bound < 0] = 0
            
            
            if len(contributions.shape) == 2:
                confidence_interval = np.vstack([lower_bound,upper_bound])
            elif len(contributions.shape) == 3:
                confidence_interval = np.zeros(( lower_bound.shape[0], lower_bound.shape[1],2))
                confidence_interval[:,:,0] = lower_bound
                confidence_interval[:,:,1] = upper_bound
                
            elif len(contributions.shape) == 4:
                raise NotImplementedError("Confidence intervals for 4D contributions are not implemented - weights needs to be considered.")
                
            else:
                raise ValueError("The lower_bound array must be 1D or 2D.")
        
        
    return confidence_interval