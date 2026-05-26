import numpy as np  
import networkx as nx   
from scipy.sparse.csgraph import dijkstra
from spacenet.helpers.batched_dijkstra import batched_dijkstra
from spacenet.helpers.update_node_node_distance_cache import update_node_node_distance_cache    
from spacenet.helpers.get_node_node_distance import get_node_node_distance

def node_node_distance(spatial_network,sources, weight='Distance',limit=np.inf,low_memory=False,verbose=False):

    
    if verbose:
        print('Computing node-node distances...')   
            
    # check if the there is a cache already, use source nodes and current limit to check if we need recompute distances
    recompute_needed=False
    if weight not in spatial_network.distance_cache:
        recompute_needed = True  
    else:
        cached_sources = spatial_network.distance_cache[weight]['source_nodes']
        cached_limit = spatial_network.distance_cache[weight]['current_limit']
        
        # check if the sources and limit match the cache
        in_source_list_mask = np.isin(sources,cached_sources,assume_unique=True)  # check if all sources are in the cache
        in_source_limits = cached_limit[in_source_list_mask]
        
        # not in the cache then neeed to be computed
        new_sources = sources[~in_source_list_mask]
        
        # if in cache but limits are smaller than the requested limit, then need to be recomputed for those sources
        new_sources = np.concatenate([new_sources, sources[in_source_list_mask][in_source_limits < limit]])  # combine new sources and sources that are in cache but with smaller limits
        
        if len(new_sources)>0:
            recompute_needed=True
            sources = new_sources
    
    if recompute_needed:
        
        if low_memory:
            
            network_distances = batched_dijkstra(spatial_network, sources, batch_size=5000, weight=weight,limit=limit,verbose=verbose)
            
        else:
            nodes = list(spatial_network.nodes())
            node_idx = {node: i for i, node in enumerate(nodes)}

            sparse_adj_mat = nx.to_scipy_sparse_array(spatial_network, weight=weight, nodelist=nodes, format='csr')
            
            # Get indices of sources
            sources_idx = [node_idx[s] for s in sources]
            
            # Run Dijkstra from multiple sources independently
            dist_matrix = dijkstra(sparse_adj_mat, directed=False, unweighted=False, indices=sources_idx, limit=limit, min_only=False)
            
            # Convert back to dict form if needed
            network_distances = {sources[i]: {nodes[j]: dist for j, dist in zip(np.flatnonzero(~np.isinf(row)), row[~np.isinf(row)]) } for i, row in enumerate(dist_matrix)}
            
            
        # update network distances in cache 
        update_node_node_distance_cache(spatial_network,sources,network_distances,weight=weight,limit=limit)
        
    
    # get the distance cache from the network and return it
    returned_distance = get_node_node_distance(spatial_network,weight=weight,sources=sources)
        
        
        
    return returned_distance
    