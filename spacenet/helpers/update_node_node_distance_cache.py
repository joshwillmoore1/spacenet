import numpy as np  

def update_node_node_distance_cache(spatial_network,sources,network_distances,weight='Distance',limit=np.inf):
    
    
    # distance cache structure: {'weight' :   {'distances': ... ,'source_nodes': array_of_sources, 'current_limit': array_of_limits }       } where distance_dict is a dict of dicts: {source_node: {target_node: distance, ...}, ...}
    
    
    # check if the network has a node-node distance cache for this weight
    if weight in spatial_network.distance_cache: 
        
        cached_sources = spatial_network.distance_cache[weight]['source_nodes']
        cached_limit = spatial_network.distance_cache[weight]['current_limit']
        
        # check if the sources and limit match the cache
        mask_sources_in_cache = np.isin(cached_sources,sources,assume_unique=True)  # check if all sources are in the cache
        
        if np.sum(mask_sources_in_cache)>0:
            indices_in_cache = np.where(mask_sources_in_cache)[0]   
            cached_limit[indices_in_cache]=limit
        
        # now get the left over sources 
        new_sources = sources[~np.isin(sources,cached_sources,assume_unique=True)]
        if len(new_sources)>0:
            cached_sources = np.concatenate([cached_sources, new_sources])  # combine cached sources and new sources
            cached_limit = np.concatenate([cached_limit, np.array([limit]*len(new_sources))])
        
        # now update the cache with the new distances
        for source in sources:
            if source in network_distances:
                spatial_network.distance_cache[weight]['distances'][source] = network_distances[source]
        
    else:
        # no cache for this weight, create one
        spatial_network.distance_cache[weight] = {'distances': None, 'source_nodes': None, 'current_limit': None}
        sources_for_cache = np.asarray(list(network_distances.keys()))
        these_limits = np.asarray([limit]*len(sources_for_cache))
        
        # update the cache with the new distances, sources, and limits
        spatial_network.distance_cache[weight]['distances'] = network_distances
        spatial_network.distance_cache[weight]['source_nodes'] = sources_for_cache
        spatial_network.distance_cache[weight]['current_limit'] = these_limits
    
    return None