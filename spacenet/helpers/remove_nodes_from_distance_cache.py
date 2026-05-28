import numpy as np

def remove_nodes_from_distance_cache(spatial_network,nodes_to_remove=None):
    """
    
    Removes specified nodes from the distance cache of a spatial network. 
    This function is used to maintain the integrity of the distance cache when nodes are removed from the spatial network, ensuring that any cached distances involving the removed nodes are also removed from the cache.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to remove nodes from the distance cache.
        
    nodes_to_remove : array-like, optional
        A list or array of node indices to remove from the distance cache. If None, no nodes will be removed from the cache. Default is None.
        
    Returns
    -------
    None
         This function modifies the distance cache of the spatial network in place and does not return any value.
    
    """
    
    
    this_distance_cache = spatial_network.distance_cache
    
    if nodes_to_remove is not None:
        
        if not isinstance(nodes_to_remove,(list,np.ndarray)):
            raise ValueError("nodes_to_remove must be a list or array of node indices.")
        
        nodes_to_remove = np.asarray(nodes_to_remove)
        
        if this_distance_cache != {}:
            
            
            for weight in this_distance_cache.keys():
                
                cached_sources = spatial_network.distance_cache[weight]['source_nodes']
                cached_limit = spatial_network.distance_cache[weight]['current_limit']
                cached_distances = spatial_network.distance_cache[weight]['distances']
                
                # check if any of the nodes to remove are in the cached sources
                mask_sources_to_remove = np.isin(cached_sources,nodes_to_remove,assume_unique=True)
                
                # update the cache by removing the sources and their corresponding limits and distances
                new_cache_sources = cached_sources[~mask_sources_to_remove]
                new_cache_limits = cached_limit[~mask_sources_to_remove]
                
                # remove the distances for the nodes to remove from the remaining sources in the cache
                new_cache_distances = {source: cached_distances[source] for source in new_cache_sources}
                
                # remove the distances to the nodes to remove from the remaining sources in the cache
                nodes_to_remove_set = set(nodes_to_remove)
                
                for source in new_cache_sources:
                    new_cache_distances[source] = {target: dist for target, dist in new_cache_distances[source].items() if target not in nodes_to_remove_set}
                            
                spatial_network.distance_cache[weight]['source_nodes'] = new_cache_sources
                spatial_network.distance_cache[weight]['current_limit'] = new_cache_limits
                spatial_network.distance_cache[weight]['distances'] = new_cache_distances
            
        else:
            
            return None
        
    else:
        return None
   