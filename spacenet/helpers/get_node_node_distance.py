def get_node_node_distance(spatial_network,weight='Distance',sources=None):
        
    this_distance_dictionary=spatial_network.distance_cache[weight]['distances']
    
    # get a subset of the distance dictionary for the sources of interest   
    if sources is not None:
        node_node_distances = {source: this_distance_dictionary[source] for source in sources if source in this_distance_dictionary}
    else:
        node_node_distances = this_distance_dictionary
    
    return node_node_distances