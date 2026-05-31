import numpy as np
from spacenet.helpers import node_node_distance
from spacenet.utils import add_node_labels


def closeness(spatial_network, nodes=None,edge_weight_name='Distance',max_distance=np.inf,add_as_node_label=False,node_label_name='closeness'):
    """
    
    Parameters
    ----------
    
    
    
    Returns
    -------
    
    
    
    
    
    """
    
    
    # if no nodes give run on all
    if nodes is None:
        nodes = np.array(list(spatial_network.nodes))
    
    # get the distances
    node_distances = node_node_distance(spatial_network,sources=nodes,weight=edge_weight_name,limit=max_distance)
    
    # compute closeness for each node
    closenss_values = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        these_distances = node_distances[node]
        this_distance_list = []
        for target in these_distances:
            if target != node:
                this_dist_val = these_distances[target]
                if this_dist_val <= max_distance:
                    this_distance_list.append(this_dist_val)
            
        num_in_reach = len(this_distance_list)
        sum_distance = sum(this_distance_list)
        closenss_values[i] = (num_in_reach**2) / sum_distance
        
        
    # scaling for Wasserman and Faust formula 
    closenss_values = (1/(spatial_network.number_of_nodes()-1))*closenss_values
    
    if add_as_node_label:
        add_node_labels(spatial_network,closenss_values,node_label_name=node_label_name,nodes=nodes)
    
    return closenss_values, nodes