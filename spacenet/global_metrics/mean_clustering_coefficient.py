import numpy as np  
from spacenet.node_metrics.clustering_coefficient import clustering_coefficient

def mean_clustering_coefficient(spatial_network,edge_weight_name='Distance'):
    """
    
    Computes the mean cluster coefficient of a spatial network, which is the average of the clustering coefficients of all nodes in the graph. The clustering coefficient of a node is a measure of how connected its neighbors are to each other, and the mean cluster coefficient provides an overall measure of the tendency of nodes in the graph to cluster together.  
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to compute the mean cluster coefficient.
    edge_weight_name : str, optional
        The name of the edge attribute to use as the weight for computing the cluster coefficient. Default is 'Distance'.
    
    Returns
    -------
    
    mean_clustering_coefficient : float
        The mean cluster coefficient of the spatial network, which is the average of the clustering coefficients of all nodes in the graph.
        
    """
    cluster_coefficients,_ = clustering_coefficient(spatial_network,edge_weight_name=edge_weight_name)
    mean_clustering_coefficient_value = np.mean(cluster_coefficients)
    
    return mean_clustering_coefficient_value