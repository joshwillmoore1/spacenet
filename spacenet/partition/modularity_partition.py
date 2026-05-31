import sknetwork as skn
import networkx as nx
import scipy.sparse as sp

def modularity_partition(spatial_network,edge_weight_name='Inverse Distance',algorithm='leiden',resolution=1.0,modularity='newman'):
    """
    
    Performs modularity-based community detection on a spatial network using the specified algorithm and parameters. 
    The function computes the modularity partition of the spatial network, which identifies communities of nodes that are more densely connected to each other than to the rest of the network.
    The edge weights used for computing the modularity partition can be specified through the `edge_weight_name` parameter, allowing for flexibility in how the community structure is determined based on the spatial relationships between nodes.
    
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to compute the modularity partition.
    
    edge_weight_name : str, optional
        The name of the edge attribute to use as the weight for computing the modularity partition. Default is 'Inverse Distance'.
    
    algorithm : str, optional
        The algorithm to use for community detection. Supported options are 'leiden' and 'louvain'. Default is 'leiden'.
    
    resolution : float, optional    
        The resolution parameter for the community detection algorithm. Higher values lead to smaller communities, while lower values lead to larger communities. Default is 1.0.
    
    modularity : str, optional
        The modularity function to optimize. Supported options are 'newman', 'dugue' and 'potts'. Default is 'newman'. 

    Returns
    -------
    PartitionResult
        A dataclass containing the partitioning results, including:
        
        - labels: A dictionary mapping each node to its assigned community.
        
        - n_clusters: The number of communities detected in the partition.
        
        - modularity: The modularity score of the partition, which is a measure of the strength of the community structure in the graph. Higher modularity values indicate stronger community structure.
    
    Notes
    -----
    References:
    
    - Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 1-12.
    
    - Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
    
    """

    this_adj_matrix = sp.csr_matrix(nx.to_numpy_array(spatial_network, weight=edge_weight_name))
    

    if algorithm.lower() == 'leiden':
        communities = skn.clustering.Leiden(resolution=resolution,modularity=modularity).fit(this_adj_matrix)
        partition_output = PartitionResult(spatial_network,this_adj_matrix,communities)

    elif algorithm.lower() == 'louvain':
        communities = skn.clustering.Louvain(resolution=resolution,modularity=modularity).fit(this_adj_matrix)
        partition_output = PartitionResult(spatial_network,this_adj_matrix,communities)
        
    else:
        raise ValueError(f"Algorithm '{algorithm}' is not supported. Please choose 'leiden' or 'louvain'.")

    return partition_output


# create a cluster object similar to sklearn
class PartitionResult:
    def __init__(self, spatial_graph,this_adj_matrix, communities):
        
        nodes = spatial_graph.nodes()
        labels_dictionary = {node: label for node, label in zip(nodes, communities.labels_)}
        
        self.labels = labels_dictionary
        self.n_clusters = len(set(communities.labels_))
        self.modularity = skn.clustering.get_modularity(this_adj_matrix,communities.labels_)