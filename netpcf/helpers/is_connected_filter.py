import networkx as nx
import numpy as np

def is_connected_filter(network,nodes_A,nodes_B,filter_largest_connected=False):
    """
    
    A helper function to check if the network is connected and filter to the largest connected component if not. 
    This is important for the netPCF as it is only defined for connected graphs. 
    If the network is not connected, we can either filter to the largest connected component or raise an error.
    
    Parameters
    ----------
    network : networkx.Graph
        The input network on which the correlation function will be computed.
    nodes_A : numpy.ndarray
        An array of node indices corresponding to the first population of interest.
    nodes_B : numpy.ndarray
        An array of node indices corresponding to the second population of interest.
    filter_largest_connected : bool, optional
        If True, if the network is not connected, the largest connected component will be extracted for analysis. If False, a RuntimeError will be raised if the network is not connected. Default is False.
    
    Returns
    -------
    network : networkx.Graph
        The (possibly filtered) network that will be used for analysis.
    nodes_A : numpy.ndarray
        The (possibly filtered) array of node indices for the first population of interest.
    nodes_B : numpy.ndarray
        The (possibly filtered) array of node indices for the second population of interest.
    
    
    """
    if not nx.is_connected(network):
        
        if filter_largest_connected:
            
            all_node_ids = np.asarray(list(network.nodes))
            # extract the largest connected component
            connected_components = list(nx.connected_components(network))   
            largest_cc = max(connected_components, key=len)
            network = network.subgraph(largest_cc).copy()
            
            # filter the nodes_A and nodes_B to be only in the largest connected component
            sub_node_ids = np.asarray(list(network.nodes))
            
            if len(sub_node_ids) / len(all_node_ids) < 0.6:
                raise RuntimeError("After filtering to the largest connected component, less than 60% of the original nodes remain. Please provide a connected network.")
            
            nodes_A=nodes_A[np.isin(nodes_A,sub_node_ids,assume_unique=True)]
            nodes_B=nodes_B[np.isin(nodes_B,sub_node_ids,assume_unique=True)]
            
            print("Warning: The provided network was not connected. The largest connected component has been extracted for analysis.")
            
        else:
            # what we could do here is to extract the largest connected component for the analysis
            raise RuntimeError("netPCF is only defined for connected graphs. The provided network is not connected. Please provide a connected network.")

    return network, nodes_A, nodes_B