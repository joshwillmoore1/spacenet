import networkx as nx

def cyclomatic_number(spatial_network):
    """
    Calculate the cyclomatic number of a network.
    The cyclomatic number counts the number of elementary cycles.
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network for which to compute the cyclomatic number.
    
    Returns
    -------
    cyclomatic_num : int
        The cyclomatic number of the specified network.
        
    Notes
    -----
        The cyclomatic number is calculated using the formula:
        cyclomatic_number = E - N + C for C the number of connected components, E the number of edges, and N the number of nodes.
        
        This is a measure of the complexity of the network, indicating the number of independent paths through the network.
        The cyclomatic number is also known as the first Betti number in algebraic topology.
    """

    
    numEdges=spatial_network.number_of_edges()
    numNodes=spatial_network.number_of_nodes()
    numComp=nx.number_connected_components(spatial_network)
    cyclomatic_num= numEdges-numNodes+numComp
    return cyclomatic_num