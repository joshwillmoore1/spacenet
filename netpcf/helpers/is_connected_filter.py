import networkx as nx
import numpy as np

def is_connected_filter(network,nodes_A,nodes_B,filter_largest_connected=False):
    
    if not nx.is_connected(network):
        
        if filter_largest_connected:
            
            all_node_ids = np.asarray(list(network.nodes))
            # extract the largest connected component
            connected_components = list(nx.connected_components(network))   
            largest_cc = max(connected_components, key=len)
            network = network.subgraph(largest_cc).copy()
            
            # filter the nodes_A and nodes_B to be only in the largest connected component
            sub_node_ids = np.asarray(list(network.nodes))
            
            if len(sub_node_ids) / len(all_node_ids) < 0.8:
                raise RuntimeError("After filtering to the largest connected component, less than 80% of the original nodes remain. Please provide a connected network.")
            
            nodes_A=nodes_A[np.isin(nodes_A,sub_node_ids,assume_unique=True)]
            nodes_B=nodes_B[np.isin(nodes_B,sub_node_ids,assume_unique=True)]
            
            print("Warning: The provided network was not connected. The largest connected component has been extracted for analysis.")
            
        else:
            # what we could do here is to extract the largest connected component for the analysis
            raise RuntimeError("netPCF is only defined for connected graphs. The provided network is not connected. Please provide a connected network.")

    return network, nodes_A, nodes_B