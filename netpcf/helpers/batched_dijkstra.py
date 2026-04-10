import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm  # optional progress bar

def batched_dijkstra(G, sources, batch_size=5000, weight='Distance',limit=np.inf,verbose=False):
    """
    
    Compute shortest path distances from multiple source nodes to all other nodes in the graph using Dijkstra's algorithm in batches.
    This is useful for large graphs where computing all pairwise shortest paths at once may be memory-intensive.
    
    Parameters
    ----------
    
    G : networkx.Graph
        The input graph on which to compute shortest path distances.
    sources : array-like
        A list or array of source node indices for which to compute shortest path distances to all other nodes in the graph.
    batch_size : int, optional
        The number of source nodes to process in each batch when computing shortest path distances. Default is 5000.
    weight : str, optional
        The name of the edge attribute to use as the weight for computing shortest path distances. Default is 'Distance'.
    limit : float, optional
        The maximum distance to consider when computing shortest path distances. Nodes that are farther than this distance from the source will be ignored. Default is np.inf (no limit).
    verbose : bool, optional
        Whether to print progress messages during computation. Default is False.
    
    Returns
    -------
    
    all_distances : dict
        A dictionary mapping each source node to a dictionary of shortest path distances to all other nodes in the graph. The inner dictionary maps target node indices to their corresponding shortest path distances from the source node.
    
    """
    # Build node index mapping
    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    idx_node = {i: node for node, i in node_idx.items()}
    
    # Sparse adjacency matrix
    A = nx.to_scipy_sparse_array(G, weight=weight, nodelist=nodes, format='csr')
    
    # Map sources to indices
    source_indices = np.array([node_idx[s] for s in sources])
    
    # Batch processing
    all_distances = {}
    
    for i in tqdm(range(0, len(source_indices), batch_size), desc="Batching Dijkstra", disable=not verbose):
        batch = source_indices[i:i + batch_size]
        
        # Run Dijkstra on this batch
        dist_matrix = dijkstra(A, directed=G.is_directed(), indices=batch, unweighted=False,limit=limit,min_only=False)
        
        for j, source_idx in enumerate(batch):
            source_node = idx_node[source_idx]
            distances = {
                idx_node[k]: d for k, d in enumerate(dist_matrix[j]) if not np.isinf(d)
            }
            all_distances[source_node] = distances
            
    return all_distances
