import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm  # optional progress bar

def batched_dijkstra(G, sources, batch_size=5000, weight='Distance',limit=np.inf,verbose=False):
    """
    TODO: Docstring for batched_dijkstra.
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
