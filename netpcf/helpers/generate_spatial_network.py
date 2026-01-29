import numpy as np
import networkx as nx
from scipy.spatial import Delaunay,cKDTree
from itertools import chain
import copy


def generate_spatial_network(points,network_type='Delaunay',inverse_distance_function=None,min_edge_distance=0,max_edge_distance=np.inf,number_of_nearest_neighbours=10):
    """
    Generate a spatial network using set of points.
    Edges will be created between objects based on the selected network type and distance constraints.
    Edge weights for 'Distance' and 'Inverse Distance' will be added to the network. 
    Node indices correspond to the indices of the input points array.
    
    Parameters
    ----------
    domain : object
        The domain containing the objects to be used for network generation.
    network_type : str, optional
        The type of network to generate. Options are 'Delaunay' (Delaunay triangulation), 'KNN' (K-nearest neighbour), 'Proximity' and 'RNG' (relative neighbourhood graph).
    inverse_distance_function : callable, optional
        Function to compute inverse distance, by default None.
    min_edge_distance : float, optional
        Minimum edge distance, by default 0.
    max_edge_distance : float, optional
        Maximum edge distance, by default np.inf.
    number_of_nearest_neighbours : int, optional
        Number of nearest neighbours for KNN. Only used for KNN networks, by default 10.
   
    Returns
    -------
    networkx.Graph
        The generated network. KNN networks will return a directed network (networkx.DiGraph).
        
    Notes
    -----
    
    - Delaunay networks estimates a natural geometric triangulation for pointclouds. In spatial applications, a Delaunay network approximates contact-based connectivity using only centroid data. See `Delaunay networks <https://en.wikipedia.org/wiki/Delaunay_triangulation>`_ for more information.

    - K-Nearest Neighbours (KNN) networks estimates the local connectivity of points in space by connecting any object to it's k closest objects. See `KNN networks <https://en.wikipedia.org/wiki/Nearest_neighbor_graph>`_ for more information.

    - Proximity networks define connectivity purely by distance thresholds. See `Proximity networks <https://www.sciencedirect.com/science/article/pii/S037015731000308X>`_ for more information.
    
    - Relative Neighbourhood Graph (RNG) esimtates the natural connectivity of points in space. MuSpAn implements the Urquhart approximation to the RNG for computational efficiency. See `RNGs <https://en.wikipedia.org/wiki/Relative_neighborhood_graph>`_ for more information.
    
    If parallisation is producing a `ValueError: cannot pickle '_thread._local' object`, this is likely due to the `domain` object not being serializable. In this case, add `if __name__ == "__main__":` to the top of your script or you can set `n_jobs=1` to disable parallisation.
    
    Examples
    --------
    
    Example of generating a Delaunay network from all objects in the domain:
    
    .. code-block:: python
    
        # To complete later...
                           
    """
    
    
    current_implemented_network_generators = ['delaunay','knn','proximity','rng']
    if network_type.lower() not in current_implemented_network_generators:
        raise ValueError(f'The network type: {network_type} is not currently implemented, try one of {current_implemented_network_generators}')

    if inverse_distance_function is not None:
        #check if the inverse distance function is callable
        if callable(inverse_distance_function):
            this_inverse_distance_function=inverse_distance_function
        else:
            raise ValueError(f"Parameter inverse_distance_function is not callable: please pass a function that takes a distance as an argument and returns a value") 
    else:
        this_inverse_distance_function=lambda x: 1/(1+(4/max_edge_distance)*x)    
    
    
    if min_edge_distance > max_edge_distance:
        raise ValueError(f"Minimum edge distance {min_edge_distance} is greater than maximum edge distance {max_edge_distance}")
    
    
    object_positions=points
    object_indices=np.arange(len(points),dtype=int)
    if network_type.lower()=='delaunay':
        
        delaunay_network = Delaunay(object_positions)
        
        # Extract the edge list from the Delaunay triangulation
        simplices = delaunay_network.simplices
        edge_list = np.vstack(list(edge for simplex in simplices for edge in zip(simplex, np.roll(simplex, -1))))

        # Compute distances and filter edges in a single step
        distances = np.linalg.norm(object_positions[edge_list[:, 0]] - object_positions[edge_list[:, 1]], axis=1)
        valid_edges_mask = (distances <= max_edge_distance) & (distances >= min_edge_distance)
        valid_edges = edge_list[valid_edges_mask]
        valid_distances = distances[valid_edges_mask]
                
        
        # Prepare edge lists with distances and inverse distances
        final_edge_list = np.hstack([valid_edges, valid_distances[:, None]])
        final_edge_list_inv = np.hstack([valid_edges, this_inverse_distance_function(valid_distances)[:, None]])
        
        # initalise the graph
        G=nx.Graph()
        G.add_nodes_from(object_indices)
        
    elif network_type.lower()=='rng':
        
        delaunay_network = Delaunay(object_positions)
        #(indptr, indices) = delaunay_network.vertex_neighbor_vertices
        
        # Extract the edge list from the Delaunay triangulation
        simplices = delaunay_network.simplices
        edge_list = np.vstack(list(edge for simplex in simplices for edge in zip(simplex, np.roll(simplex, -1))))

        # Compute distances and filter edges in a single step
        distances = np.linalg.norm(object_positions[edge_list[:, 0]] - object_positions[edge_list[:, 1]], axis=1)
        
        # Create a dictionary to store edges and their distances for quick lookup
        edge_distance_dict = {(min(edge[0], edge[1]), max(edge[0], edge[1])): dist for edge, dist in zip(edge_list, distances)}

        # Use a set to track edges to remove
        edges_to_remove = set()
        # Iterate over each triangle (simplex) in the Delaunay triangulation
        for simplex in simplices:
            # Extract the edges of the triangle
            edges = [(simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3)]
            for u, v in edges:
                distance_uv = edge_distance_dict[(min(u, v), max(u, v))]
                for w in simplex:
                    if w != u and w != v:
                        distance_uw = edge_distance_dict.get((min(u, w), max(u, w)), float('inf'))
                        distance_wv = edge_distance_dict.get((min(v, w), max(v, w)), float('inf'))
                        if distance_uw < distance_uv and distance_wv < distance_uv:
                            edges_to_remove.add((min(u, v), max(u, v)))
                            break

        # Filter edges and distances efficiently
        mask = [edge not in edges_to_remove for edge in map(lambda e: (min(e[0], e[1]), max(e[0], e[1])), edge_list)]
        edge_list = edge_list[mask]
        distances = distances[mask]
        
        valid_edges_mask = (distances <= max_edge_distance) & (distances >= min_edge_distance)
        valid_edges = edge_list[valid_edges_mask]
        valid_distances = distances[valid_edges_mask]
        
        
        # Prepare edge lists with distances and inverse distances
        final_edge_list = np.hstack([valid_edges, valid_distances[:, None]])
        final_edge_list_inv = np.hstack([valid_edges, this_inverse_distance_function(valid_distances)[:, None]])
        
        # intialise the graph
        G=nx.Graph()
        G.add_nodes_from(object_indices)
            

    elif network_type.lower()=='knn':
        
        tree = cKDTree(object_positions)
        nn_distance, nn_indices = tree.query(object_positions, k=number_of_nearest_neighbours+1,distance_upper_bound=max_edge_distance*1.001)  
        
        # remove self connections
        nn_indices = nn_indices[:,1:]
        nn_distance = nn_distance[:,1:]

        flat_dists=nn_distance.flatten()
        end_edges=nn_indices.flatten()

        # make an array with columns: start node, end node, distance
        start_edges = np.repeat(np.arange(len(object_positions),dtype=int), number_of_nearest_neighbours)
        final_edge_list = np.column_stack([start_edges, end_edges, flat_dists])
        
        # Filter edges based on distance constraints
        valid_edges_mask = (final_edge_list[:,-1] <= max_edge_distance) & (final_edge_list[:,-1] >= min_edge_distance)
        final_edge_list = final_edge_list[valid_edges_mask]
        
        # node centroid indices to object indices
        

        # Prepare edge lists with distances and inverse distances
        final_edge_list_inv = copy.deepcopy(final_edge_list)
        final_edge_list_inv[:, -1] = this_inverse_distance_function(final_edge_list[:, -1])
        
        # intialise the graph - note Digraph for knn
        G=nx.DiGraph()  
        G.add_nodes_from(object_indices)            
                        
    elif network_type.lower()=='proximity':
        #object_positions=np.vstack([domain.objects[v].centroid for v in object_indices])
        #object_types=np.array([domain.objects[id].object_type for id in object_indices])
        only_points=True
        
        # now use the cKDTree to find and upper bound on the neighbours of the objects                 
        kd_tree = cKDTree(object_positions)
        sDistances=kd_tree.query_ball_point(object_positions, r=2*max_edge_distance, p=2, eps=1e-4, workers=1, return_sorted=None, return_length=False)
        
        # Use a generator to create edges and Flatten the sDistances array and generate edges
        global_edges = np.array(list(chain.from_iterable(((i, k) for k in edges if i != k) for i, edges in enumerate(sDistances))),dtype=np.int32)

        # Use a set to handle unique edges for scalability
        unique_global_edges_set =set(map(tuple, map(sorted, global_edges)))

        # Convert the set back to a numpy array
        unique_global_edges = np.array(list(unique_global_edges_set), dtype=np.int32)

        # Compute the distances between the unique edges
        distances = np.linalg.norm(object_positions[unique_global_edges[:, 0]] - object_positions[unique_global_edges[:, 1]], axis=1)
            
        # Filter edges based on distance constraints
        valid_mask = (distances <= max_edge_distance) & (distances >= min_edge_distance)
        valid_edges = unique_global_edges[valid_mask]
        valid_distances = distances[valid_mask]
        
        # node centroid indices to object indices

        # Prepare edge lists with distances and inverse distances
        final_edge_list = np.hstack([valid_edges, valid_distances[:, None]])
        final_edge_list_inv = np.hstack([valid_edges, this_inverse_distance_function(valid_distances)[:, None]])
        
        # if there are any other object type then we need to be careful about approximating distance with bounding spheres to then compute the exact distance   
        
        # intialise the graph
        G=nx.Graph()
        G.add_nodes_from(object_indices)
                    
    else:
        raise ValueError(f"Network type {network_type} is not currently implemented - should not reach this point")
    
    # probably should bring out the distance filtering to here as it's the same for all networks
    # Filter edges based on distance constraints
    
    # dump the edges into the network
    G.add_weighted_edges_from(final_edge_list, weight='Distance')
    G.add_weighted_edges_from(final_edge_list_inv, weight='Inverse Distance')

    
    return G
