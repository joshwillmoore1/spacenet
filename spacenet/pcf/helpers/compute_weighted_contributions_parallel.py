from spacenet.pcf.helpers.compute_weighted_contributions import compute_weighted_contributions
import multiprocessing
from tqdm import tqdm

def compute_weighted_contributions_parallel(object_indices_A, object_indices_B, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,these_marker_contributions_weighting,node_to_edges, n_jobs=-1,verbose=True):
    """
    
    Computes the local contributions to the pair correlation function for a list of reference nodes (object_indices_A) and a set of target nodes (object_indices_B) at specified radii (r), weighted by marker contributions, in parallel across multiple CPU cores.
    
    Parameters
    ----------
    object_indices_A : list 
        A list of object indices for which to compute contributions.
    object_indices_B : list
        A list of object indices that contribute to the kernel for each object in A.
    r : numpy.ndarray
        An array of radii at which to compute contributions.
    spatial_kernel_bandwidth : float    
        The bandwidth parameter for the spatial kernel function.
    spatial_kernel_n : float
        The exponent parameter for the spatial kernel function.
    total_length : float
        The total length of the network, used for density normalization.
    all_network_distances : dict
        A dictionary mapping each node index to a dictionary of shortest distances to all other nodes in the network. This should be precomputed for efficiency.
    these_marker_contributions_weighting : np.array
        An array of shape (num_objects_B, num_markers) containing the contributions of each object in population B to each marker. This should be precomputed based on the marker values and the weighting scheme for the contributions.
    node_to_edges : dict
        A dictionary mapping node indices to a list of edges (and their weights) that are connected to that node. This should be precomputed for efficiency.
    n_jobs : int, optional
        The number of parallel jobs to run when computing contributions. If n_jobs > 1, the contributions will be computed in parallel across multiple CPU cores. Default is -1 (use all available cores).
    verbose : bool, optional
        Whether to print progress messages during computation. Default is True.
    
    Returns
    -------
    
    local_contributions : np.array  
        An array of local contributions to the pair correlation function for the reference node at each radius in r, weighted by the marker contributions. The shape of this array will be (num_markers, len(r)).
    
    """
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        
    if verbose:
        print(f"Computing contributions in parallel using {n_jobs} cores...")
    # Prepare arguments for each task
    tasks = [(obj_a, object_indices_B, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances[obj_a],these_marker_contributions_weighting,node_to_edges) for obj_a in object_indices_A]

    # set the chunk size for the parallel processing
    chunk_size = max(1, len(tasks) // (n_jobs * 4))  
    
    # Use ProcessPoolExecutor
    with multiprocessing.Pool(processes=n_jobs) as pool:
        if verbose:
            results = list(tqdm(pool.imap(__process, tasks, chunksize=chunk_size), total=len(tasks),desc="Computing contributions", unit="contributions"))

        else:
            results = list(pool.map(__process, tasks,chunksize=chunk_size))

    return results



def __process(args):
    obj_a, object_indices_B, r, spatial_kernel_bandwidth, spatial_kernel_n, total_length, all_network_distances_obj_a,these_marker_contributions_weighting,node_to_edges = args
    return compute_weighted_contributions(obj_a, object_indices_B, r, spatial_kernel_bandwidth,
                                 spatial_kernel_n, total_length, all_network_distances_obj_a,these_marker_contributions_weighting,node_to_edges)
