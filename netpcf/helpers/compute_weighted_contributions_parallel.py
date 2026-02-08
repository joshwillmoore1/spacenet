from netpcf.helpers.compute_weighted_contributions import compute_weighted_contributions
import multiprocessing
from tqdm import tqdm

def compute_weighted_contributions_parallel(object_indices_A, object_indices_B, r, spatial_kernel_bandwidth,
                                   spatial_kernel_n, total_length, all_network_distances,these_marker_contributions_weighting,node_to_edges, n_jobs=-1,verbose=True):
    
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
