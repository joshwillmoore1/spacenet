import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn
import networkx as nx   
from collections import defaultdict

from spacenet.pcf.helpers.compute_contributions import compute_contributions
from spacenet.pcf.helpers.compute_weighted_contributions import compute_weighted_contributions
from spacenet.pcf.helpers.polynomial_kernel import polynomial_kernel
from spacenet.pcf.helpers.integrated_poly_finite_kernel import integrated_poly_finite_kernel


def _legacy_compute_contributions(object_id_A, object_indices_B, r, spatial_kernel_bandwidth,
                                  spatial_kernel_n, total_length, this_node_shortest_distance, node_to_edges):
    node_list = np.array(list(this_node_shortest_distance.keys()))
    node_distances = np.array(list(this_node_shortest_distance.values()))
    node_distance_dict = dict(zip(node_list, node_distances))
    total_density = len(object_indices_B) / total_length

    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    which_nodes_in_kernels = [node_list[kernel_r_indicators[:, i]] for i in range(len(r))]

    object_indices_B_set = set(object_indices_B)
    which_nodes_in_kernels_and_in_pop = [
        np.array([node for node in which_nodes_in_kernel if node != object_id_A and node in object_indices_B_set])
        for which_nodes_in_kernel in which_nodes_in_kernels
    ]

    numerator_contributions = np.zeros(len(r), dtype=np.float64)
    total_kernel_lengths = np.ones(len(r), dtype=np.float64)

    for r_index, (r_value, which_nodes_in_kernel_and_in_pop_local) in enumerate(
        zip(r, which_nodes_in_kernels_and_in_pop)
    ):
        if which_nodes_in_kernel_and_in_pop_local.size > 0:
            node_indices = np.isin(node_list, which_nodes_in_kernel_and_in_pop_local)
            distances = np.abs(node_distances[node_indices] - r_value)

            numerator = np.sum(
                polynomial_kernel(
                    distances.astype(np.float64),
                    n=np.float64(spatial_kernel_n),
                    delta_r=np.float64(spatial_kernel_bandwidth)
                )
            )
            numerator_contributions[r_index] = numerator

            edges_seen = set()
            filtered_edges_with_data = {}
            nodes_in_kernel = set(which_nodes_in_kernels[r_index])

            for node in nodes_in_kernel:
                for edge, weight in node_to_edges.get(node, []):
                    if edge not in edges_seen:
                        filtered_edges_with_data[edge] = weight
                        edges_seen.add(edge)

            weights, d_1, d_2 = zip(*(
                (data, node_distance_dict[this_edge[0]], node_distance_dict[this_edge[1]])
                for this_edge, data in filtered_edges_with_data.items()
            ))

            total_length_in_kernel = np.sum(integrated_poly_finite_kernel(
                r=r_value,
                w=np.array(weights, dtype=np.float64),
                d_1=np.array(d_1, dtype=np.float64),
                d_2=np.array(d_2, dtype=np.float64),
                delta_r=spatial_kernel_bandwidth,
                n=spatial_kernel_n
            ))
            total_kernel_lengths[r_index] = total_length_in_kernel

    local_contributions = numerator_contributions / total_kernel_lengths
    return local_contributions / total_density


def _legacy_compute_weighted_contributions(object_id_A, object_indices_B, r, spatial_kernel_bandwidth,
                                           spatial_kernel_n, total_length, this_node_shortest_distance,
                                           these_marker_contributions_weighting, node_to_edges):
    valid_indices = np.isin(object_indices_B, object_id_A, invert=True)
    object_indices_B = object_indices_B[valid_indices]
    these_marker_contributions_weighting = these_marker_contributions_weighting[valid_indices, :]

    node_list = np.array(list(this_node_shortest_distance.keys()))
    node_distances = np.array(list(this_node_shortest_distance.values()))
    node_distance_dict = dict(zip(node_list, node_distances))
    total_density = np.sum(these_marker_contributions_weighting, axis=0, keepdims=True) / total_length

    kernel_r_indicators = (node_distances[:, None] >= (r - spatial_kernel_bandwidth)) & \
                          (node_distances[:, None] <= (r + spatial_kernel_bandwidth))
    which_nodes_in_kernels = [node_list[kernel_r_indicators[:, i]] for i in range(len(r))]

    object_indices_B_set = set(object_indices_B)
    which_nodes_in_kernels_and_in_pop = [
        np.array([node for node in which_nodes_in_kernel if node != object_id_A and node in object_indices_B_set])
        for which_nodes_in_kernel in which_nodes_in_kernels
    ]

    numerator_contributions = np.zeros((these_marker_contributions_weighting.shape[1], len(r)), dtype=np.float64)
    total_kernel_lengths = np.ones((these_marker_contributions_weighting.shape[1], len(r)), dtype=np.float64)

    for r_index, (r_value, which_nodes_in_kernel_and_in_pop_local) in enumerate(
        zip(r, which_nodes_in_kernels_and_in_pop)
    ):
        if which_nodes_in_kernel_and_in_pop_local.size > 0:
            node_indices = np.isin(node_list, which_nodes_in_kernel_and_in_pop_local)
            distances = np.abs(node_distances[node_indices] - r_value)

            these_marker_contributions_weighting_in_kernel = these_marker_contributions_weighting[
                np.isin(object_indices_B, which_nodes_in_kernel_and_in_pop_local, assume_unique=True), :
            ]

            edges_seen = set()
            filtered_edges_with_data = {}
            nodes_in_kernel = set(which_nodes_in_kernels[r_index])

            for node in nodes_in_kernel:
                for edge, weight in node_to_edges.get(node, []):
                    if edge not in edges_seen:
                        filtered_edges_with_data[edge] = weight
                        edges_seen.add(edge)

            weights, d_1, d_2 = zip(*(
                (data, node_distance_dict[this_edge[0]], node_distance_dict[this_edge[1]])
                for this_edge, data in filtered_edges_with_data.items()
            ))

            total_length_in_kernel = np.sum(integrated_poly_finite_kernel(
                r=r_value,
                w=np.array(weights, dtype=np.float64),
                d_1=np.array(d_1, dtype=np.float64),
                d_2=np.array(d_2, dtype=np.float64),
                delta_r=spatial_kernel_bandwidth,
                n=spatial_kernel_n
            ))

            spatial_kernel_contributions = np.array(
                polynomial_kernel(distances, n=spatial_kernel_n, delta_r=spatial_kernel_bandwidth),
                ndmin=1
            )
            these_marker_contributions_weighting_in_kernel *= spatial_kernel_contributions[:, np.newaxis]

            numerator_contributions[:, r_index] = np.sum(these_marker_contributions_weighting_in_kernel, axis=0)
            total_kernel_lengths[:, r_index] = total_length_in_kernel

    local_contributions = numerator_contributions / total_kernel_lengths
    local_contributions /= total_density.T
    return local_contributions

@pytest.fixture
def setup_network_and_params():
    
    # define a spatial region and generate a Poisson point process
    region_size = (1000, 1000)
    lambda_points = 0.001  
    num_points = int(region_size[0] * region_size[1] * lambda_points)
    points = np.random.uniform(0, region_size[0], size=(num_points, 2))
    
    # parameters for the  all pair correlation functions
    netPCF_kwargs = dict(spatial_kernel_bandwidth=75,
                        spatial_kernel_n=2,
                        r_min=0, 
                        r_max=700,
                        r_step=10, 
                        return_confidence_interval=True,
                        verbose=False,
                        n_jobs=1)  

    # a network generated from the points
    G_ppp=sn.utils.generate_spatial_network(points,network_type='delaunay',max_edge_distance=100)
    
    return G_ppp, netPCF_kwargs


def test_null_cross_pcf(setup_network_and_params):
    this_metwork,netPCF_kwargs = setup_network_and_params
    
    
    # compute the cross-PCF
    r,g,CI = sn.pcf.cross_pair_correlation_function(this_metwork,**netPCF_kwargs)
    
    
    # check that g is approximately 1 for all r values
    assert np.allclose(g, 1, atol=0.2), "Cross-PCF deviates significantly from 1 for a homogeneous Poisson process."


def test_compute_contributions_matches_legacy_helper():
    G = nx.path_graph(8)
    for u, v in G.edges():
        G[u][v]['Distance'] = float((u + v) % 3 + 1)

    edges_with_data = {(u, v): data for u, v, data in G.edges(data='Distance')}
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))

    object_id_A = 2
    object_indices_B = np.array(list(G.nodes()))
    r = np.arange(0, 8, 1.0)
    total_length = G.size(weight='Distance')
    this_node_shortest_distance = dict(nx.single_source_dijkstra_path_length(G, object_id_A, weight='Distance'))

    optimized = compute_contributions(
        object_id_A, object_indices_B, r, 1.5, 2, total_length, this_node_shortest_distance, node_to_edges
    )
    legacy = _legacy_compute_contributions(
        object_id_A, object_indices_B, r, 1.5, 2, total_length, this_node_shortest_distance, node_to_edges
    )

    np.testing.assert_allclose(optimized, legacy, rtol=1e-14, atol=1e-14)


def test_compute_weighted_contributions_matches_legacy_helper():
    G = nx.path_graph(8)
    for u, v in G.edges():
        G[u][v]['Distance'] = float((u + v) % 3 + 1)

    edges_with_data = {(u, v): data for u, v, data in G.edges(data='Distance')}
    node_to_edges = defaultdict(list)
    for edge, weight in edges_with_data.items():
        node_to_edges[edge[0]].append((edge, weight))
        node_to_edges[edge[1]].append((edge, weight))

    object_id_A = 2
    object_indices_B = np.array(list(G.nodes()))
    r = np.arange(0, 8, 1.0)
    total_length = G.size(weight='Distance')
    this_node_shortest_distance = dict(nx.single_source_dijkstra_path_length(G, object_id_A, weight='Distance'))
    marker_weights = np.arange(len(object_indices_B) * 3, dtype=float).reshape(len(object_indices_B), 3) + 1.0

    optimized = compute_weighted_contributions(
        object_id_A, object_indices_B, r, 1.5, 2, total_length, this_node_shortest_distance, marker_weights, node_to_edges
    )
    legacy = _legacy_compute_weighted_contributions(
        object_id_A, object_indices_B, r, 1.5, 2, total_length, this_node_shortest_distance, marker_weights, node_to_edges
    )

    np.testing.assert_allclose(optimized, legacy, rtol=1e-14, atol=1e-14)

    
