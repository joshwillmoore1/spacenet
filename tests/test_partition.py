
import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn
import networkx as nx   


@pytest.fixture
def setup_network_and_params():
    
    points = np.array([[0, 0], [1, 0], [0.5,0.5], [2,0],[2.5,0.5],[3,0]])
    G = sn.utils.generate_spatial_network(points,network_type='proximity',max_edge_distance=1.5)
    
    return G

def test_modularity_leiden(setup_network_and_params):
    G = setup_network_and_params
    partition_result = sn.partition.modularity_partition(G,algorithm='leiden',resolution=1.0)
    test_leiden = {0:0,1:0,2:0,3:1,4:1,5:1} 
    assert(partition_result.labels == test_leiden)
    
def test_modularity_louvain(setup_network_and_params):
    G = setup_network_and_params
    partition_result = sn.partition.modularity_partition(G,algorithm='louvain',resolution=1.0)
    test_louvain= {0:0,1:0,2:0,3:1,4:1,5:1} 
    assert(partition_result.labels == test_louvain)
    
def test_compact_volume_partition(setup_network_and_params):
    G = setup_network_and_params
    CV_partition_result = sn.partition.compact_volume_partition(G, k=2)
    test_cv = {0:0,1:0,2:0,3:1,4:1,5:1}
    comm_volumes = CV_partition_result.community_volumes

    assert(CV_partition_result.labels == test_cv)
    assert(comm_volumes[0]==comm_volumes[1])