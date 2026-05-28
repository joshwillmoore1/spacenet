
import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn
import networkx as nx   


@pytest.fixture
def setup_network_and_params():
    
    # grid of points 
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G = sn.utils.generate_spatial_network(coordinates,max_edge_distance=40)
    
    return G

def test_spatial_network_volume(setup_network_and_params):
    G = setup_network_and_params
    total_volume = sn.summary.volume(G)
    
    assert(total_volume==10*4 + 10*np.sqrt(2))
    
    
def test_alpha_index(setup_network_and_params):
    G = setup_network_and_params
    alpha_index = sn.summary.alpha_index(G)
    
    assert(alpha_index==2/3)