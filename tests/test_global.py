
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
    G = sn.utils.spatial_network_from_points(coordinates,max_edge_distance=40)
    
    return G

def test_spatial_network_volume(setup_network_and_params):
    G = setup_network_and_params
    total_volume = sn.global_metrics.volume(G)
    
    assert(total_volume==10*4 + 10*np.sqrt(2))
    
    
def test_alpha_index(setup_network_and_params):
    G = setup_network_and_params
    alpha_index = sn.global_metrics.alpha_index(G)
    
    assert(alpha_index==2/3)
    
def test_beta_index(setup_network_and_params):
    G = setup_network_and_params    
    beta_index = sn.global_metrics.beta_index(G)

    assert(beta_index==1.25)
    
def test_gamma_index(setup_network_and_params):
    G = setup_network_and_params    
    gamma_index = sn.global_metrics.gamma_index(G)
    assert(gamma_index==10/12)
    
def test_cyclomatic_number(setup_network_and_params):
    G = setup_network_and_params    
    cyclomatric_number = sn.global_metrics.cyclomatic_number(G) 
    assert(cyclomatric_number==2)
    
def test_diameter(setup_network_and_params):
    G = setup_network_and_params    
    diameter = sn.global_metrics.diameter(G) 
    assert(diameter==20)

def test_mean_clustering(setup_network_and_params):
    G = setup_network_and_params    
    mean_clustering_coefficient = sn.global_metrics.mean_clustering_coefficient(G)  
    mean_clustering_coefficient = np.round(mean_clustering_coefficient,3)
    assert(mean_clustering_coefficient==0.661)
    
def test_mean_detour_index(setup_network_and_params):
    G = setup_network_and_params    
    mean_detour_index = sn.global_metrics.mean_detour_index(G)
    mean_detour_index = np.round(mean_detour_index,3)
    assert(mean_detour_index==1.069)