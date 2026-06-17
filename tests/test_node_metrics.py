import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn



@pytest.fixture
def setup_network_and_params():
    
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0],[4, 0],[1.5,0.5]])

    G = sn.utils.spatial_network_from_points(points,network_type='proximity',max_edge_distance=1.5)
    return G

def test_degree(setup_network_and_params):
    G = setup_network_and_params
    
    deg_vals,nodes_1 = sn.node_metrics.degree(G,nodes=None,edge_weight_name='Distance',add_as_node_label=True,node_label_name='degree')
    deg_vals=np.round(deg_vals,decimals=3)
    assert((deg_vals == np.array([1.   , 2.707, 2.707, 2.   , 1.   , 1.414])).all())
    
def test_clustering_coefficient(setup_network_and_params):
    G = setup_network_and_params
    
    clustering_vals,nodes_2 = sn.node_metrics.clustering_coefficient(G,nodes=None,edge_weight_name='Distance',add_as_node_label=True,node_label_name='clustering')
    clustering_vals=np.round(clustering_vals,decimals=3)
    assert((clustering_vals == np.array([0.   , 0.265, 0.265, 0.   , 0.   , 0.794])).all())


def test_eccentricity(setup_network_and_params):
    G = setup_network_and_params

    eccentricity_vals,nodes_3 = sn.node_metrics.eccentricity(G,nodes=None,edge_weight_name='Distance',add_as_node_label=True,node_label_name='eccentricity')
    eccentricity_vals=np.round(eccentricity_vals,decimals=3)
    assert((eccentricity_vals == np.array([4.   , 3.   , 2.   , 3.   , 4.   , 2.707])).all())