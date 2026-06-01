import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn
import networkx as nx   
import pandas as pd
from collections import defaultdict



@pytest.fixture
def setup_network_and_params():
    
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0],[4, 0]])
    G = sn.utils.generate_spatial_network(points,network_type='proximity',max_edge_distance=1.5)
    return G

def test_node_reach(setup_network_and_params):
    G = setup_network_and_params
    
    node_reach,node_ids_1 = sn.centrality.node_reach(G,max_distance=2.1,add_as_node_label=True,node_label_name='node_reach')
    assert((node_reach == np.array([0.4, 0.6, 0.8, 0.6, 0.4])).all())
    
def test_laplacian(setup_network_and_params):
    G = setup_network_and_params
    
    laplacian_values,node_ids_2 = sn.centrality.laplacian(G,add_as_node_label=True,node_label_name='laplacian')
    laplacian_values = np.round(laplacian_values, decimals=3)
    assert((laplacian_values == np.array([0.273, 0.545, 0.636, 0.545, 0.273])).all())
   
def test_degree_centrality(setup_network_and_params):
    G = setup_network_and_params
    
    degree_centrality,node_ids_3 = sn.centrality.degree(G,add_as_node_label=True,node_label_name='degree_centrality')
    assert((degree_centrality == np.array([0.25, 0.5, 0.5, 0.5, 0.25])).all())
    
def test_closeness(setup_network_and_params):
    G = setup_network_and_params
    
    closeness_centrality,node_ids_4 = sn.centrality.closeness(G,add_as_node_label=True,node_label_name='closeness_centrality')
    closeness_centrality = np.round(closeness_centrality, decimals=3)
    assert((closeness_centrality == np.array([0.4  , 0.571, 0.667, 0.571, 0.4 ])).all())