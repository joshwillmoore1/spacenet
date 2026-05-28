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

def test_generate_spatial_network():
    # test a square network with 4 nodes and edges of length 10
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G_5 = sn.utils.generate_spatial_network(coordinates,max_edge_distance=5)
    G_11 = sn.utils.generate_spatial_network(coordinates,max_edge_distance=11)
    G_50 = sn.utils.generate_spatial_network(coordinates,max_edge_distance=50)
    
    
    assert(G_5.number_of_edges()==0), f"Expected 0 edges for max_edge_distance=5, but got {G_5.number_of_edges()}."
    assert(G_11.number_of_edges()==4), f"Expected 4 edges for max_edge_distance=11, but got {G_11.number_of_edges()}."
    assert(G_50.number_of_edges()==5), f"Expected 6 edges for max_edge_distance=50, but got {G_50.number_of_edges()}."
    
    assert(G_5.number_of_nodes()==4),  f"Expected 4 nodes, but got {G_5.number_of_nodes()}."
    assert(G_11.number_of_nodes()==4), f"Expected 4 nodes, but got {G_11.number_of_nodes()}."
    assert(G_50.number_of_nodes()==4), f"Expected 4 nodes, but got {G_50.number_of_nodes()}."
        
    # check Distance is an edge weight
    for u, v, data in G_11.edges(data=True):
        assert 'Distance' in data, f"Edge ({u}, {v}) does not have 'Distance' attribute."
        assert data['Distance'] == 10, f"Edge ({u}, {v}) has 'Distance' attribute with value {data['Distance']} instead of 10."
    
    


def test_add_node_labels(setup_network_and_params):
    this_metwork,netPCF_kwargs = setup_network_and_params
    
    # add some node labels
    labels = np.random.choice(['A', 'B', 'C'], size=this_metwork.number_of_nodes())
    sn.utils.add_node_labels(this_metwork, labels)
    
    # check that the labels were added correctly
    for node in this_metwork.nodes:
        assert 'label' in this_metwork.nodes[node], f"Node {node} does not have a 'label' attribute."
        assert this_metwork.nodes[node]['label'] in ['A', 'B', 'C'], f"Node {node} has an invalid label: {this_metwork.nodes[node]['label']}."


    
def test_delete_node_labels(setup_network_and_params):
    this_metwork,netPCF_kwargs = setup_network_and_params
    
    # add some node labels
    labels = np.random.choice(['A', 'B', 'C'], size=this_metwork.number_of_nodes())
    sn.utils.add_node_labels(this_metwork, labels)
    
    # delete the node labels
    sn.utils.delete_node_labels(this_metwork)
    
    # check that the labels were deleted correctly
    for node in this_metwork.nodes:
        assert 'label' not in this_metwork.nodes[node], f"Node {node} still has a 'label' attribute after deletion." 
    

def test_query_nodes(setup_network_and_params):
    this_metwork,netPCF_kwargs = setup_network_and_params
    
    # add some node labels
    labels = np.random.choice(['A', 'B', 'C'], size=this_metwork.number_of_nodes())
    sn.utils.add_node_labels(this_metwork, labels)
    
    # query nodes with label 'A'
    node_indices_A = sn.utils.query_nodes(this_metwork, node_label_name='label', relation='is', node_label_value='A')
    
    # check that the returned node indices have label 'A'
    for node in node_indices_A:
        assert this_metwork.nodes[node]['label'] == 'A', f"Node {node} does not have label 'A' but was returned in the query."
        
        
def test_distance():
    
    # test a square network with 4 nodes and edges of length 10
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G = sn.utils.generate_spatial_network(coordinates,max_edge_distance=40)
    distance = sn.helpers.node_node_distance(G,sources=np.array(list(G.nodes())),limit=200,verbose=False)
    
    assert(distance[0][0]==0)
    assert(distance[0][1]==10)
    assert(distance[0][2]==10)
    assert(distance[0][3]==np.sqrt(2)*10)
    
    
    
def test_distance_cached():
    
    # test a square network with 4 nodes and edges of length 10
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G = sn.utils.generate_spatial_network(coordinates,max_edge_distance=40)
    _ = sn.helpers.node_node_distance(G,sources=np.array(list(G.nodes())),limit=200,verbose=False)
    
    
    #check the graph now has cache for distances
    assert('Distance' in G.distance_cache)
    
    # get the cached distances
    cached_distance = sn.helpers.node_node_distance(G,sources=np.array(list(G.nodes())),limit=200,verbose=False)
    
    assert(cached_distance[0][0]==0)
    assert(cached_distance[0][1]==10)
    assert(cached_distance[0][2]==10)
    assert(cached_distance[0][3]==np.sqrt(2)*10)
    
    
def test_clear_distance_cached():
    
    # test a square network with 4 nodes and edges of length 10
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G = sn.utils.generate_spatial_network(coordinates,max_edge_distance=40)
    _ = sn.helpers.node_node_distance(G,sources=np.array(list(G.nodes())),limit=200,verbose=False)
    
    
    #check the graph now has cache for distances
    assert('Distance' in G.distance_cache)
    
    # get the cached distances
    sn.helpers.clear_distance_cache(G)
    
    # check the cache was cleared
    assert('Distance' not in G.distance_cache), "Distance cache was not cleared properly."
    
    
    
def test_node_to_dataframe():
    # test a square network with 4 nodes and edges of length 10
    coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*10
    G = sn.utils.generate_spatial_network(coordinates,max_edge_distance=40)
    
    # add some labels
    sn.utils.add_node_labels(G,labels=['A','B','C','D'],node_label_name='celltype')
    sn.utils.add_node_labels(G,labels=['A'],node_label_name='celltype - restricted',nodes=np.array([0]))
    
    test_columns = ['position', 'celltype', 'celltype - restricted']
    
    test_df = sn.utils.nodes_to_dataframe(G)
    
    for col in test_columns:
        assert col in test_df.columns, f"Expected column '{col}' in the node dataframe, but it is missing."
        
    # check that the position column is correct
    for node in G.nodes(data=True):
        node_id = node[0]
        attributes = node[1]
        expected_position = attributes['position']
        actual_position = test_df.loc[node_id, 'position']
        assert np.array_equal(expected_position, actual_position), f"Node {node_id} has incorrect position in the dataframe. Expected {expected_position}, got {actual_position}."
        
    # check that the celltype column is correct
    for node in G.nodes(data=True):
        node_id = node[0]
        attributes = node[1]
        expected_celltype = attributes['celltype']
        actual_celltype = test_df.loc[node_id, 'celltype']
        assert expected_celltype == actual_celltype, f"Node {node_id} has incorrect celltype in the dataframe. Expected {expected_celltype}, got {actual_celltype}."
    
    # check that the celltype - restricted column is correct    
    for node in G.nodes(data=True):
        node_id = node[0]
        attributes = node[1]
        if 'celltype - restricted' in attributes:
            expected_celltype_restricted = attributes['celltype - restricted']

            actual_celltype_restricted = test_df.loc[node_id, 'celltype - restricted']
            assert expected_celltype_restricted == actual_celltype_restricted, f"Node {node_id} has incorrect celltype - restricted in the dataframe. Expected {expected_celltype_restricted}, got {actual_celltype_restricted}."