import sys
sys.path.append('..')

import pytest
import numpy as np
import spacenet as sn

@pytest.fixture
def setup_network_and_params():
    
    sprial_df = sn.datasets.load_dataset('Spiral')
    points = sprial_df[['x','y']].values
    labels = sprial_df['Marker (categorical)'].values
    labels_cont = sprial_df['Marker (continuous)'].values

    G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)
    sn.utils.add_node_labels(G,labels,node_label_name='Marker (categorical)')
    sn.utils.add_node_labels(G,labels_cont,node_label_name='Marker (continuous)')


    nodes_a = sn.utils.query_nodes(G,node_label_name='Marker (categorical)',relation='is',node_label_value='A')
    nodes_b = sn.utils.query_nodes(G,node_label_name='Marker (categorical)',relation='is',node_label_value='B')
    
    return G,nodes_a,nodes_b


def test_polynomial_kernel():
    x_vals = np.array([0,0.25,0.5,0.75,1])
    poly_vals = sn.point_patterns.helpers.polynomial_kernel(x_vals,n=2)
    assert(poly_vals[0] == 3/4)
    assert(poly_vals[-1] == 0)

def test_integrated_poly_finite_kernel():
        
    int_poly_vals = sn.point_patterns.helpers.integrated_poly_finite_kernel(r=np.array([1]),w=np.array([1]),d_1=np.array([0]),d_2=np.array([1]),n=2)
    int_poly_vals_over_extend = sn.point_patterns.helpers.integrated_poly_finite_kernel(r=np.array([2]),w=np.array([1]),d_1=np.array([0]),d_2=np.array([1]),n=2)   
    assert(int_poly_vals[0]==0.5)
    assert(int_poly_vals_over_extend[0]==0)


def test_pair_correlation_function(setup_network_and_params):
    G,_,_ = setup_network_and_params
    r_all,g_all,ci_all=sn.point_patterns.cross_pair_correlation_function(G,spatial_kernel_bandwidth=60,spatial_kernel_n=2,r_max=1000,r_step=5,return_confidence_interval=True)

    assert(len(r_all)==len(g_all)==len(ci_all[0]))
    assert(np.isclose(g_all,1.0,atol=0.05).all())
    assert((ci_all[0]<g_all).all())
    assert((ci_all[1]>g_all).all())
    
def test_cross_pair_correlation_function(setup_network_and_params):
    G,nodes_a,_ = setup_network_and_params
    r_a,g_a,ci_a=sn.point_patterns.cross_pair_correlation_function(G,nodes_a=nodes_a,nodes_b=nodes_a,spatial_kernel_bandwidth=60,spatial_kernel_n=2,r_max=1000,r_step=5,return_confidence_interval=True)
    
    
    assert(len(r_a)==len(g_a)==len(ci_a[0]))
    assert((ci_a[0]<g_a).all())
    assert((ci_a[1]>g_a).all())
    assert(g_a[np.where(r_a==0)[0][0]]>1.5)
    assert(g_a[np.where(r_a==200)[0][0]]<0.5)
    assert(g_a[np.where(r_a==400)[0][0]]>1.5)
    assert(g_a[np.where(r_a==600)[0][0]]<0.5)
    
def test_weighted_pair_correlation_function(setup_network_and_params):
    G,nodes_a,_ = setup_network_and_params
    
    tau,r,g,ci=sn.point_patterns.weighted_pair_correlation_function(G,node_label_name_b='Marker (continuous)',nodes_a=nodes_a,nodes_b=None,spatial_kernel_bandwidth=60,spatial_kernel_n=2,r_max=1000,r_step=5,return_confidence_interval=True)
    
    tau_index_1 = np.where(tau==0)[0][0]
    tau_index_2 = np.where(tau==0.5)[0][0]
    tau_index_3 = np.where(tau==1)[0][0]
    
    
    assert(len(r)==len(g[0,:])==len(ci[0,0,:]))
    assert((ci[0,tau_index_1,:]<g[tau_index_1,:]).all())
    assert((ci[1,tau_index_1,:]>g[tau_index_1,:]).all())

    assert((ci[0,tau_index_2,:]<g[tau_index_2,:]).all())
    assert((ci[1,tau_index_2,:]>g[tau_index_2,:]).all())

    assert((ci[0,tau_index_3,:]<g[tau_index_3,:]).all())
    assert((ci[1,tau_index_3,:]>g[tau_index_3,:]).all())


    assert(g[tau_index_1,np.where(r==0)[0][0]]<1.5)
    assert(g[tau_index_1,np.where(r==200)[0][0]]>0.5)
    assert(g[tau_index_1,np.where(r==400)[0][0]]<1.5)
    assert(g[tau_index_1,np.where(r==600)[0][0]]>0.5)


    assert(g[tau_index_3,np.where(r==0)[0][0]]>1.5)
    assert(g[tau_index_3,np.where(r==200)[0][0]]<0.5)
    assert(g[tau_index_3,np.where(r==400)[0][0]]>1.5)
    assert(g[tau_index_3,np.where(r==600)[0][0]]<0.5)
    
    
def test_cross_weighted_pair_correlation_function(setup_network_and_params):
    G,_,_= setup_network_and_params
    
    tau_a,tau_b,r,g,ci=sn.point_patterns.cross_weighted_pair_correlation_function(G,node_label_name_a='Marker (continuous)',node_label_name_b='Marker (continuous)',spatial_kernel_bandwidth=60,spatial_kernel_n=2,r_max=1000,r_step=5,return_confidence_interval=True)
    
    tau_a_index_1 = np.where(tau_a==0)[0][0]
    tau_a_index_2 = np.where(tau_a==0.5)[0][0]
    tau_a_index_3 = np.where(tau_a==1)[0][0]

    tau_b_index_1 = np.where(tau_b==0)[0][0]
    tau_b_index_2 = np.where(tau_b==0.5)[0][0]
    tau_b_index_3 = np.where(tau_b==1)[0][0]
    
    assert(len(r)==len(g[0,0,:])==len(ci[0,0,0,:]))

    assert((ci[0,tau_a_index_1,tau_b_index_1,:]<=g[tau_a_index_1,tau_b_index_1,:]).all())
    assert((ci[1,tau_a_index_1,tau_b_index_1,:]>=g[tau_a_index_1,tau_b_index_1,:]).all())

    assert((ci[0,tau_a_index_2,tau_b_index_1,:]<=g[tau_a_index_2,tau_b_index_1,:]).all())
    assert((ci[1,tau_a_index_2,tau_b_index_1,:]>=g[tau_a_index_2,tau_b_index_1,:]).all())

    assert((ci[0,tau_a_index_3,tau_b_index_1,:]<=g[tau_a_index_3,tau_b_index_1,:]).all())
    assert((ci[1,tau_a_index_3,tau_b_index_1,:]>=g[tau_a_index_3,tau_b_index_1,:]).all())

    assert((ci[0,tau_a_index_1,tau_b_index_2,:]<=g[tau_a_index_1,tau_b_index_2,:]).all())
    assert((ci[1,tau_a_index_1,tau_b_index_2,:]>=g[tau_a_index_1,tau_b_index_2,:]).all())

    assert((ci[0,tau_a_index_2,tau_b_index_2,:]<=g[tau_a_index_2,tau_b_index_2,:]).all())
    assert((ci[1,tau_a_index_2,tau_b_index_2,:]>=g[tau_a_index_2,tau_b_index_2,:]).all())

    assert((ci[0,tau_a_index_3,tau_b_index_3,:]<=g[tau_a_index_3,tau_b_index_3,:]).all())
    assert((ci[1,tau_a_index_3,tau_b_index_3,:]>=g[tau_a_index_3,tau_b_index_3,:]).all())

    assert(g[tau_a_index_1,tau_b_index_1,np.where(r==0)[0][0]]>1.5)    
    assert(g[tau_a_index_1,tau_b_index_1,np.where(r==100)[0][0]]<0.5)    
    assert(g[tau_a_index_1,tau_b_index_1,np.where(r==200)[0][0]]>1.5)    

    assert(g[tau_a_index_1,tau_b_index_3,np.where(r==0)[0][0]]<0.5)    
    assert(g[tau_a_index_1,tau_b_index_3,np.where(r==100)[0][0]]>1.5)    
    assert(g[tau_a_index_1,tau_b_index_3,np.where(r==200)[0][0]]<0.5)    


