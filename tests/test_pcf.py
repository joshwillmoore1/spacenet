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