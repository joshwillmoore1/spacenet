import numpy as np

def polynomial_kernel_bandwidth_scale(spatial_kernel_n,proportion_kernel_mass=0.75):
    """
    Provides an explicit solution to the polynomial form of the kernel mass ratio. 
    The scale factor can be applied to the ansatz for the kernel bandwidth as derived in the paper to obtain the appropriate bandwidth for a given spatial kernel exponent and desired proportion of kernel mass within that bandwidth.

    
    Parameters
    ----------
    
    spatial_kernel_n : int
        The exponent parameter for the spatial kernel function. This controls the shape of the kernel.
    proportion_kernel_mass : float, optional
        The proportion of the kernel mass that should be contained within the bandwidth. Default is 0.75 (i.e., 75% of the kernel mass should be within the bandwidth).
        
    Returns
    -------
    scale_factor : float
        The scale factor by which to multiply the characteristic length of network edges to obtain the spatial kernel bandwidth that contains the specified proportion of the kernel mass.
    
    Notes
    -----
    
    This functional form is derived in the paper Moore et al. 2026, see ...
        
    """
    
    
    # explicit solutions to polynomial form of kernel mass ratio
    c1= (1/(1-np.sqrt(1-proportion_kernel_mass))) - 1/proportion_kernel_mass
    c2 = 1/(2*np.cos((1/3)*(5*np.pi-np.arccos(proportion_kernel_mass)))) - 1/proportion_kernel_mass
    
    B = (2*c2 - c1)/(c1-c2)
    A = c1*(B+1)
    
    # apply to the ansatz for the kernel bandwidth as derived in the paper Moore et al. 2026
    scale_factor = (A / (B + spatial_kernel_n)) + 1/proportion_kernel_mass
    
    return scale_factor
    
