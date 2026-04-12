import numpy as np
from spacenet.pcf.helpers.polynomial_kernel import polynomial_kernel 
import numba

@numba.njit(
    fastmath=True
)
def integrated_poly_finite_kernel(r,w,d_1,d_2,delta_r=1,n=2):
    """
    
    Computes the integrated polynomial kernel function for given distances and bandwidth parameters. The integrated kernel is defined as:
    
    .. math::
        L_{i} \\left(  r \\right) = \\sum_{e_{nm} \\in E}  \\int_{0}^{w_{nm}} \\kappa_{\\Delta r, n } \\left(  |  d_{in} + \\frac{d_{im} - d_{in}}{w_{nm}}x  - r |  \\right) dx

    The kernel function is integrated over the length of the edge, weighted by the edge weight.   

    Parameters
    ----------
    r : float
        The distance at which to evaluate the integrated kernel function.
    w : numpy.ndarray
        An array of edge weights corresponding to the lengths of the edges in the network.
    d_1 : numpy.ndarray
        An array of distances from the reference node to one endpoint of each edge.
    d_2 : numpy.ndarray
        An array of distances from the reference node to the other endpoint of each edge.
    delta_r : float, optional
        The bandwidth parameter for the polynomial kernel function. Default is 1.
    n : int, optional
        The exponent parameter for the polynomial kernel function. Default is 2.
    
    Returns
    -------
    integrated_kernel_values : numpy.ndarray    
        An array of integrated kernel values from node i at all r values.
    
    Notes
    -----
    See reference paper for details on the derivation of the integrated kernel function and its implementation.
    
    
    """
    
    # Initialize the integrated kernel values as a NumPy array
    integrated_kernel_values = np.zeros_like(w, dtype=np.float64)

    # Handle the special case where d_1 == d_2
    special_case_mask = (d_1 == d_2)
    integrated_kernel_values[special_case_mask] = (
        w[special_case_mask] * polynomial_kernel(np.abs(d_1[special_case_mask] - r), n=n, delta_r=delta_r)
    )
    # Handle the general case where d_1 != d_2
    general_case_mask = ~special_case_mask
    d_a = np.minimum(d_1, d_2)
    d_b = np.maximum(d_1, d_2)

    scale = w / (d_b - d_a)
    upper_limit = np.maximum(0, np.minimum(w, scale * (r - d_a + delta_r)))
    lower_limit = np.maximum(0, np.minimum(upper_limit, scale * (r - d_a - delta_r)))
    M = np.maximum(lower_limit, np.minimum(upper_limit, scale * (r - d_a)))

    valid_mask = (lower_limit >= 0) & (upper_limit <= w) & general_case_mask

    if np.any(valid_mask):
        k = (d_b[valid_mask] - d_a[valid_mask]) / w[valid_mask]
        I_1 = np.zeros_like(k)
        I_2 = np.zeros_like(k)

        nonzero_k_mask = (k != 0)
        
        # find the value d_a, k, M for the nonzero k values
        valid_d_a = d_a[valid_mask]
        d_a_mask = valid_d_a[nonzero_k_mask]
        #d_a_mask = d_a[valid_mask][nonzero_k_mask]
        k_mask = k[nonzero_k_mask]
        
        valid_m_mask = M[valid_mask]  # M is already filtered by valid_mask
        M_mask = valid_m_mask[nonzero_k_mask]
        
        if np.any(nonzero_k_mask):
            
            scale_coef = (1 / (k_mask* (n + 1)))
            
            valid_lower_limit = lower_limit[valid_mask] 
            lower_limit_mask = valid_lower_limit[nonzero_k_mask]
            
            valid_upper_limit = upper_limit[valid_mask]
            upper_limit_mask = valid_upper_limit[nonzero_k_mask]
            
            I_1[nonzero_k_mask] = -(scale_coef * (
                (np.maximum(0, -1 * (d_a_mask + k_mask * M_mask - r)))**(n + 1) -
                (np.maximum(0, -1 * (d_a_mask + k_mask * lower_limit_mask - r)))**(n + 1)
            ))
            I_2[nonzero_k_mask] = (scale_coef * (
                (np.maximum(0, d_a_mask + k_mask * upper_limit_mask - r))**(n + 1) -
                (np.maximum(0, d_a_mask + k_mask * M_mask - r))**(n + 1)
            ))

        integrated_kernel_values[valid_mask] = (
            ((n + 1) / (2 * n * delta_r)) *
            (upper_limit[valid_mask] - lower_limit[valid_mask] -
             (1 / (delta_r**n)) * (I_1 + I_2))
        )
    return integrated_kernel_values