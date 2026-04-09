import numba 

@numba.njit(
    fastmath=True
)
def polynomial_kernel(x, n=2, delta_r=1):
    """
    Computes the polynomial kernel function for a given distance or array of distances. The polynomial kernel is defined as:
    
    .. math::

        \\kappa_{\\Delta r, n}(x)
        = \\max\\left(0, \\frac{n+1}{2n \\Delta r}
        \\left(1 - \\left(\\frac{x}{\\Delta r}\\right)^n\\right)\\right)
    
           

    Parameters
    ----------
    x : float or numpy.ndarray
        The distance(s) at which to evaluate the kernel function.
    n : int, optional
        The exponent parameter for the polynomial kernel. Default is 2.
    delta_r : float, optional
        The bandwidth parameter for the kernel function. Default is 1.
        
    Returns
    -------
    kernel_values : float or numpy.ndarray
        The value(s) of the polynomial kernel function at the given distance(s).
    
    """
    kernel_values = ((n + 1) / (2 * n * delta_r)) * (1 - (x / delta_r) ** n)
    
    # Replace np.maximum with a Numba-compatible operation
    kernel_values = kernel_values * (kernel_values > 0)
    
    #if np.ndim(kernel_values) == 0 or (np.ndim(kernel_values) == 1 and kernel_values.size == 1):
    #    return kernel_values[()]  # Return a scalar if input was a single float
    return kernel_values