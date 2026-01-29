import numba 

@numba.njit(
    fastmath=True
)
def polynomial_kernel(x, n=2, delta_r=1):
    kernel_values = ((n + 1) / (2 * n * delta_r)) * (1 - (x / delta_r) ** n)
    
    # Replace np.maximum with a Numba-compatible operation
    kernel_values = kernel_values * (kernel_values > 0)
    
    #if np.ndim(kernel_values) == 0 or (np.ndim(kernel_values) == 1 and kernel_values.size == 1):
    #    return kernel_values[()]  # Return a scalar if input was a single float
    return kernel_values