import numpy as np
import numba
import math
from functools import lru_cache
from spacenet.point_patterns.helpers.polynomial_kernel import polynomial_kernel

@lru_cache(maxsize=16)
def _cached_leggauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


@numba.njit(fastmath=True, inline="always")
def _integrate_interval(
    a, b, t0, w, delta_r, n, inv_2c_rel2,
    nodes, weights
):
    """
    Integrate over dimensionless interval [a,b].

    The interval is already clipped to kernel
    support and must not cross t = 0.5.
    """
    if b <= a:
        return 0.0

    mid = 0.5 * (a+b)
    half = 0.5 * (b-a)
    scale = w/delta_r

    total = 0.0

    for k in range(len(nodes)):
        t = mid + half*nodes[k]

        # Polynomial kernel
        u = abs(t-t0)*scale
        kernel = 1.0-u**n

        # Distance to nearest endpoint
        nearest = min(t, 1.0-t)

        # Nearest-endpoint Gaussian
        q = math.exp(
            -nearest*nearest*inv_2c_rel2
        )

        total += weights[k]*q*kernel

    return half*total


@numba.njit(fastmath=True, inline="always")
def _integrate_branch_fast(
    a, b, t0, w, delta_r, n, inv_2c_rel2,
    nodes, weights
):
    """
    Integrate one monotonic geodesic branch.

    Clips to kernel support and splits at:
      - kernel centre t0
      - edge midpoint 0.5
    """
    radius = delta_r/w

    lo = max(a, t0-radius)
    hi = min(b, t0+radius)

    if hi <= lo:
        return 0.0

    # Up to three intervals after splitting
    # at t0 and 0.5.
    cuts = np.empty(4, dtype=np.float64)
    count = 2
    cuts[0] = lo
    cuts[1] = hi

    if lo < t0 < hi:
        cuts[count] = t0
        count += 1

    if lo < 0.5 < hi:
        cuts[count] = 0.5
        count += 1

    # Sort at most four values
    for i in range(count):
        for j in range(i+1, count):
            if cuts[j] < cuts[i]:
                tmp = cuts[i]
                cuts[i] = cuts[j]
                cuts[j] = tmp

    total = 0.0

    for i in range(count-1):
        total += _integrate_interval(
            cuts[i], cuts[i+1],
            t0, w, delta_r, n,
            inv_2c_rel2, nodes, weights
        )

    return total


@numba.njit(fastmath=True)
def _integrated_smooth_fast(
    r, w, d_1, d_2, delta_r, n, c,
    nodes, weights
):
    result = np.zeros(len(w), dtype=np.float64)

    kernel_prefactor = (n+1.0)/(2.0*n*delta_r)

    sqrt2pi = math.sqrt(2.0*math.pi)
    sqrt2 = math.sqrt(2.0)

    for j in range(len(w)):
        wi = w[j]
        d1 = d_1[j]
        d2 = d_2[j]

        if wi <= 0.0:
            continue

        if not (
            np.isfinite(d1) and np.isfinite(d2)
        ):
            continue

        # Physical confidence scale relative to edge
        c_rel = c/wi
        inv_2c_rel2 = 0.5/(c_rel*c_rel)

        # Edge-specific normalisation
        Z = (
            c_rel*sqrt2pi
            *math.erf(1.0/(2.0*sqrt2*c_rel))
        )

        # Shortest-path switching point
        ts = (d2+wi-d1)/(2.0*wi)
        ts = min(1.0, max(0.0, ts))

        # Kernel centres
        t1 = (r-d1)/wi
        t2 = 1.0-(r-d2)/wi

        # Integrate both geodesic branches
        v1 = _integrate_branch_fast(
            0.0, ts, t1, wi,
            delta_r, n, inv_2c_rel2,
            nodes, weights
        )

        v2 = _integrate_branch_fast(
            ts, 1.0, t2, wi,
            delta_r, n, inv_2c_rel2,
            nodes, weights
        )

        result[j] = (
            wi*kernel_prefactor*(v1+v2)/Z
        )

    return result


@numba.njit(inline="always")
def _poly_kernel_antiderivative(u, dr, n):
    if u <= -dr:
        return -0.5

    if u >= dr:
        return 0.5

    z = abs(u) / dr

    return (
        (n + 1.0) / (2.0 * n)
        * (u / dr)
        * (1.0 - z**n / (n + 1.0))
    )



def integrated_poly_finite_kernel(
    r, w, d_1, d_2, delta_r=1.0, n=2.0,
    redistribution_scale=1.0, quadrature_order=16
):
    """
    Nearest-endpoint Gaussian weighted
    polynomial kernel integration.

    Parameters
    ----------
    r : float
        Target geodesic separation.
    w : ndarray
        Edge lengths.
    d_1, d_2 : ndarray
        Endpoint shortest-path distances.
    delta_r : float
        Polynomial kernel support radius.
    n : float
        Positive real kernel exponent.
    redistribution_scale : float
        Physical confidence length scale.
    quadrature_order : int
        Gauss-Legendre quadrature order.

    Returns
    -------
    ndarray
        Integrated contribution for each edge.
    """
    if delta_r <= 0 or n <= 0 or redistribution_scale < 0:
        raise ValueError(
            "delta_r and n must be positive, c must be non-negative"   
        )
    
    # Endpoint-weighted redistribution scale: use simple formula
    if redistribution_scale==0:
        return 0.5 * w * (
        polynomial_kernel(
            np.abs(d_1 - r),
            n=n,
            delta_r=delta_r
        )
        +
        polynomial_kernel(
            np.abs(d_2 - r),
            n=n,
            delta_r=delta_r
        )
    )
        
    # uniform redistribution scale: use exact analytical formula
    elif  np.isinf(redistribution_scale):
        result = np.zeros(len(w), dtype=np.float64)

        for j in range(len(w)):
            wi = w[j]
            ds = d_1[j]
            dt = d_2[j]

            if wi <= 0.0:
                continue

            if not (np.isfinite(ds) and np.isfinite(dt)):
                continue

            d_star = 0.5 * (ds + dt + wi)

            result[j] = (
                2.0 * _poly_kernel_antiderivative(
                    d_star-r, delta_r, n
                )
                - _poly_kernel_antiderivative(
                    ds-r, delta_r, n
                )
                - _poly_kernel_antiderivative(
                    dt-r, delta_r, n
                )
            )

        return result
    
    
    # generic case: finite redistribution scale
    else:
        w = np.ascontiguousarray(w, dtype=np.float64)
        d_1 = np.ascontiguousarray(d_1, dtype=np.float64)
        d_2 = np.ascontiguousarray(d_2, dtype=np.float64)

        if w.shape != d_1.shape or w.shape != d_2.shape:
            raise ValueError("Input shapes must match")

        nodes, weights = _cached_leggauss(quadrature_order)

        return _integrated_smooth_fast(
            r, w, d_1, d_2,
            delta_r, n, redistribution_scale,
            nodes, weights
        )

