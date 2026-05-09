"""
Example code for GPU acceleration with Airy functions. Comparing CPU and GPU run times on a 2D complex grid.
"""

## IMPORTS ##

import math
import numpy as np
from scipy.special import airy, gamma
from numba import cuda


def make_grid(N, x_min=-10, x_max=10, y_min=-10, y_max=10):
    """Create a 2D complex grid z = x + iy."""
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    return x, y, X, Y, Z


def airy_coeffs(n_terms=150):
    """Compute Taylor-series coefficients for Ai(z)."""
    coeffs = np.zeros(n_terms, dtype=np.complex128)

    coeffs[0] = 1.0 / (3 ** (2 / 3) * gamma(2 / 3))
    coeffs[1] = -1.0 / (3 ** (1 / 3) * gamma(1 / 3))

    for k in range(2, n_terms):
        coeffs[k] = coeffs[k - 3] / (k * (k - 1))

    return coeffs


def airy_cpu(Z, coeffs):
    """Evaluate Ai(z) on the CPU."""
    Ai = np.zeros_like(Z, dtype=np.complex128)

    for index, z in np.ndenumerate(Z):
        value = coeffs[-1]

        for c in coeffs[-2::-1]:
            value = value * z + c

        Ai[index] = value

    return Ai


@cuda.jit
def airy_cuda_kernel(Z_real, Z_imag, coeffs, magnitude, n_terms):
    """CUDA kernel: each thread computes Ai(z) for one grid point."""
    i, j = cuda.grid(2)

    if i < magnitude.shape[0] and j < magnitude.shape[1]:
        zr = Z_real[i, j]
        zi = Z_imag[i, j]

        real_part = coeffs[n_terms - 1].real
        imag_part = coeffs[n_terms - 1].imag

        for k in range(n_terms - 2, -1, -1):
            c_real = coeffs[k].real
            c_imag = coeffs[k].imag

            new_real = real_part * zr - imag_part * zi + c_real
            new_imag = real_part * zi + imag_part * zr + c_imag

            real_part = new_real
            imag_part = new_imag

        magnitude[i, j] = math.sqrt(real_part**2 + imag_part**2)


def airy_cuda(X, Y, coeffs, threads_per_block=(16, 16)):
    """Evaluate Ai(z) on the GPU."""
    n_terms = len(coeffs)

    Z_real_gpu = cuda.to_device(X)
    Z_imag_gpu = cuda.to_device(Y)
    coeffs_gpu = cuda.to_device(coeffs)

    magnitude_gpu = cuda.device_array(X.shape, dtype=np.float64)

    blocks_per_grid = (math.ceil(X.shape[0] / threads_per_block[0]),math.ceil(X.shape[1] / threads_per_block[1]),)

    airy_cuda_kernel[blocks_per_grid, threads_per_block](Z_real_gpu,Z_imag_gpu,coeffs_gpu,magnitude_gpu,n_terms,)

    cuda.synchronize()

    return magnitude_gpu.copy_to_host()


def asymptotic_coeffs(n_terms):
    """Compute coefficients used in the Airy asymptotic approximations."""
    coeffs = np.ones(n_terms, dtype=complex)

    for k in range(1, n_terms):
        coeffs[k] = ((6 * k - 5) * (6 * k - 1)) / (48 * k) * coeffs[k - 1]

    return coeffs


def airy_asymp_pos(z, coeffs):
    """Asymptotic approximation for Ai(z) on the positive-real side."""
    z = np.asarray(z, dtype=complex)

    zeta = (2 / 3) * z**1.5

    series = sum(((-1) ** k) * coeffs[k] / zeta**k
        for k in range(len(coeffs)))

    return np.exp(-zeta) / (2 * np.sqrt(np.pi) * z**0.25) * series


def airy_asymp_neg(z, coeffs):
    """Asymptotic approximation for Ai(z) on the negative-real side."""
    w = -np.asarray(z, dtype=complex)

    zeta = (2 / 3) * w**1.5
    phi = zeta - np.pi / 4

    n_terms = len(coeffs)

    even = sum(((-1) ** k) * coeffs[2 * k] / zeta ** (2 * k)
        for k in range(n_terms // 2))

    odd = sum(((-1) ** k) * coeffs[2 * k + 1] / zeta ** (2 * k + 1)
        for k in range((n_terms - 1) // 2))

    return (1 / np.sqrt(np.pi)) * w ** (-0.25) * (
        np.cos(phi) * even - np.sin(phi) * odd)


def airy_spliced(Z, taylor_coeffs, cut=8.0, n_pos=50, n_neg=40):
    """
    Compute a spliced approximation to Ai(z).

    Uses the Taylor series for |z| <= cut and asymptotic approximations
    for |z| > cut.
    """
    Ai_mid = airy_cpu(Z, taylor_coeffs)

    pos_coeffs = asymptotic_coeffs(n_pos)
    neg_coeffs = asymptotic_coeffs(n_neg)

    Ai_tail = np.where(Z.real >= 0,airy_asymp_pos(Z, pos_coeffs),airy_asymp_neg(Z, neg_coeffs),)

    return np.where(np.abs(Z) <= cut, Ai_mid, Ai_tail)


def airy_scipy(x):
    """Return SciPy's Ai(x) value on the real axis."""
    return airy(x)[0]


def find_roots(values, x):
    """Estimate roots from sign changes using linear interpolation."""
    sign_changes = np.where(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]

    roots = []

    for i in sign_changes:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = values[i], values[i + 1]

        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(root)

    return roots