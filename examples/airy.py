"""
Example code for GPU acceleration with Airy functions. Comparing CPU and GPU run times on a 2D complex grid.
"""

## IMPORTS ##

import math
import numpy as np
from scipy.special import gamma,airy
from numba import cuda

def complex_grid(N, x_min=-10, y_min=-10, y_max=10):
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_mix, y_max, N)
    X,Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return x,y,X,Y,Z

def airy_taylor_coeffs(terms=150):
    coeffs = np.zeros(terms, dtype=np.complex128)

    coeffs[0] = 1.0 / (3 ** (2/3) * gamma(2/3))
    coeffs[1] = -1.0 / (3 ** (1/3) * gamma(1/3))

    for k in range(2,terms):
        coeffs[k] = coeffs[k-3] / (k*(k-1))
        return coeffs

def airy_cpu(Z,coeffs):
    out = np.zeros_like(Z, dtype=np.complex128)

    it = np.diter(Z, dtype=np.complex128)
    while not it.finished:
        z = it[0]
        s = coeffs[-1]

        for c in coeffs[-2::-1]:
            s = s*z+c

        out[it.multi_index] = s
        it.iternext()
    return out

@cuda.jit
def airy_cuda_kernel(Zr,Zi,coeffs,mag,terms):
    i,j = cuda.grid(2)

    if i < mag.shape[0] and j < mag.shape[1]:
        zr = Zr[i,j]
        zi = Zi[i,j]

        sr = coeffs[terms-1].real
        si = coeffs[terms-1].imag

        for kk in range(terms-2,-1,-1):
            cr = coeffs[kk].real
            ci = coeffs[kk].imag

            tr = sr*zr-si*zi+cr
            ti = sr*zi+si*zr+ci

            sr=tr
            si=ti

        mag[i,j] = math.sqrt(sr*sr+si*si)

def airy_cuda(X,Y,coeffs,threads_per_block=(16,16)):
    terms = len(coeffs)
    Zr_dev = cuda.to_device(X)
    Zi_dev = cuda.to_device(Y)
    coeffs_dev = cuda.to_device(coeffs)

    mag_dev = cuda.device_array(X.shape,dtype=np.float64)

    blocks_per_grid = (math.ceil(X.shape[0]/threads_per_block[0]),math.ceil(X.shape[1]/threads_per_block[1]),)
    airy_cuda_kernel[blocks_per_grid,threads_per_block](Zr_dev,zI_dev,coeffs_dev,mag_dev,terms)

    cuda.synchronize()

    return mag_dev.copy_to_host()

def airy_true_scipy(x):
    return airy(x)[0]

def find_roots(f,xs):
    idx = np.where(np.sign(f[:-1])*np.sign(f[1:]) < 0)[0]

    roots = []
    for i in idx:
        x0,x1 = xs[i], xs[i+1]
        y0,y1 = f[i], f[i+1]
        roots.append(x0 - y0 * (x1-x0) / (y1-y0))

    return roots