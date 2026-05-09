"""
Monte Carlo estimate of pi as a test using CUDA
"""

## IMPORTS ##
import math
import numpy as np
from numba import cuda

def monte_carlo_cpu(n_points):
    """Estimate pi on the CPU."""
    x = np.random.random(n_points)
    y = np.random.random(n_points)

    inside = (x**2 + y**2) <= 1.0

    return 4 * np.sum(inside)/n_points

@cuda.jit
def monte_carlo_kernel(x, y, inside):
    i = cuda.grid(1)

    if i < x.size:
        inside[i] = x[i]**2 + y[i]**2 <= 1.0

def monte_carlo_cuda(n_points, threads_per_block=256):
    """Estimate pi using CUDA."""
    x = np.random.random(n_points)
    y = np.random.random(n_points)

    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)

    inside_gpu = cuda.device_array(n_points, dtype=np.bool_)

    blocks_per_grid = math.ceil(n_points/threads_per_block)

    monte_carlo_kernel[blocks_per_grid, threads_per_block](x_gpu,y_gpu,inside_gpu,)

    cuda.synchronize()

    inside = inside_gpu.copy_to_host()

    return 4 * np.sum(inside)/n_points