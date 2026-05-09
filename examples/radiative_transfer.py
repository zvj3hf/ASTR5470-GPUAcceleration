"""
Radiative transfer example using CUDA

"""

## IMPORTS ##
import math
import numpy as np
from numba import cuda

def radiative_transfer_cpu(I0,tau):
    """Compute outgoing intensity on the CPU."""
    return I0 * np.exp(-tau)


@cuda.jit

def radiative_transfer_kernel(I0,tau,I_out):
    """CUDA Kernel: each thread computes one ray"""
    i = cuda.grid(1)

    if i < I0.size:
        I_out[i] = I0[i] * math.exp(-tau[i])

def radiative_transfer_cuda(I0,tau,threads_per_block=256):
    """Compute outgoing intensity on the GPU"""
    I0_gpu = cuda.to_device(I0)
    tau_gpu = cuda.to_device(tau)

    I_out_gpu = cuda.device_array(I0.shape,dtype=np.float64)

    blocks_per_grid = math.ceil(I0.size/threads_per_block)

    radiative_transfer_kernel[blocks_per_grid,threads_per_block](I0_gpu,tau_gpu,I_out_gpu,)

    cuda.synchronize()

    return I_out_gpu.copy_to_host()

def make_rays(n_rays,tau_max=5.0):
    """Sample ray intensities and optical depths."""
    I0 = np.ones(n_rays,dtype=np.float64)
    tau = np.linspace(0, tau_max, n_rays, dtype=np.float64)

    return I0, tau
