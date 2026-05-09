"""
First test for the algorithm: Airy Functions. Does the CPU match the GPU?
"""

## IMPORTS ##
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pytest
from numba import cuda

from examples.airy import (make_grid,airy_coeffs,airy_cpu,airy_cuda,)

def test_airy():
    N=64
    _,_,X,Y,Z = make_grid(N,-3,-3,-3,3)
    coeffs = airy_coeffs(80)

    cpu_mag = np.abs(airy_cpu(Z,coeffs))
    gpu_mag = airy_cuda(X,Y,coeffs)

    assert np.allclose(cpu_mag,gpu_mag,rtol=1e-12,atol=1e-14)