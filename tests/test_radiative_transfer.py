import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np

from examples.radiative_transfer import (make_rays,radiative_transfer_cpu,radiative_transfer_cuda,)

def test_radiative_transfer():
    I0, tau = make_rays(n_rays=10000, tau_max=5.0)

    cpu_result = radiative_transfer_cpu(I0, tau)
    gpu_result = radiative_transfer_cuda(I0, tau)

    assert np.allclose(cpu_result, gpu_result, rtol=1e-10, atol=1e-12)