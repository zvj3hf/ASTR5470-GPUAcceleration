"""
CPU vs CUDA performance benchmark
"""

## IMPORTS ##
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pandas as pd
import numpy as np
from src.timing import timer

from examples.montecarlo import (monte_carlo_cpu, monte_carlo_cuda,)
from examples.radiative_transfer import (make_rays, radiative_transfer_cpu,radiative_transfer_cuda,)
from examples.airy import (make_grid, airy_coeffs,airy_cpu,airy_cuda,)

results = []

# Monte Carlo:

n_points = 1000000

with timer() as t:
    monte_carlo_cpu(n_points)

cpu_time = t["elapsed"]

with timer() as t:
    monte_carlo_cuda(n_points)

gpu_time = t["elapsed"]

results.append({"example": "monte_carlo", "cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": cpu_time/gpu_time,})


# Radiative transfer:

I0, tau = make_rays(1000000)

with timer() as t:
    radiative_transfer_cuda(I0,tau)

gpu_time = t["elapsed"]

results.append({"example": "radiative_transfer", "cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": cpu_time/gpu_time,})

# Airy:

_,_,X,Y,Z = make_grid(400)

coeffs = airy_coeffs(150)

with timer() as t:
    np.abs(airy_cpu(Z,coeffs))

cpu_time = t["elapsed"]

with timer() as t:
    airy_cuda(X,Y,coeffs)

gpu_time = t["elapsed"]

results.append({"example": "airy", "cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": cpu_time/gpu_time,})

# Save everything

df = pd.DataFrame(results)

print(df)

df.to_csv("outputs/benchmark_results.csv",index=False)

print("Saved results to outputs/benchmark_results.csv")