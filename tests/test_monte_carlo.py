import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from examples.montecarlo import (monte_carlo_cpu,monte_carlo_cuda,)

def test_monte_carlo_cuda():
    n_points = 100000

    pi_cpu = monte_carlo_cpu(n_points)
    pi_gpu = monte_carlo_cuda(n_points)

    assert abs(pi_cpu - np.pi) < 0.05
    assert abs(pi_gpu - np.pi) < 0.05