"""
Script for running example code.
"""

## IMPORTS ##

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from examples.airy import (make_grid,airy_coeffs,airy_cpu,airy_cuda,airy_spliced,)
from src.io_utils import load_configuration, check_dir, save_array, save_text
from src.timing import timer

def run_airy(cfg):
    """Run airy example using CPU or CUDA"""

    N = cfg["grid"]["N"]
    output_dir = cfg["output"]["output_dir"]
    backend = cfg["backend"]
    check_dir(output_dir)
    check_dir("figures")

    x,y,X,Y,Z = make_grid(N,cfg["grid"]["x_min"],cfg["grid"]["x_max"],cfg["grid"]["y_min"],cfg["grid"]["y_max"],)

    coeffs = airy_coeffs(cfg["airy"]["taylor"])

    print(f"Running Airy example on a {N}x{N} grid")

    print(f"On: {backend}")

    if backend == "cpu":
        with timer() as t:
            Ai = airy_spliced(Z,coeffs,cut=cfg["airy"]["cut"])
            magnitude = np.abs(Ai)

    elif backend == "cuda":
        threads = (cfg["cuda"]["threads_per_block_x"], cfg["cuda"]["threads_per_block_y"],)

        with timer() as t:
            magnitude = airy_cuda(X,Y,coeffs,threads)

    else:
        raise ValueError("backend much be either 'cpu' or 'cuda'")

    print(f"Time elapsed {t['elapsed']:.4f} secs")

    save_array(os.path.join(output_dir,"airy_magnitude.npy"),magnitude)

    save_text(os.path.join(output_dir,"timing.txt"),f"backend: {backend}\n" f"N = {N}\n" f"time_seconds: {t['elapsed']:.4f}\n",)

    fig_path = os.path.join("figures", "airy_magnitude.png")

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X,Y,magnitude,norm=LogNorm(vmin=1e-5, vmax=1e1),shading="auto",)

    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Spliced |Ai(z)|")
    plt.colorbar(label="Magnitude")

    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved results to {output_dir}")
    print(f"Saved figure to {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_configuration(args.config)
    run_airy(cfg)


if __name__ == "__main__":
    main()