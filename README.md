# GPU Acceleration Model for ASTR 5470 Final Project

This is an easy-to-use GPU acceleration algorithm using CUDA, Numba, and CuPy, and the goal is to provide an easily usable framework for any code. Please make sure you have a GPU available to use, like NVIDIA or something that can run CUDA.

Current tests/examples implemented:
- Airy Function Calculation
- Monte Carlo pi Estimation
- Simple Radiative Transfer Calculation

# Features

- CPU and CUDA implementations
- Configurable YAML input files
- GPU validation tests (checking that the GPU and CPU results match)
- Reusable and customizable CUDA launch configuration

# Installation

conda create -n gpu_algorithm python=3.11
conda activate gpu_algorithm

Install dependencies:

```bash
pip install -r requirements.txt
```

# Project Structure

```text
config/     YAML configuration files
examples/   Implementation Examples
scripts/    Scripts for running and benchmarking
src/        Utility modules
tests/      Validation tests
figures/    Generated figures
outputs/    Outputs and benchmark data
```

# Running Examples

python -m scripts.run_example --config config/default.yaml

Change the example in config/default.yaml for whichever example you wish to run.

# Testing
pytest tests/ -v

# Benchmarks

Run performance check with:

```bash
python scripts/benchmark.py

# Outputs

This code generates:
- benchmark CSV files
- figures
- timing information
- NumPy output arrays

# AI Usage Statement

On my honor, I used AI in the following ways:

- To debug all files and ensure that everything would run smoothly.
- To figure out what checks need to be made to make sure the user interface is easy to use
- To understand what kind of files I need for the github.

