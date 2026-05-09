"""
Plot benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/benchmark_results.csv")

plt.figure(figsize=(6, 4))

plt.bar(df["example"], df["speedup"])

plt.ylabel("CPU / GPU Speedup")
plt.title("GPU Acceleration Performance")

plt.savefig("figures/benchmark_speedup.png", dpi=200)

plt.show()