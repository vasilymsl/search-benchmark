#!/usr/bin/env python3
"""
Generate publication-quality figures for the IEEE report from benchmark-results.json.
Output: docs/figures/{fig1_quality.png, fig2_latency.png, fig3_quality_vs_latency.png}

Usage: python3 scripts/generate-report-figures.py
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "benchmark-results.json"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-style figure conventions
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLOR = {"BM25": "#e69f00", "Semantic": "#9061c2", "Hybrid": "#2ca25f"}

with open(RESULTS_FILE) as f:
    data = json.load(f)
summary = {s["method"]: s for s in data["summary"]}

# ---------- Figure 1: Retrieval Quality ----------
fig, ax = plt.subplots(figsize=(5.5, 3.3))
metrics = ["mrrAt10", "precisionAt5", "recallAt5", "ndcgAt10"]
metric_labels = ["MRR@10", "P@5", "R@5", "NDCG@10"]
methods = ["BM25", "Semantic", "Hybrid"]

x = np.arange(len(metrics))
width = 0.26
for i, m in enumerate(methods):
    vals = [summary[m][k]["mean"] for k in metrics]
    offsets = (i - 1) * width
    bars = ax.bar(x + offsets, vals, width, label=m, color=COLOR[m], edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Retrieval quality comparison on SciFact (300 queries)")
ax.legend(loc="upper right", ncol=3, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_quality.png")
plt.close()
print(f"✓ {OUT_DIR / 'fig1_quality.png'}")

# ---------- Figure 2: Per-Method Latency ----------
fig, ax = plt.subplots(figsize=(5.5, 3.0))
means = [summary[m]["latencyMs"]["mean"] for m in methods]
stds = [summary[m]["latencyMs"]["std"] for m in methods]
colors = [COLOR[m] for m in methods]
bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors,
              edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.8})
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.12,
            f"{mean:.2f} ± {std:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Latency per query (ms)")
ax.set_title("Per-method query latency (Node.js, Apple Silicon)")
ax.set_ylim(0, max(means) + max(stds) + 1.0)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_latency.png")
plt.close()
print(f"✓ {OUT_DIR / 'fig2_latency.png'}")

# ---------- Figure 3: Quality vs Latency Scatter ----------
fig, ax = plt.subplots(figsize=(5.5, 3.3))
for m in methods:
    x_val = summary[m]["latencyMs"]["mean"]
    y_val = summary[m]["mrrAt10"]["mean"]
    ax.scatter(x_val, y_val, s=180, color=COLOR[m], edgecolor="black",
               linewidth=0.8, label=m, zorder=3)
    ax.annotate(m, (x_val, y_val), xytext=(7, 7), textcoords="offset points",
                fontsize=9, fontweight="bold")
ax.set_xlabel("Latency per query (ms)")
ax.set_ylabel("MRR@10")
ax.set_xlim(-0.3, max(summary[m]["latencyMs"]["mean"] for m in methods) + 1.0)
ax.set_ylim(0.5, 0.7)
ax.set_title("Quality vs. latency trade-off")
ax.annotate("", xy=(0.1, 0.68), xytext=(4.3, 0.56),
            arrowprops=dict(arrowstyle="-", color="gray", linestyle=":", lw=0.8))
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_quality_vs_latency.png")
plt.close()
print(f"✓ {OUT_DIR / 'fig3_quality_vs_latency.png'}")

print("\nAll figures saved to", OUT_DIR)
