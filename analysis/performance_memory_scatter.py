from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "processed"
    / "final_analysis"
)

FIGURE_DIR = ROOT / "figures" / "final"
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

pareto = pd.read_csv(
    ANALYSIS_DIR / "pareto_analysis.csv"
)


model_labels = {
    "qwen2.5-0.5b-instruct": "Qwen2.5 0.5B",
    "qwen2.5-1.5b-instruct": "Qwen2.5 1.5B",
    "qwen2.5-3b-instruct": "Qwen2.5 3B",
    "falcon3-3b-instruct": "Falcon3 3B",
}

quant_labels = {
    "fp16": "FP16",
    "int8": "INT8",
    "int4": "4-bit NF4",
}


# Different marker shapes identify model/model scale
model_markers = {
    "qwen2.5-0.5b-instruct": "o",
    "qwen2.5-1.5b-instruct": "s",
    "qwen2.5-3b-instruct": "^",
    "falcon3-3b-instruct": "D",
}


# Use Matplotlib's default color cycle for precision
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

quant_colors = {
    "fp16": default_colors[0],
    "int8": default_colors[1],
    "int4": default_colors[2],
}


fig, ax = plt.subplots(
    figsize=(9, 6)
)


# --------------------------------------------------
# Scatter points
# --------------------------------------------------

for _, row in pareto.iterrows():

    ax.scatter(
        row["model_footprint_gb"],
        row["mean_macro_f1"],
        s=130,
        marker=model_markers[row["model"]],
        color=quant_colors[row["quantization"]],
        edgecolors="white",
        linewidths=0.8,
        zorder=3
    )


# --------------------------------------------------
# Axes
# --------------------------------------------------

ax.set_xlabel(
    "Model Footprint (GB)",
    fontsize=13
)

ax.set_ylabel(
    "Mean Macro-F1",
    fontsize=13
)

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.grid(
    linestyle="--",
    alpha=0.25,
    zorder=0
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# --------------------------------------------------
# Model legend: marker shape
# --------------------------------------------------

model_handles = [
    Line2D(
        [0],
        [0],
        marker=model_markers[model],
        linestyle="None",
        markerfacecolor="gray",
        markeredgecolor="gray",
        markersize=8,
        label=label
    )
    for model, label in model_labels.items()
]


# --------------------------------------------------
# Precision legend: color
# --------------------------------------------------

quant_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=quant_colors[quant],
        markeredgecolor=quant_colors[quant],
        markersize=8,
        label=label
    )
    for quant, label in quant_labels.items()
]


# First legend: models
legend_models = ax.legend(
    handles=model_handles,
    title="Model",
    loc="upper center",
    bbox_to_anchor=(0.33, 1.18),
    ncol=2,
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    columnspacing=1.5,
    handletextpad=0.5
)

ax.add_artist(legend_models)


# Second legend: quantization
ax.legend(
    handles=quant_handles,
    title="Precision",
    loc="upper center",
    bbox_to_anchor=(0.76, 1.18),
    ncol=3,
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    columnspacing=1.3,
    handletextpad=0.5
)


# Leave room for legends
plt.tight_layout(
    rect=[0, 0, 1, 0.87]
)


# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    FIGURE_DIR / "performance_memory_scatter.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR / "performance_memory_scatter.pdf",
    bbox_inches="tight"
)

plt.show()