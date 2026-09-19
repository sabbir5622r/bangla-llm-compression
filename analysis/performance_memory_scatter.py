from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "processed"
    / "final_analysis"
)

FIGURE_DIR = ROOT / "figures" / "final"

pareto = pd.read_csv(
    ANALYSIS_DIR / "pareto_analysis.csv"
)

model_labels = {
    "qwen2.5-0.5b-instruct": "0.5B",
    "qwen2.5-1.5b-instruct": "1.5B",
    "qwen2.5-3b-instruct": "3B",
}

quant_labels = {
    "fp16": "FP16",
    "int8": "INT8",
    "int4": "4-bit",
}

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

for _, row in pareto.iterrows():

    label = (
        f"{model_labels[row['model']]} "
        f"{quant_labels[row['quantization']]}"
    )

    ax.scatter(
        row["model_footprint_gb"],
        row["mean_macro_f1"],
        s=85
    )

    ax.annotate(
        label,
        (
            row["model_footprint_gb"],
            row["mean_macro_f1"]
        ),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel(
    "Model Footprint (GB)"
)

ax.set_ylabel(
    "Mean Macro-F1"
)

ax.set_title(
    "Performance–Memory Trade-off",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    linestyle="--",
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

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