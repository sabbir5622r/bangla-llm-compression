from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "processed"
    / "final_analysis"
)

FIGURE_DIR = ROOT / "figures" / "final"

performance = pd.read_csv(
    ANALYSIS_DIR / "compression_summary.csv"
)

model_order = [
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
]

model_labels = {
    "qwen2.5-0.5b-instruct": "0.5B",
    "qwen2.5-1.5b-instruct": "1.5B",
    "qwen2.5-3b-instruct": "3B",
}

task_order = [
    "stance",
    "nli",
    "fake_news",
]

task_labels = {
    "stance": "Stance",
    "nli": "NLI",
    "fake_news": "Fake News",
}

rows = []
labels = []

for model in model_order:

    for task in task_order:

        row = performance[
            (performance["model"] == model)
            &
            (performance["task"] == task)
        ].iloc[0]

        rows.append([
            row["int8_retention_pct"],
            row["int4_retention_pct"],
        ])

        labels.append(
            f"{model_labels[model]} — {task_labels[task]}"
        )

matrix = np.array(rows)

fig, ax = plt.subplots(
    figsize=(7.5, 7)
)

image = ax.imshow(
    matrix,
    aspect="auto"
)

ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "INT8",
        "4-bit NF4"
    ]
)

ax.set_yticks(
    np.arange(len(labels))
)

ax.set_yticklabels(
    labels
)

for i in range(matrix.shape[0]):

    for j in range(matrix.shape[1]):

        ax.text(
            j,
            i,
            f"{matrix[i, j]:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

ax.set_title(
    "FP16 Performance Retained After Quantization",
    fontsize=14,
    fontweight="bold",
    pad=12
)

cbar = fig.colorbar(
    image,
    ax=ax
)

cbar.set_label(
    "Performance Retention (%)"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "retention_heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR / "retention_heatmap.pdf",
    bbox_inches="tight"
)

plt.show()