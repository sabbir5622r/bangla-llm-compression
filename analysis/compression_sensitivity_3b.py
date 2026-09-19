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

performance = pd.read_csv(
    ANALYSIS_DIR / "compression_summary.csv"
)

model = "qwen2.5-3b-instruct"

task_labels = {
    "stance": "Stance",
    "nli": "NLI",
    "fake_news": "Fake News",
}

quantizations = [
    "FP16",
    "INT8",
    "4-bit NF4"
]

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

for task, task_label in task_labels.items():

    row = performance[
        (performance["model"] == model)
        &
        (performance["task"] == task)
    ].iloc[0]

    values = [
        row["fp16"],
        row["int8"],
        row["int4"],
    ]

    ax.plot(
        quantizations,
        values,
        marker="o",
        linewidth=2.3,
        markersize=7,
        label=task_label
    )

    for x, value in zip(
        quantizations,
        values
    ):

        ax.text(
            x,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

ax.set_ylabel(
    "Macro-F1"
)

ax.set_xlabel(
    "Quantization Level"
)

ax.set_title(
    "Qwen2.5-3B Compression Sensitivity",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    linestyle="--",
    alpha=0.25
)

ax.legend(
    frameon=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "3b_compression_sensitivity.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR / "3b_compression_sensitivity.pdf",
    bbox_inches="tight"
)

plt.show()