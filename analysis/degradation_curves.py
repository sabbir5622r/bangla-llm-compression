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
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

performance = pd.read_csv(
    ANALYSIS_DIR / "compression_summary.csv"
)

models = {
    "qwen2.5-0.5b-instruct": "0.5B",
    "qwen2.5-1.5b-instruct": "1.5B",
    "qwen2.5-3b-instruct": "3B",
}

tasks = {
    "stance": "Stance",
    "nli": "NLI",
    "fake_news": "Fake News",
}

quantizations = ["FP16", "INT8", "4-bit NF4"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.5, 4.2),
    sharey=True
)

for ax, (task, task_label) in zip(
    axes,
    tasks.items()
):

    task_df = performance[
        performance["task"] == task
    ].set_index("model")

    for model, model_label in models.items():

        values = [
            task_df.loc[model, "fp16"],
            task_df.loc[model, "int8"],
            task_df.loc[model, "int4"],
        ]

        ax.plot(
            quantizations,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=model_label
        )

        for x, value in zip(
            quantizations,
            values
        ):
            ax.text(
                x,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                fontsize=8
            )

    ax.set_title(
        task_label,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Quantization Level"
    )

    ax.grid(
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(
    "Macro-F1"
)

axes[0].set_ylim(
    0,
    0.85
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03)
)

fig.suptitle(
    "Qwen2.5 Performance Degradation Under Quantization",
    fontsize=15,
    fontweight="bold",
    y=1.10
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "compression_degradation_curves.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR / "compression_degradation_curves.pdf",
    bbox_inches="tight"
)

plt.show()