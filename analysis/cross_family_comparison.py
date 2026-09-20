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

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

performance = pd.read_csv(
    ANALYSIS_DIR
    / "cross_family_comparison.csv"
)

models = {
    "Qwen2.5 3B":
        "qwen2.5-3b-instruct",

    "Falcon3 3B":
        "falcon3-3b-instruct",
}

task_labels = {
    "stance": "Stance",
    "nli": "NLI",
    "fake_news": "Fake News",
}

quantizations = [
    "FP16",
    "INT8",
    "4-bit NF4",
]


for task, task_label in (
    task_labels.items()
):

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for model_label, model_name in (
        models.items()
    ):

        row = performance[
            (performance["model"] == model_name)
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
            label=model_label,
        )

        for x, value in zip(
            quantizations,
            values,
        ):
            ax.text(
                x,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=8.5,
            )

    ax.set_ylabel(
        "Macro-F1"
    )

    ax.set_xlabel(
        "Quantization Level"
    )

    ax.set_title(
        f"Cross-Family Compression Comparison: "
        f"{task_label}",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    plt.tight_layout()

    png_path = (
        FIGURE_DIR
        / f"cross_family_{task}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"cross_family_{task}.pdf"
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.show()

    print(
        "Saved:"
    )

    print(
        png_path
    )

    print(
        pdf_path
    )