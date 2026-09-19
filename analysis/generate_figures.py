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

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

performance = pd.read_csv(
    ANALYSIS_DIR / "compression_summary.csv"
)

efficiency = pd.read_csv(
    ANALYSIS_DIR / "efficiency_summary.csv"
)

pareto = pd.read_csv(
    ANALYSIS_DIR / "pareto_analysis.csv"
)

MODEL_ORDER = [
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
]

MODEL_LABELS = {
    "qwen2.5-0.5b-instruct": "0.5B",
    "qwen2.5-1.5b-instruct": "1.5B",
    "qwen2.5-3b-instruct": "3B",
}

TASK_ORDER = [
    "stance",
    "nli",
    "fake_news",
]

TASK_LABELS = {
    "stance": "Stance",
    "nli": "NLI",
    "fake_news": "Fake News",
}

QUANTIZATIONS = [
    ("fp16", "FP16"),
    ("int8", "INT8"),
    ("int4", "4-bit NF4"),
]

x = np.arange(len(MODEL_ORDER))
bar_width = 0.23

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.5),
    sharey=True
)

for ax, task in zip(
    axes,
    TASK_ORDER
):

    task_df = performance[
        performance["task"] == task
    ].set_index("model")

    groups = [
        (
            "FP16",
            [
                task_df.loc[m, "fp16"]
                for m in MODEL_ORDER
            ]
        ),
        (
            "INT8",
            [
                task_df.loc[m, "int8"]
                for m in MODEL_ORDER
            ]
        ),
        (
            "4-bit NF4",
            [
                task_df.loc[m, "int4"]
                for m in MODEL_ORDER
            ]
        ),
    ]

    for i, (label, values) in enumerate(groups):

        positions = (
            x
            + (i - 1) * bar_width
        )

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=label,
            edgecolor="black",
            linewidth=0.6,
        )

        for bar, value in zip(
            bars,
            values
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    ax.set_title(
        TASK_LABELS[task],
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            MODEL_LABELS[m]
            for m in MODEL_ORDER
        ]
    )

    ax.set_xlabel(
        "Model Size"
    )

    ax.set_ylim(
        0,
        0.85
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(
    "Macro-F1"
)

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle(
    "Performance Across Quantization Levels",
    fontsize=15,
    fontweight="bold",
    y=1.10
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "macro_f1_quantization_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "macro_f1_quantization_comparison.pdf",
    bbox_inches="tight"
)

plt.show()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.3),
    sharey=True
)

x = np.arange(
    len(MODEL_ORDER)
)

width = 0.32

for ax, task in zip(
    axes,
    TASK_ORDER
):

    task_df = performance[
        performance["task"] == task
    ].set_index("model")

    int8_retention = [
        task_df.loc[
            m,
            "int8_retention_pct"
        ]
        for m in MODEL_ORDER
    ]

    int4_retention = [
        task_df.loc[
            m,
            "int4_retention_pct"
        ]
        for m in MODEL_ORDER
    ]

    bars1 = ax.bar(
        x - width / 2,
        int8_retention,
        width,
        label="INT8",
        edgecolor="black",
        linewidth=0.6,
    )

    bars2 = ax.bar(
        x + width / 2,
        int4_retention,
        width,
        label="4-bit NF4",
        edgecolor="black",
        linewidth=0.6,
    )

    for bars in [
        bars1,
        bars2
    ]:
        for bar in bars:

            value = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(
        100,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_title(
        TASK_LABELS[task],
        fontweight="bold"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            MODEL_LABELS[m]
            for m in MODEL_ORDER
        ]
    )

    ax.set_xlabel(
        "Model Size"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(
    "FP16 Performance Retained (%)"
)

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle(
    "Performance Retention After Quantization",
    fontsize=15,
    fontweight="bold",
    y=1.10
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "performance_retention.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "performance_retention.pdf",
    bbox_inches="tight"
)

plt.show()

fig, ax = plt.subplots(
    figsize=(8.5, 5)
)

x = np.arange(
    len(MODEL_ORDER)
)

width = 0.23

for i, (
    quant,
    label
) in enumerate(QUANTIZATIONS):

    values = []

    for model in MODEL_ORDER:

        row = efficiency[
            (
                efficiency["model"]
                == model
            )
            &
            (
                efficiency["quantization"]
                == quant
            )
        ]

        values.append(
            float(
                row.iloc[0][
                    "model_footprint_gb"
                ]
            )
        )

    positions = (
        x
        + (i - 1)
        * width
    )

    bars = ax.bar(
        positions,
        values,
        width=width,
        label=label,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.06,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xticks(x)

ax.set_xticklabels(
    [
        MODEL_LABELS[m]
        for m in MODEL_ORDER
    ]
)

ax.set_xlabel(
    "Model Size"
)

ax.set_ylabel(
    "Model Footprint (GB)"
)

ax.set_title(
    "Model Footprint Across Quantization Levels",
    fontsize=14,
    fontweight="bold",
)

ax.grid(
    axis="y",
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
    FIGURE_DIR
    / "model_footprint.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "model_footprint.pdf",
    bbox_inches="tight"
)

plt.show()

fig, ax = plt.subplots(
    figsize=(8.5, 5)
)

x = np.arange(
    len(MODEL_ORDER)
)

width = 0.23

for i, (
    quant,
    label
) in enumerate(QUANTIZATIONS):

    values = []

    for model in MODEL_ORDER:

        row = efficiency[
            (
                efficiency["model"]
                == model
            )
            &
            (
                efficiency["quantization"]
                == quant
            )
        ]

        values.append(
            float(
                row.iloc[0][
                    "generated_tokens_per_second"
                ]
            )
        )

    positions = (
        x
        + (i - 1)
        * width
    )

    bars = ax.bar(
        positions,
        values,
        width=width,
        label=label,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xticks(x)

ax.set_xticklabels(
    [
        MODEL_LABELS[m]
        for m in MODEL_ORDER
    ]
)

ax.set_xlabel(
    "Model Size"
)

ax.set_ylabel(
    "Generated Tokens / Second"
)

ax.set_title(
    "Inference Throughput Across Quantization Levels",
    fontsize=14,
    fontweight="bold",
)

ax.grid(
    axis="y",
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
    FIGURE_DIR
    / "throughput_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "throughput_comparison.pdf",
    bbox_inches="tight"
)

plt.show()

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

for _, row in pareto.iterrows():

    label = (
        f"{MODEL_LABELS[row['model']]} "
        f"{row['quantization'].upper()}"
    )

    if row["quantization"] == "int4":
        label = (
            f"{MODEL_LABELS[row['model']]} "
            "4-bit"
        )

    ax.scatter(
        row["model_footprint_gb"],
        row["mean_macro_f1"],
        s=70,
    )

    ax.annotate(
        label,
        (
            row["model_footprint_gb"],
            row["mean_macro_f1"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Model Footprint (GB)"
)

ax.set_ylabel(
    "Mean Macro-F1 Across Tasks"
)

ax.set_title(
    "Performance–Memory Trade-off",
    fontsize=14,
    fontweight="bold",
)

ax.grid(
    linestyle="--",
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "performance_memory_tradeoff.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    FIGURE_DIR
    / "performance_memory_tradeoff.pdf",
    bbox_inches="tight"
)

plt.show()

print("\nFigures saved to:")
print(FIGURE_DIR)

for path in sorted(
    FIGURE_DIR.glob("*")
):
    print(path.name)