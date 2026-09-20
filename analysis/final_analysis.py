from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "results" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "final_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


QWEN_FP16_PATH = PROCESSED_DIR / "fp16_metrics.csv"
QWEN_INT8_PATH = PROCESSED_DIR / "int8_metrics.csv"
QWEN_INT4_PATH = PROCESSED_DIR / "int4_metrics.csv"

FALCON_FP16_PATH = PROCESSED_DIR / "falcon3_fp16_metrics.csv"
FALCON_INT8_PATH = PROCESSED_DIR / "falcon3_int8_metrics.csv"
FALCON_INT4_PATH = PROCESSED_DIR / "falcon3_int4_metrics.csv"

QWEN_EFFICIENCY_PATH = (
    PROCESSED_DIR / "efficiency_benchmark.csv"
)

FALCON_EFFICIENCY_PATH = (
    PROCESSED_DIR / "falcon3_efficiency_benchmark.csv"
)


qwen_fp16 = pd.read_csv(QWEN_FP16_PATH)
qwen_int8 = pd.read_csv(QWEN_INT8_PATH)
qwen_int4 = pd.read_csv(QWEN_INT4_PATH)

falcon_fp16 = pd.read_csv(FALCON_FP16_PATH)
falcon_int8 = pd.read_csv(FALCON_INT8_PATH)
falcon_int4 = pd.read_csv(FALCON_INT4_PATH)

qwen_efficiency = pd.read_csv(
    QWEN_EFFICIENCY_PATH
)

falcon_efficiency = pd.read_csv(
    FALCON_EFFICIENCY_PATH
)


qwen_fp16["quantization"] = "fp16"
qwen_int8["quantization"] = "int8"
qwen_int4["quantization"] = "int4"

falcon_fp16["quantization"] = "fp16"
falcon_int8["quantization"] = "int8"
falcon_int4["quantization"] = "int4"


metrics = pd.concat(
    [
        qwen_fp16,
        qwen_int8,
        qwen_int4,
        falcon_fp16,
        falcon_int8,
        falcon_int4,
    ],
    ignore_index=True
)


efficiency = pd.concat(
    [
        qwen_efficiency,
        falcon_efficiency,
    ],
    ignore_index=True
)


model_order = {
    "qwen2.5-0.5b-instruct": 0,
    "qwen2.5-1.5b-instruct": 1,
    "qwen2.5-3b-instruct": 2,
    "falcon3-3b-instruct": 3,
}

task_order = {
    "stance": 0,
    "nli": 1,
    "fake_news": 2,
}

quant_order = {
    "fp16": 0,
    "int8": 1,
    "int4": 2,
}


metrics["_model_order"] = (
    metrics["model"]
    .map(model_order)
)

metrics["_task_order"] = (
    metrics["task"]
    .map(task_order)
)

metrics["_quant_order"] = (
    metrics["quantization"]
    .map(quant_order)
)


metrics = (
    metrics
    .sort_values(
        [
            "_model_order",
            "_task_order",
            "_quant_order",
        ]
    )
    .drop(
        columns=[
            "_model_order",
            "_task_order",
            "_quant_order",
        ]
    )
)


metrics.to_csv(
    OUTPUT_DIR / "master_results.csv",
    index=False,
)


performance = metrics.pivot(
    index=["model", "task"],
    columns="quantization",
    values="macro_f1",
).reset_index()

performance.columns.name = None


performance["int8_f1_drop"] = (
    performance["fp16"]
    - performance["int8"]
)

performance["int4_f1_drop"] = (
    performance["fp16"]
    - performance["int4"]
)

performance["int8_retention"] = (
    performance["int8"]
    / performance["fp16"]
)

performance["int4_retention"] = (
    performance["int4"]
    / performance["fp16"]
)

performance["int8_retention_pct"] = (
    performance["int8_retention"]
    * 100
)

performance["int4_retention_pct"] = (
    performance["int4_retention"]
    * 100
)

performance["int8_relative_change_pct"] = (
    (
        performance["int8"]
        - performance["fp16"]
    )
    / performance["fp16"]
    * 100
)

performance["int4_relative_change_pct"] = (
    (
        performance["int4"]
        - performance["fp16"]
    )
    / performance["fp16"]
    * 100
)


performance.to_csv(
    OUTPUT_DIR / "compression_summary.csv",
    index=False,
)


fp16_efficiency = (
    efficiency[
        efficiency["quantization"]
        == "fp16"
    ]
    .set_index("model")
)


efficiency_rows = []

for _, row in efficiency.iterrows():

    baseline = fp16_efficiency.loc[
        row["model"]
    ]

    footprint_reduction_pct = (
        (
            baseline["model_footprint_gb"]
            - row["model_footprint_gb"]
        )
        / baseline["model_footprint_gb"]
        * 100
    )

    latency_change_pct = (
        (
            row["mean_latency_sec"]
            - baseline["mean_latency_sec"]
        )
        / baseline["mean_latency_sec"]
        * 100
    )

    throughput_change_pct = (
        (
            row["generated_tokens_per_second"]
            - baseline[
                "generated_tokens_per_second"
            ]
        )
        / baseline[
            "generated_tokens_per_second"
        ]
        * 100
    )

    example_throughput_change_pct = (
        (
            row["examples_per_second"]
            - baseline["examples_per_second"]
        )
        / baseline["examples_per_second"]
        * 100
    )

    result = row.to_dict()

    result[
        "footprint_reduction_pct"
    ] = footprint_reduction_pct

    result[
        "latency_change_pct"
    ] = latency_change_pct

    result[
        "throughput_change_pct"
    ] = throughput_change_pct

    result[
        "example_throughput_change_pct"
    ] = example_throughput_change_pct

    efficiency_rows.append(
        result
    )


efficiency_summary = pd.DataFrame(
    efficiency_rows
)

efficiency_summary.to_csv(
    OUTPUT_DIR
    / "efficiency_summary.csv",
    index=False,
)


fp16_baseline = (
    metrics[
        metrics["quantization"]
        == "fp16"
    ][
        [
            "model",
            "task",
            "macro_f1",
        ]
    ]
    .rename(
        columns={
            "macro_f1":
                "fp16_macro_f1"
        }
    )
)


paper_table = metrics.merge(
    fp16_baseline,
    on=[
        "model",
        "task",
    ],
    how="left",
)


paper_table[
    "f1_retention_pct"
] = (
    paper_table["macro_f1"]
    / paper_table["fp16_macro_f1"]
    * 100
)

paper_table[
    "absolute_f1_change"
] = (
    paper_table["macro_f1"]
    - paper_table["fp16_macro_f1"]
)

paper_table[
    "relative_f1_change_pct"
] = (
    (
        paper_table["macro_f1"]
        - paper_table["fp16_macro_f1"]
    )
    / paper_table["fp16_macro_f1"]
    * 100
)


paper_table = paper_table.merge(
    efficiency_summary[
        [
            "model",
            "quantization",
            "model_footprint_gb",
            "peak_inference_memory_gb",
            "mean_latency_sec",
            "median_latency_sec",
            "examples_per_second",
            "generated_tokens_per_second",
            "footprint_reduction_pct",
            "latency_change_pct",
            "throughput_change_pct",
        ]
    ],
    on=[
        "model",
        "quantization",
    ],
    how="left",
)


paper_table.to_csv(
    OUTPUT_DIR
    / "paper_master_table.csv",
    index=False,
)


average_performance = (
    metrics
    .groupby(
        [
            "model",
            "quantization",
        ],
        as_index=False,
    )
    .agg(
        mean_macro_f1=(
            "macro_f1",
            "mean",
        ),
        mean_accuracy=(
            "accuracy",
            "mean",
        ),
        mean_macro_precision=(
            "macro_precision",
            "mean",
        ),
        mean_macro_recall=(
            "macro_recall",
            "mean",
        ),
    )
)


average_performance.to_csv(
    OUTPUT_DIR
    / "average_performance.csv",
    index=False,
)


pareto = average_performance.merge(
    efficiency_summary[
        [
            "model",
            "quantization",
            "model_footprint_gb",
            "mean_latency_sec",
            "generated_tokens_per_second",
            "footprint_reduction_pct",
        ]
    ],
    on=[
        "model",
        "quantization",
    ],
    how="left",
)


pareto.to_csv(
    OUTPUT_DIR
    / "pareto_analysis.csv",
    index=False,
)


qwen_models = [
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
]


qwen_performance = performance[
    performance["model"].isin(
        qwen_models
    )
].copy()


task_sensitivity = (
    qwen_performance
    .groupby(
        "task",
        as_index=False,
    )
    .agg(
        fp16_mean=(
            "fp16",
            "mean",
        ),
        int8_mean=(
            "int8",
            "mean",
        ),
        int4_mean=(
            "int4",
            "mean",
        ),
        int8_mean_retention_pct=(
            "int8_retention_pct",
            "mean",
        ),
        int4_mean_retention_pct=(
            "int4_retention_pct",
            "mean",
        ),
    )
)


task_sensitivity.to_csv(
    OUTPUT_DIR
    / "task_sensitivity.csv",
    index=False,
)


model_sensitivity = (
    qwen_performance
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        fp16_mean=(
            "fp16",
            "mean",
        ),
        int8_mean=(
            "int8",
            "mean",
        ),
        int4_mean=(
            "int4",
            "mean",
        ),
        int8_mean_retention_pct=(
            "int8_retention_pct",
            "mean",
        ),
        int4_mean_retention_pct=(
            "int4_retention_pct",
            "mean",
        ),
    )
)


model_sensitivity.to_csv(
    OUTPUT_DIR
    / "model_sensitivity.csv",
    index=False,
)


cross_family_models = [
    "qwen2.5-3b-instruct",
    "falcon3-3b-instruct",
]


cross_family = performance[
    performance["model"].isin(
        cross_family_models
    )
].copy()


cross_family.to_csv(
    OUTPUT_DIR
    / "cross_family_comparison.csv",
    index=False,
)


cross_family_efficiency = (
    efficiency_summary[
        efficiency_summary["model"].isin(
            cross_family_models
        )
    ]
    .copy()
)


cross_family_efficiency.to_csv(
    OUTPUT_DIR
    / "cross_family_efficiency.csv",
    index=False,
)


print(
    "\nPERFORMANCE RETENTION\n"
)

print(
    performance[
        [
            "model",
            "task",
            "fp16",
            "int8",
            "int4",
            "int8_retention_pct",
            "int4_retention_pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nEFFICIENCY SUMMARY\n"
)

print(
    efficiency_summary[
        [
            "model",
            "quantization",
            "model_footprint_gb",
            "footprint_reduction_pct",
            "mean_latency_sec",
            "generated_tokens_per_second",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nQWEN TASK SENSITIVITY\n"
)

print(
    task_sensitivity
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nQWEN MODEL SENSITIVITY\n"
)

print(
    model_sensitivity
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nCROSS FAMILY COMPARISON\n"
)

print(
    cross_family[
        [
            "model",
            "task",
            "fp16",
            "int8",
            "int4",
            "int8_retention_pct",
            "int4_retention_pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nFILES SAVED\n"
)

for path in sorted(
    OUTPUT_DIR.glob("*.csv")
):
    print(path)