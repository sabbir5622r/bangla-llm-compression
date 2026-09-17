from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


TASK_LABELS = {
    "sentiment": ["Positive", "Negative", "Neutral"],
    "nli": ["Entailment", "Neutral", "Contradiction"],
    "fake_news": ["Authentic", "Fake"],
}


def compute_metrics(y_true, y_pred, task):
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task}")

    labels = TASK_LABELS[task]

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if not y_true:
        raise ValueError("No predictions were provided.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics["f1_per_class"] = {
        label: score
        for label, score in zip(labels, per_class_f1)
    }

    return metrics