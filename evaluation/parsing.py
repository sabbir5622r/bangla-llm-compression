import re

from evaluation.prompts import TASK_LABELS


def parse_output(output, task):
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task}")

    text = output.strip()

    for label in TASK_LABELS[task]:
        if text.lower() == label.lower():
            return label, True

    matches = []

    for label in TASK_LABELS[task]:
        if re.search(
            rf"\b{re.escape(label)}\b",
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(label)

    if len(matches) == 1:
        return matches[0], True

    return None, False