TASK_LABELS = {
    "sentiment": ["Positive", "Negative", "Neutral"],
    "nli": ["Entailment", "Neutral", "Contradiction"],
    "fake_news": ["Authentic", "Fake"],
}


def build_prompt(task, example):
    if task == "sentiment":
        return (
            "Classify the sentiment of the following Bangla text.\n"
            "Choose exactly one label: Positive, Negative, Neutral.\n"
            "Return only the label.\n\n"
            f"Text: {example['text']}"
        )

    if task == "nli":
        return (
            "Determine the relationship between the following Bangla "
            "premise and hypothesis.\n"
            "Choose exactly one label: Entailment, Neutral, Contradiction.\n"
            "Return only the label.\n\n"
            f"Premise: {example['Premise']}\n"
            f"Hypothesis: {example['Hypothesis']}"
        )

    if task == "fake_news":
        return (
            "Classify the following Bangla news article.\n"
            "Choose exactly one label: Authentic, Fake.\n"
            "Return only the label.\n\n"
            f"Article: {example['text']}"
        )

    raise ValueError(f"Unknown task: {task}")