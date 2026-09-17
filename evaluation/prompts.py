TASK_LABELS = {
    "stance": ["Pro-Uprising", "Anti-Uprising", "Neutral"],
    "nli": ["Entailment", "Neutral", "Contradiction"],
    "fake_news": ["Authentic", "Fake"],
}


def build_prompt(task, example):
    if task == "stance":
        return (
            "Determine the stance of the following Bangla social media "
            "comment toward the July Revolution/uprising in Bangladesh.\n\n"
            "Use these labels:\n"
            "Pro-Uprising: the comment supports the uprising, protesters, "
            "or protesting students, or expresses opposition to actions "
            "against them.\n"
            "Anti-Uprising: the comment opposes, criticizes, or attacks "
            "the uprising, protesters, or protesting students, or supports "
            "actions against them.\n"
            "Neutral: the comment does not express a clear position either "
            "for or against the uprising.\n\n"
            "Choose exactly one label: Pro-Uprising, Anti-Uprising, Neutral.\n"
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