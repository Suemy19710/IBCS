import torch.nn as nn
from torchvision import models
from torchvision import transforms
from typing import List 

NUM_CLASSES = 6


CLASS_TO_RULE = {
    0: ("Compliant", None),
    1: ("Non-compliant", "S1_AxisNotZero"),
    2: ("Non-compliant", "S2_UnequalTickSpacing"),
    3: ("Non-compliant", "S3_DistortedScaleRange"),
    4: ("Non-compliant", "S4_MissingAxisValues"),
    5: ("Non-compliant", "S5_MisusedDualAxis"),
}


RULE_DESCRIPTIONS = {
    "S1_AxisNotZero": "Axis does not start at zero",
    "S2_UnequalTickSpacing": "Unequal spacing between tick marks",
    "S3_DistortedScaleRange": "Distorted or inconsistent scale ranges",
    "S4_MissingAxisValues": "Missing or incomplete axis values",
    "S5_MisusedDualAxis": "Misuse of dual axes",
}
def create_mobilenet_rule_model(num_classes: int = NUM_CLASSES):
    model = models.mobilenet_v3_small(pretrained=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def preprocess_image(image_path: str):
    preprocess_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def generate_feedback(rule: str, label: str, confidence: float, details: dict = None) -> dict:
    feedback = {
        "label": label,
        "confidence": round(confidence, 2),
        "feedback": []
    }

    # If compliant
    if label == "Compliant":
        feedback["feedback"].append(
            f"Great! This visualization appears compliant with the expected charting rules. "
            f"Layout, readability, and structure look suitable. "
            f"Confidence: {confidence:.0%}."
        )
        return feedback

    # Non-compliant: Give specific, helpful suggestions
    suggestions: List[str] = []
    rule_desc = RULE_DESCRIPTIONS.get(rule, rule)

    # S1 – Axis does not start at zero
    if rule == "S1_AxisNotZero":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Start value axes at zero whenever possible. Non-zero starts can exaggerate small differences.",
            "Use a clear break symbol (like `//` or `~`) if you *must* cut off part of the axis.",
            "Document exceptions. If you don't start at zero for a good reason (e.g. medical doses), mention it in the title or subtitle."
        ])
        if details:
            if "non_zero_start" in details.get("violations", []):
                suggestions.insert(1, "Your axis seems to start above zero. This can mislead readers about the real magnitude of changes.")

    # S2 – Unequal tick spacing
    elif rule == "S2_UnequalTickSpacing":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Keep tick marks evenly spaced for linear scales. Irregular spacing makes trends hard to read.",
            "Check for mixed scales. Make sure you're not accidentally mixing log and linear behavior.",
            "Use gridlines consistently so the gaps between values visually match the numeric distances."
        ])
        if details:
            if "irregular_ticks" in details.get("violations", []):
                suggestions.insert(1, "Irregular tick spacing detected. Align tick positions with their numeric values.")

    # S3 – Distorted scale range
    elif rule == "S3_DistortedScaleRange":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Use comparable ranges when comparing charts. If two charts compare the same metric, keep the same min/max.",
            "Avoid extreme zooming on small ranges if it exaggerates noise.",
            "Highlight truncated ranges clearly in the title or annotation so viewers know the scale is limited."
        ])
        if details:
            v = details.get("violations", [])
            if "inconsistent_range" in v:
                suggestions.insert(1, "Different charts use different ranges for the same metric. Normalize the range to compare fairly.")
            if "overzoomed" in v:
                suggestions.insert(1, "The scale is very tight. Consider widening it to give more context.")

    # S4 – Missing axis values
    elif rule == "S4_MissingAxisValues":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Label your axes clearly. Make sure both axis titles and units are visible, e.g. `Revenue (kEUR)`.",
            "Include enough tick labels so readers can estimate values, not just direction.",
            "Avoid overlapping labels. Rotate or abbreviate labels rather than dropping them entirely."
        ])
        if details:
            v = details.get("violations", [])
            if "missing_units" in v:
                suggestions.insert(1, "Units are missing on at least one axis. Add `(%)`, `(days)`, `(EUR)`, etc.")
            if "too_few_labels" in v:
                suggestions.insert(1, "Very few axis labels detected. Add more tick labels so values can be interpreted.")

    # S5 – Misused dual axis
    elif rule == "S5_MisusedDualAxis":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Avoid dual axes if possible. They are often confusing and easy to misread.",
            "If you must use dual axes, use clearly different chart types (e.g. bars for one metric, line for the other).",
            "Align the story, not the shapes. Make sure you're not implying correlation just because lines overlap.",
            "Consider small multiples instead. Two simpler charts are often clearer than one complex dual-axis chart."
        ])
        if details:
            v = details.get("violations", [])
            if "unlabeled_secondary_axis" in v:
                suggestions.insert(1, "The secondary axis is unlabeled. Add a clear title and units.")
            if "confusing_overlap" in v:
                suggestions.insert(1, "Lines/series overlap in a misleading way. Separate them or split into multiple charts.")

    # Fallback if we get an unknown rule code
    else:
        suggestions.append(
            f"Rule code `{rule}` was detected, but no detailed guidance is configured. "
            "Please check your rule mapping or add guidance for this rule."
        )

    feedback["feedback"] = suggestions
    return feedback