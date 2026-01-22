import torch.nn as nn
from torchvision import models
from torchvision import transforms
from typing import List 

NUM_CLASSES = 3  # Compliant, Non-compliant S1, Non-compliant S2


CLASS_TO_RULE = {
    0: ("Compliant", None),
    1: ("Non-compliant", "S1_AxisNotZero"),
    2: ("Non-compliant", "S2_IBCSOverallRuleViolation"),
}


RULE_DESCRIPTIONS = {
    "S1_AxisNotZero": "Axis does not start at zero",
    "S2_IBCSOverallRuleViolation": "Overall IBCS rule violation",
}

# -----------------------------
# Rule-specific resources
# -----------------------------
RULE_RESOURCES = {
    "S1_AxisNotZero": {
        "label": "IBCS – Scaling / Axis starts at zero",
        "link": "https://www.ibcs.com/resource/top-and-bottom-5-of-international-business-communication-standards/"
    },
    "S2_IBCSOverallRuleViolation": {
        "label": "IBCS – General chart design principles",
        "link": "https://www.ibcs.com/IBCS/"
    }
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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def generate_feedback(
    rule: str,
    label: str,
    confidence: float,
    details: dict = None
) -> dict:

    feedback = {
        "label": label,
        "confidence": round(confidence, 2),
        "feedback": []
    }

    # -----------------------------
    # Compliant case
    # -----------------------------
    if label == "Compliant":
        feedback["feedback"].append(
            f"Great! This visualization appears compliant with the expected charting rules. "
            f"Layout, readability, and structure look suitable. "
            f"Confidence: {confidence:.0%}."
        )
        return feedback

    # -----------------------------
    # Non-compliant feedback
    # -----------------------------
    suggestions: List[str] = []
    rule_desc = RULE_DESCRIPTIONS.get(rule, rule)

    # S1 – Axis does not start at zero
    if rule == "S1_AxisNotZero":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Start value axes at zero whenever possible. Non-zero starts can exaggerate small differences.",
            "Use a clear break symbol (like `//` or `~`) if you must cut off part of the axis.",
            "Document exceptions. If you don't start at zero for a good reason (e.g. medical doses), mention it in the title or subtitle."
        ])

        if details:
            if "non_zero_start" in details.get("violations", []):
                suggestions.insert(
                    1,
                    "Your axis seems to start above zero. This can mislead readers about the real magnitude of changes."
                )

    # S2 – Overall IBCS Rule Violation
    elif rule == "S2_IBCSOverallRuleViolation":
        suggestions.extend([
            f"Issue detected: {rule_desc}.",
            "Follow IBCS standards for chart types, colors, labels, and layout to ensure clarity and consistency.",
            "Use standardized symbols and notations as per IBCS guidelines.",
            "Ensure all chart elements (titles, axes, legends) are clearly labeled and easy to understand."
        ])

        if details:
            violations = details.get("violations", [])
            if "wrong_chart_type" in violations:
                suggestions.insert(
                    1,
                    "The chosen chart type may not be suitable for the data being presented. "
                    "Consider using a different type that aligns with IBCS recommendations."
                )
            if "color_scheme" in violations:
                suggestions.insert(
                    1,
                    "The color scheme used does not conform to IBCS standards. "
                    "Use the recommended colors for better readability and consistency."
                )

    # Fallback for unknown rules
    else:
        suggestions.append(
            f"Rule code `{rule}` was detected, but no detailed guidance is configured. "
            "Please check your rule mapping or add guidance for this rule."
        )

    # -----------------------------
    # Add "Read more" if configured
    # -----------------------------
    resource = RULE_RESOURCES.get(rule)
    if resource:
        suggestions.append(
            f"Read more: {resource['label']} – {resource['link']}"
        )

    feedback["feedback"] = suggestions
    return feedback