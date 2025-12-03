import torch.nn as nn
from torchvision import models
from torchvision import transforms

NUM_CLASSES = 6

CLASS_TO_RULE = {
    0: ("Compliant", None),
    1: ("Non-compliant", "Scaling"),
    2: ("Non-compliant", "Titles"),
    3: ("Non-compliant", "Color_misuse"),
    4: ("Non-compliant", "Axis_check"),
    5: ("Non-compliant", "Clutter_detection"),
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
    if label == 'Compliant':
        feedback["feedback"].append(
            f"Great! It follows the {rule if rule else 'IBCS'} rules. This dashboard meets the required compliance standards. Layout, readability, and structure appear suitable."
            f"Confidence: {confidence:.0%}."
        )
        return feedback
    
    # Non-compliant: Give specific, helpful suggestions
    suggestions: List[str] = []
    
    if rule == 'Scaling':
        suggestions.extend([
            "**Start bar charts at zero.** If your y-axis starts above zero, it can make small differences look huge.",
            "**Use the same scale for similar charts.** If you're comparing sales across regions, all charts should have identical y-axis ranges.",
            "**Show scale breaks clearly.** If you must skip part of the scale, use a visible break symbol (like ~) so readers know.",
            "**Label your units.** Add '(in EUR)', '(%)' or similar to your axis labels.",
            "**Avoid dual axes unless absolutely necessary.** Two different scales on one chart confuse readers."
        ])
        
        if details:
            violations = details.get('violations', [])
            if 'axis_misaligned' in violations:
                suggestions.insert(0, "**Axis alignment issue detected.** Check the areas highlighted in red.")
            if 'non_zero_start' in violations:
                suggestions.insert(0, "**Your y-axis doesn't start at zero.** This can mislead viewers.")
            if 'inconsistent_scale' in violations:
                suggestions.insert(0, "**Different charts use different scales.** Make them uniform for fair comparison.")
    
    elif rule == 'Titles':
        suggestions.extend([
            "\n**Use descriptive titles.** Good example: 'Monthly Revenue (EUR) - Q1 2025'. Bad example: 'Chart 1'.",
            "\n**Include the 5 W's:** What (Revenue), Where (Netherlands), When (January 2025), how much (in thousands).",
            "\n**Put titles at the top** of each chart, not inside it.",
            "\n**Keep it concise but clear.** Aim for one line if possible."
        ])
        
        if details and 'missing_title' in details.get('violations', []):
            suggestions.insert(0, "**Missing title detected.** Every chart needs a clear heading.")
    
    elif rule == 'Color_misuse':
        suggestions.extend([
            "**Use color sparingly.** Only highlight what matters—usually negatives (red) or key data points.",
            "**Stick to IBCS colors:** Blue/grey for normal data, red for negative, green for positive variance.",
            "**Avoid rainbow charts.** Too many colors make it hard to focus.",
            "**Test in grayscale.** If your chart doesn't make sense in black and white, you're relying too much on color."
        ])
        
        if details and 'excessive_colors' in details.get('violations', []):
            suggestions.insert(0, "**Too many colors detected.** Simplify your color palette.")
    
    elif rule == 'Axis_check':
        suggestions.extend([
            "**Label both axes clearly.** Include units like '(thousands)', '(%)' or '(days)'.",
            "**Use readable tick marks.** Not too many (cluttered) or too few (unclear).",
            "**Rotate labels if needed.** Long category names work better at 45° or vertically.",
            "**Remove unnecessary gridlines.** Keep only horizontal lines for bar charts, only vertical for column charts."
        ])
        
        if details and 'missing_units' in details.get('violations', []):
            suggestions.insert(0, "**Missing units on axis.** Add '(EUR)', '(%)' etc. to your labels.")
    
    elif rule == 'Clutter_detection':
        suggestions.extend([
            "**Remove decorative elements.** 3D effects, shadows, and borders don't add value.",
            "**Delete redundant legends.** If you only have one data series, label it directly on the chart.",
            "**Simplify backgrounds.** Use white or light grey—no patterns or gradients.",
            "**Cut unnecessary labels.** If every bar is labeled, you don't need y-axis tick marks."
        ])
        
        if details and 'excessive_elements' in details.get('violations', []):
            suggestions.insert(0, "**Too many visual elements.** Simplify for better readability.")
    
    else:
        suggestions.append(f"Unknown rule: '{rule}'. Please check your configuration.")
    

    feedback["feedback"] = suggestions
    return feedback
 