
# Pipeline:
# 1) YOLO model detects axis-related issues on dashboard images.
# 2) We derive:
#    - status: "Compliant" or "Non-compliant"
#    - issues: list of issue codes (S1..S5)
# 3) A small CPU LLM turns that into human-readable feedback.

import io
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch
from ultralytics import YOLO
from transformers import pipeline


# ------------------------------------------------
# Device setup
# ------------------------------------------------
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# FastAPI setup
# ------------------------------------------------
app = FastAPI(
    title="IBCS Axis Compliance API (YOLO-based)",
    description="Uses YOLO to detect axis-related IBCS issues and a LLM "
                "to generate explanations and recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# YOLO model (axis rules)
# ------------------------------------------------
"""
Your YOLO model must be trained with these 6 classes in this order:

0: OK_NoIssue
1: S1_AxisNotZero
2: S2_UnequalTickSpacing
3: S3_DistortedScaleRange
4: S4_MissingAxisValues
5: S5_MisusedDualAxis
"""

yolo_path = YOLO("Notebook/runs/detect/ibcs_v2/weights/best.pt")

try:
    yolo_model = YOLO(yolo_path)
    # print(f"[INFO] YOLO model loaded from {yolo_path}")
except Exception as e:
    print(f"[CRITICAL] Failed to load YOLO model: {e}")
    yolo_model = None

OK_CLASS = "OK_NoIssue"
ISSUE_CLASSES = [
    "S1_AxisNotZero",
    "S2_UnequalTickSpacing",
    "S3_DistortedScaleRange",
    "S4_MissingAxisValues",
    "S5_MisusedDualAxis",
]

# ------------------------------------------------
# Small LLM for feedback (CPU)
# ------------------------------------------------
"""
Use a small, CPU-friendly text-generation model.
'distilgpt2' is a tiny baseline. Replace with a better small model later.
"""

TEXT_LLM_MODEL_NAME = "distilgpt2"  # TODO: swap for a better small model later

feedback_llm = pipeline(
    task="text-generation",
    model=TEXT_LLM_MODEL_NAME,
    device=-1,   # CPU
)
print(f"[INFO] Text generation model loaded: {TEXT_LLM_MODEL_NAME}")

# ------------------------------------------------
def summarize_issues(issues: List[str]) -> str:
    """Convert issue codes into a short technical description."""
    if not issues:
        return "No axis-related issues detected. Axes appear to start at zero and follow IBCS."

    parts = []
    for issue in issues:
        if issue == "S1_AxisNotZero":
            parts.append("The axis does not start at zero, which can exaggerate differences.")
        elif issue == "S2_UnequalTickSpacing":
            parts.append("Tick marks on the axis are irregularly spaced, which can confuse readers.")
        elif issue == "S3_DistortedScaleRange":
            parts.append("The scale range is distorted, making values look larger or smaller than they are.")
        elif issue == "S4_MissingAxisValues":
            parts.append("Axis values or labels are missing, reducing interpretability.")
        elif issue == "S5_MisusedDualAxis":
            parts.append("Dual axes are misused, making comparison between series unclear or misleading.")
        else:
            parts.append(f"Issue: {issue}")
    return " ".join(parts)

def generate_feedback_with_llm(status: str, issues: List[str]) -> List[str]:
    """
    Use the small LLM to produce:
    - Issue explanation
    - Recommendation

    Returns: list of strings for the API response.
    """
    technical_desc = summarize_issues(issues)

    prompt = f"""
You are an IBCS compliance assistant focusing on axis rules.

We classify dashboards as "Compliant" when their axes start at zero
and scaling is not misleading. We classify them as "Non-compliant" when
there are axis-related issues such as non-zero baselines or distorted scales.

Classification result:
- Status: {status}
- Detected issues: {", ".join(issues) if issues else "None"}
- Technical description: {technical_desc}

If Status is "Compliant":
- Explain in one or two short sentences why the chart is okay according to IBCS.
- Give one short recommendation to keep following good practice.

If Status is "Non-compliant":
- Explain in one or two short sentences what is wrong with the axis/scaling.
- Give one clear recommendation in one sentence on how to fix it,
  for example: start axes at zero, avoid distorted ranges, use equal tick spacing,
  avoid misused dual axes, or show all necessary axis values.

Respond in this exact multi-line format:

Status:
Issue:
Recommendation:
"""

    out = feedback_llm(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
    )

    text = out[0]["generated_text"]

    status_line = ""
    issue_line = ""
    recommendation_line = ""

    try:
        if "Status:" in text:
            part = text.split("Status:", 1)[1]
            status_line = part.split("\n", 1)[0].strip()
        if "Issue:" in text:
            part = text.split("Issue:", 1)[1]
            issue_line = part.split("Recommendation:", 1)[0].strip()
        if "Recommendation:" in text:
            part = text.split("Recommendation:", 1)[1]
            recommendation_line = part.strip()
    except Exception as e:
        print(f"[WARNING] Failed to parse LLM output cleanly: {e}")
    
    feedback = []
    feedback.append(f"Classification: {status}")
    if issues:
        feedback.append(f"Detected issues: {', '.join(issues)}")
    else:
        feedback.append("Detected issues: None")

    if status_line:
        feedback.append(f"Status: {status_line}")
    if issue_line:
        feedback.append(f"Issue: {issue_line}")
    if recommendation_line:
        feedback.append(f"Recommendation: {recommendation_line}")

    # Fallback
    if len(feedback) <= 2:
        if status == "Compliant":
            feedback.append("Issue: No major axis-related IBCS issues detected.")
            feedback.append("Recommendation: Keep axes starting at zero and avoid misleading scaling.")
        else:
            feedback.append("Issue: The chart likely violates one or more IBCS axis rules.")
            feedback.append("Recommendation: Ensure axes start at zero, use consistent tick spacing, "
                            "and avoid distorted scale ranges or misused dual axes.")
            

    return feedback
# ------------------------------------------------
# YOLO-based analysis
# ------------------------------------------------
def analyze_with_yolo(image: Image.Image) -> Tuple[str, List[str], float]:
    """
    Run YOLO on the image and derive:
      - status: "Compliant" / "Non-compliant"
      - issues: list of issue class names (S1..S5)
      - max_conf: highest confidence among detected boxes
    """
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    # YOLO accepts PIL images directly
    results = yolo_model(image, device=YOLO_DEVICE, verbose=False)[0]

    names = results.names  # dict: class_id -> class_name
    boxes = results.boxes

    # Collect unique predicted classes and max confidence
    pred_class_ids = set()
    max_conf = 0.0

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        pred_class_ids.add(cls_id)
        if conf > max_conf:
            max_conf = conf

    pred_classes = [names[i] for i in pred_class_ids]

    # Decide status and issues
    # If we only see OK_NoIssue (or nothing), treat as Compliant.
    # If we see any of S1..S5, treat as Non-compliant.
    if not pred_classes:
        # no detections – you can decide; here we treat as Compliant but low confidence
        status = "Compliant"
        issues = []
        max_conf = 0.0
    elif all(c == OK_CLASS for c in pred_classes):
        status = "Compliant"
        issues = []
    else:
        status = "Non-compliant"
        issues = [c for c in pred_classes if c in ISSUE_CLASSES]

    return status, issues, max_conf

# ------------------------------------------------
# Response schema
# ------------------------------------------------
class PredictionResponse(BaseModel):
    status: str                 # "Compliant" / "Non-compliant"
    issues: List[str]           # list of YOLO class names (S1..S5)
    confidence: float           # max box confidence
    feedback: List[str]         # explanation + recommendation


# ------------------------------------------------
# API endpoint
# ------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    POST /predict

    Body: multipart/form-data with 'file' = dashboard image.

    Returns:
      - status ("Compliant"/"Non-compliant")
      - issues (list of S1..S5 codes)
      - confidence (max YOLO detection confidence)
      - feedback (LLM-generated explanation & recommendation)
    """
    try:
        print(f"[INFO] Received file: {file.filename}, type: {file.content_type}")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file.")

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"[INFO] Image opened: {image.size}")
        except Exception as e:
            print(f"[ERROR] Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        status, issues, max_conf = analyze_with_yolo(image)
        feedback = generate_feedback_with_llm(status, issues)

        return PredictionResponse(
            status=status,
            issues=issues,
            confidence=max_conf,
            feedback=feedback,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ------------------------------------------------
# Health check endpoint
# ------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok" if yolo_model is not None else "degraded",
        "message": "IBCS Axis Compliance API (YOLO-based) is running",
        "yolo_loaded": yolo_model is not None,
        "device": str(YOLO_DEVICE),
        "classes": ISSUE_CLASSES + [OK_CLASS],
    }