import io, os
from typing import Optional, Dict, List
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision import models
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from Notebook.core import create_mobilenet_rule_model, generate_feedback, CLASS_TO_RULE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ibcs-tau.vercel.app",
        "http://localhost:8000",  # For local development
        "http://127.0.0.1:000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# -------------------------
# Load your PyTorch model
# -------------------------
def load_model_from_checkpoint(path: str):
    try:
        model = create_mobilenet_rule_model()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        print(f"[INFO] Model loaded from {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

try:
    model = load_model_from_checkpoint("./Checkpoints/mobilenet_rules.pth")
except Exception as e:
    print(f"[CRITICAL] Cannot start without model: {e}")
    model = None

# Image preprocessing for MobileNet (PyTorch)
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = preprocess(image)
    return tensor

# -------------------------
# Response schema
# -------------------------
class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float
    rule: Optional[str]
    feedback: List[str]


# -------------------------
# Helper: process + predict
# -------------------------

def run_prediction(
    image: Image.Image,
    details_by_rule: Optional[Dict[str, Dict]] = None
):
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Preprocess image
        img_tensor = preprocess_image(image).to(DEVICE).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            class_id = int(torch.argmax(probs).item())
            confidence = float(probs[class_id].item())

        print(f"[DEBUG] Predicted class_id: {class_id}, confidence: {confidence:.3f}")

        # Get label and rule
        if class_id not in CLASS_TO_RULE:
            print(f"[WARNING] Unknown class_id {class_id}, defaulting to non-compliant")
            label = "Non-compliant"
            rule = "Unknown"
        else:
            label, rule = CLASS_TO_RULE[class_id]

        print(f"[DEBUG] Label: {label}, Rule: {rule}")

        # Generate rule-based feedback
        try:
            if label == "Compliant":
                fb = generate_feedback("IBCS", label, confidence)
            else:
                rule_details = None
                if details_by_rule and rule and rule in details_by_rule:
                    rule_details = details_by_rule[rule]
                fb = generate_feedback(rule if rule else "Unknown", label, confidence, rule_details)
            
            feedback = fb.get("feedback", [])
        except Exception as e:
            print(f"[WARNING] Feedback generation failed: {e}")
            feedback = [
                f"Classification: {label}",
                f"Confidence: {confidence:.2%}",
                f"Rule: {rule if rule else 'N/A'}"
            ]

        return class_id, label, rule, confidence, feedback
    
    except Exception as e:
        print(f"[ERROR] Prediction failed:{str(e)}")
        import traceback 
        traceback.print_exc()
        raise


# -------------------------
# API endpoint
# -------------------------
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        print(f"[INFO] Received file: {file.filename}, type: {file.content_type}")
        
        # Validate content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        contents = await file.read()
        print(f"[INFO] File size: {len(contents)} bytes")

        # Convert to PIL image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"[INFO] Image opened: {image.size}")
        except Exception as e:
            print(f"[ERROR] Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        fake_details = {
            "S1_AxisNotZero": {
                "violations": ["non_zero_start"]
            },
            "S2_UnequalTickSpacing": {
                "violations": ["irregular_ticks"]
            },
            "S3_DistortedScaleRange": {
                "violations": ["inconsistent_range", "overzoomed"]
            },
            "S4_MissingAxisValues": {
                "violations": ["missing_units", "too_few_labels"]
            },
            "S5_MisusedDualAxis": {
                "violations": ["unlabeled_secondary_axis", "confusing_overlap"]
            },
        }


        class_id, label, rule, confidence, feedback = run_prediction(
            image, details_by_rule=fake_details
        )

        print(f"[INFO] Prediction successful: {label}")

        return PredictionResponse(
            class_id=class_id,
            label=label,
            rule=rule if rule else "unknown",
            confidence=confidence,
            feedback=feedback
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "ok" if model is not None else "degraded",
        "message": "IBCS Compliance API is running",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_TO_RULE)
    }


# Debug endpoint to check model classes
@app.get("/api/classes")
async def get_classes():
    return {
        "classes": CLASS_TO_RULE,
        "device": str(DEVICE),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
