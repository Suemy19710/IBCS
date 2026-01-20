from cProfile import label
import io, os
from typing import Optional, Dict, List
from torchvision import transforms
import torch
import torch.nn.functional as F
from ultralytics import YOLO    
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Notebook.core import create_mobilenet_rule_model, generate_feedback, CLASS_TO_RULE
from PIL import Image
import numpy as np
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="IBCS Compliance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ibcs-tau.vercel.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# -------------------------
# Model Storage
# -------------------------
models = {
    "mobilenet": None, 
    "yolo": None
}

YOLO_S1_CLASS_NAME = "non_zero_start"
YOLO_S1_CLASS_ID = 1
MOBILENET_CHECKPOINT_PATH = "./Checkpoints/mobilenet_rules.pth"
YOLO_CHECKPOINT_PATH = "./Notebook/runs/detect/ibcs_v2/weights/best.pt"

# Model  Loading
# -------------------------
def load_mobilenet(path: str):
    if not os.path.exists(path):
        logger.warning(f"MobileNet checkpoint not found at {path}")
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        model = create_mobilenet_rule_model()
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        logger.info(f"MobileNet model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load MobileNet model: {e}")
        raise

def load_yolo(path: str):
    if not os.path.exists(path):
        logger.warning(f"YOLO model not found at {path}")
        raise FileNotFoundError(f"YOLO model not found: {path}")
    try:
        model = YOLO(path)
        logger.info(f"YOLO model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

# Load models at startup
@app.on_event("startup")    
async def startup_event():
    logger.info("Server starting up... loading models")
    models["mobilenet"] = load_mobilenet(MOBILENET_CHECKPOINT_PATH)
    models["yolo"] = load_yolo(YOLO_CHECKPOINT_PATH)
    logger.info("All models loaded successfully")


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
# -------------------------
# Prediction Logic
# -------------------------
class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float
    rule: Optional[str]
    feedback: List[str]
    model_used: str 


# -------------------------
# Helper: process + predict
# -------------------------
def run_prediction(image: Image.Image):
    image_rgb = image.convert("RGB")

    # 1) YOLO first – S1_AxisNotZero
    yolo_model = models["yolo"]
    if yolo_model is not None:
        try:
            results = yolo_model(image_rgb, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())

                    if cls_id == YOLO_S1_CLASS_ID and conf >= 0.6:
                        return {
                            "class_id": 1,
                            "label": "Non-compliant",
                            "rule": "S1_AxisNotZero",
                            "confidence": conf,
                            "feedback": [
                                "YOLO detection: The axis does not start at zero. "
                                "IBCS recommends starting value axes at zero to avoid distortion."
                                "Document exceptions. If you don't start at zero for a good reason (e.g. medical doses), mention it in the title or subtitle."

                            ],
                            "model_used": "YOLO"
                        }
        except Exception as e:
            logger.error(f"YOLO prediction error: {e}")

    # 2) MobileNet – backup classifier (currently only S1 vs Compliant)
    mobilenet = models["mobilenet"]
    if mobilenet is not None:
        img_tensor = preprocess(image_rgb).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = mobilenet(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            class_id = int(torch.argmax(probs).item())
            confidence = float(probs[class_id].item())

        final_label = "Compliant"
        final_rule = None

        if class_id == 1 and confidence >= 0.5:
            final_label = "Non-compliant"
            final_rule = "S1_AxisNotZero"
        elif class_id in [2, 3, 4, 5]:
            logger.info(
                f"MobileNet predicted non-S1 class {class_id}, "
                f"but overridden to Compliant in current version."
            )
            final_label = "Compliant"
            final_rule = None
        elif class_id == 0:
            final_label = "Compliant"
            final_rule = None

        # Prepare feedback as a list of strings
        try:
            fb_dict = generate_feedback(final_rule, final_label, confidence)
            feedback = fb_dict.get("feedback", [])
        except Exception as e:
            logger.warning(f"Feedback generation failed: {e}")
            feedback = [
                f"Classification: {final_label}",
                f"Confidence: {confidence:.2%}",
                f"Rule: {final_rule if final_rule else 'N/A'}",
            ]

        return {
            "class_id": 1 if final_rule else 0,  # 0=Compliant, 1=Non-compliant
            "label": final_label,
            "rule": final_rule if final_rule else "Compliant",
            "confidence": confidence,
            "feedback": feedback,
            "model_used": "MobileNet"
        }

    raise RuntimeError("No models available to process request")

# -------------------------
# API endpoints
# -------------------------
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok" if models["mobilenet"] is not None else "degraded",
        "message": "IBCS Compliance API is running",
        "model_loaded": models["mobilenet"] is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_TO_RULE) if models["mobilenet"] is not None else 0
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if models["mobilenet"] is not None else "unhealthy",
        "model_loaded": models["mobilenet"] is not None,
        "device": str(DEVICE),
        "checkpoint_exists": os.path.exists("./Checkpoints/mobilenet_rules.pth")
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        result = run_prediction(image)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    # Lightweight endpoint for UptimeRobot to ping
    return {"status": "alive", "models": {k: v is not None for k, v in models.items()}}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
