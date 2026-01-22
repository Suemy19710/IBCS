import io, os
from typing import Optional, Dict, List
import numpy as np
import tensorflow as tf  # Changed from torch
from PIL import Image
from ultralytics import YOLO    
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Notebook.core import generate_feedback, CLASS_TO_RULE
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = tf.config.list_physical_devices("GPU") if tf.config.list_physical_devices("GPU") else tf.config.list_physical_devices("CPU")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="IBCS Compliance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
            "http://127.0.0.1:5500",      # local frontend
            "http://localhost:5500",      # if Live Server uses this
            "https://ibcs-tau.vercel.app" # deployed frontend
        # "https://ibcs-tau.vercel.app",
        # "http://localhost:8000", # 8000 for deployed backend 
        # "http://127.0.0.1:8000",
        # "http://localhost:3000", # when running localhost main.py
        # "http://127.0.0.1:3000",
        # "http://127.0.0.1:5500", # 5500 for frontend (not sure it works)

    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
)

# For Render deployment
if os.environ.get("RENDER", None):
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# -------------------------
# Model Storage
# -------------------------
models = {
    "mobilenet": None, 
    "yolo": None
}

YOLO_S1_CLASS_NAME = "non_zero_start"
YOLO_S1_CLASS_ID = 1
MOBILENET_KERAS_PATH = "./Checkpoints/mobilenet-21-01.keras" # New path
YOLO_CHECKPOINT_PATH = "./Notebook/runs/detect/ibcs_v2/weights/best.pt"
YOLO_S1_CLASS_ID = 1
YOLO_S2_CLASS_ID = 2 

# -------------------------
# Model  Loading
# -------------------------
def load_mobilenet_keras(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keras model not found: {path}")
    try:
        # Loading the .keras file
        model = tf.keras.models.load_model(path)
        logger.info(f"MobileNet (Keras) loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Keras model: {e}")
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
    try: 
        models["mobilenet"] = load_mobilenet_keras(MOBILENET_KERAS_PATH)
    except Exception as e:  
        logger.error(f"Error loading MobileNet model: {e}") 
        models["mobilenet"] = None  
    try:
        models["yolo"] = load_yolo(YOLO_CHECKPOINT_PATH)
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        models["yolo"] = None   

# Image preprocessing for MobileNet (PyTorch)
IMG_SIZE = 224
def preprocess_for_keras(image: Image.Image):
    """
    Keras MobileNet expects (224, 224, 3) and values typically in [-1, 1] 
    or [0, 1] depending on how you trained it.
    """
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32')
    
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    
    return np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
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
    final_label, final_rule = CLASS_TO_RULE[0]
    final_confidence = 0.0
    model_pathway = "System Init"

    # --- 1) MobileNet (Keras) ---
    mb_model = models["mobilenet"]
    if mb_model:
        img_batch = preprocess_for_keras(image_rgb)
        preds = mb_model.predict(img_batch, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])

        if class_id in CLASS_TO_RULE:
            final_label, final_rule = CLASS_TO_RULE[class_id]
            final_confidence = confidence
            model_pathway = f"MobileNet ({final_label})"

    # --- 2) YOLO (PyTorch) ---
    yolo_model = models["yolo"]
    if yolo_model:
        results = yolo_model(image_rgb, verbose=False)
        yolo_rule_found = None
        yolo_best_conf = 0.0

        for result in results:
            if not result.boxes: continue
            for box in result.boxes:
                cls_id, conf = int(box.cls.item()), float(box.conf.item())
                
                det_rule = None
                if cls_id == YOLO_S1_CLASS_ID: det_rule = "S1_AxisNotZero"
                elif cls_id == YOLO_S2_CLASS_ID: det_rule = "S2_IBCSOverallRuleViolation"

                if det_rule and conf >= 0.5 and conf > yolo_best_conf:
                    yolo_rule_found = det_rule
                    yolo_best_conf = conf

        # Override Logic
        if yolo_rule_found:
            final_label, final_rule, final_confidence = "Non-compliant", yolo_rule_found, yolo_best_conf
            model_pathway += " -> YOLO Overrode"

    feedback_data = generate_feedback(final_rule, final_label, final_confidence)
    logger.info(f"Prediction: {final_label} ({final_rule}) with confidence {final_confidence:.2%} via {model_pathway}")

    return {
        "class_id": 1 if final_label == "Non-compliant" else 0,
        "label": final_label,
        "rule": final_rule or "N/A",
        "confidence": final_confidence,
        "feedback": feedback_data["feedback"],
        "model_used": model_pathway
    }
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
