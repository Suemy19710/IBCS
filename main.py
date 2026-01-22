import io, os
from typing import Optional, Dict, List
import numpy as np
import tensorflow as tf
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

# Log PORT at module level for debugging
PORT = os.environ.get("PORT", "Not set")
logger.info(f"PORT environment variable at startup: {PORT}")

DEVICE = tf.config.list_physical_devices("GPU") if tf.config.list_physical_devices("GPU") else tf.config.list_physical_devices("CPU")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="IBCS Compliance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://ibcs-tau.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------------------------
# Model Storage
# -------------------------
models = {
    "mobilenet": None, 
    "yolo": None
}

# YOLO_S1_CLASS_NAME = "non_zero_start"
YOLO_S1_CLASS_ID = 1
MOBILENET_KERAS_PATH = "./Checkpoints/mobilenet-21-01.keras"
YOLO_CHECKPOINT_PATH = "./Notebook/runs/detect/ibcs_v2/weights/best.pt"
YOLO_S1_CLASS_ID = 1
YOLO_S2_CLASS_ID = 2 

# -------------------------
# Model Loading
# -------------------------
def load_mobilenet_keras(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keras model not found: {path}")
    try:
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

# Image preprocessing
IMG_SIZE = 224
def preprocess_for_keras(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32')
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

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

def run_prediction(image: Image.Image):
    image_rgb = image.convert("RGB")
    final_label, final_rule = CLASS_TO_RULE[0]
    final_confidence = 0.0
    model_pathway = "System Init"

    # MobileNet
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

    # YOLO
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
    return {
        "status": "ok" if models["mobilenet"] is not None else "degraded",
        "message": "IBCS Compliance API is running",
        "model_loaded": models["mobilenet"] is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_TO_RULE) if models["mobilenet"] is not None else 0,
        "port": os.environ.get("PORT", "Not set")
    }

@app.get("/health")
async def health():
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
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Running in __main__ mode on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
