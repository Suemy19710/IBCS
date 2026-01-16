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
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your custom modules
try:
    from Notebook.core import create_mobilenet_rule_model, generate_feedback, CLASS_TO_RULE
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    raise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="IBCS Compliance API", version="1.0.0")

# IMPROVED CORS Configuration - Must be added BEFORE routes
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
# Load your PyTorch model
# -------------------------
def load_model_from_checkpoint(path: str):
    try:
        logger.info(f"Loading model from {path}")
        model = create_mobilenet_rule_model()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Try to load model, but allow server to start even if it fails
model = None
try:
    checkpoint_path = "./Checkpoints/mobilenet_rules.pth"
    if os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path)
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}")
except Exception as e:
    logger.error(f"Cannot load model: {e}")

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
    """Preprocess PIL image for model input"""
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
    """Run model prediction on image"""
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

        logger.info(f"Predicted class_id: {class_id}, confidence: {confidence:.3f}")

        # Get label and rule
        if class_id not in CLASS_TO_RULE:
            logger.warning(f"Unknown class_id {class_id}, defaulting to non-compliant")
            label = "Non-compliant"
            rule = "Unknown"
        else:
            label, rule = CLASS_TO_RULE[class_id]

        logger.info(f"Label: {label}, Rule: {rule}")

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
            logger.warning(f"Feedback generation failed: {e}")
            feedback = [
                f"Classification: {label}",
                f"Confidence: {confidence:.2%}",
                f"Rule: {rule if rule else 'N/A'}"
            ]

        return class_id, label, rule, confidence, feedback
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback 
        traceback.print_exc()
        raise


# -------------------------
# API endpoints
# -------------------------
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok" if model is not None else "degraded",
        "message": "IBCS Compliance API is running",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(CLASS_TO_RULE) if model is not None else 0
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "checkpoint_exists": os.path.exists("./Checkpoints/mobilenet_rules.pth")
    }


@app.get("/api/classes")
async def get_classes():
    """Get available classification classes"""
    return {
        "classes": CLASS_TO_RULE,
        "device": str(DEVICE),
        "model_loaded": model is not None
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict IBCS compliance from uploaded chart image
    """
    # Check if model is loaded
    if model is None:
        logger.error("Prediction attempted with no model loaded")
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        logger.info(f"Received file: {file.filename}, type: {file.content_type}")
        
        # Validate content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Read file contents
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        # Convert to PIL image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            logger.info(f"Image opened successfully: {image.size}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Sample rule details (replace with actual detection logic)
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

        # Run prediction
        class_id, label, rule, confidence, feedback = run_prediction(
            image, details_by_rule=fake_details
        )

        logger.info(f"Prediction successful: {label} (confidence: {confidence:.3f})")

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
        logger.error(f"Endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
