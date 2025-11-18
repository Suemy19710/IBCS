# main.py
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

# Initialize FastAPI
app = FastAPI()

# Allow frontend requests (during development allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load your MobileNet model
# -------------------------
try:
    model = load_model("Checkpoints/mobilenet.keras")
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR loading model:", e)
    raise e


# Response model for API
class PredictionResponse(BaseModel):
    compliance: str
    message: str


# -------------------------
# Helper: process + predict
# -------------------------
def predict_compliance(image: Image.Image):
    # Resize to MobileNet input size
    image = image.resize((224, 224))
    img_array = np.array(image)

    # MobileNet preprocessing
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    score = float(preds[0][0])  # assuming model outputs single probability

    # Decide label
    if score >= 0.5:
        compliance = "Compliant 👍"
        message =         "This dashboard meets the required compliance standards. "
        "Layout, readability, and structure appear suitable."
    else:
        compliance = "Non-compliant 👎"
        message =         "This dashboard does not meet the required compliance standards. "
        "Improvements are recommended to enhance structure and clarity."

    return PredictionResponse(
        compliance=compliance,
        message=message
    )


# -------------------------
# API endpoint
# -------------------------
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Read file bytes
    contents = await file.read()

    # Convert to PIL image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Predict using MobileNet
    result = predict_compliance(image)

    return result
