from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import base64
import io
from PIL import Image
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# API Configuration
# -------------------------------
API_KEY = "AIzaSyAykFVuQApyKutFmMmDtbmV7xFhcItPnrg"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# -------------------------------------
# FastAPI Application Setup
# -------------------------------------
app = FastAPI(
    title="Plant Disease Recognition API",
    description="Backend API for the Plant Disease Recognition System",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# Data Models
# -------------------------------------
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    filename: str

class PredictionResponse(BaseModel):
    disease_name: str
    confidence: float
    description: Optional[str] = None
    symptoms: Optional[str] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    videos: Optional[str] = None

# -------------------------------------
# Model Loading
# -------------------------------------
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("new_trained_plant_disease_model.keras")
    return model

# -------------------------------------
# Disease Classes
# -------------------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -------------------------------------
# Image Processing and Prediction
# -------------------------------------
def preprocess_image(image_data):
    """Process base64 encoded image for model prediction"""
    try:
        # Decode base64 image
        decoded_image = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_image))
        
        # Resize and normalize
        image = image.resize((128, 128))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def predict_disease(image_array):
    """Make prediction using the model"""
    try:
        model = load_model()
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_index] * 100)
        
        return predicted_index, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------------
# Disease Information from Generative AI
# -------------------------------------
def get_disease_info(disease_name):
    """Get detailed disease information using Google's Generative AI"""
    try:
        # Clean disease name for better prompt formatting
        cleaned_name = disease_name.replace('___', ' - ').replace('_', ' ')
        
        # Create a structured prompt
        prompt = f"""
        Provide detailed information about the plant disease '{cleaned_name}' with the following sections:
        
        1. Description: A brief overview of what the disease is.
        2. Causes: What causes this disease (e.g., fungus, bacteria, virus).
        3. Symptoms: The visual symptoms that appear on the plant.
        4. Treatment: Recommended treatments and controls.
        5. Prevention: How to prevent this disease in the future.
        6. Useful Resources: Suggest types of videos that would be helpful (no actual links).
        
        Format each section with a clear heading.
        """

        # Generate content
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp")
        response = model.generate_content(prompt)
        
        if not response:
            return {
                "description": "Information not available.",
                "symptoms": "Information not available.",
                "treatment": "Information not available.",
                "prevention": "Information not available.",
                "videos": "No resource suggestions available."
            }
        
        # Simple parsing of the response into sections
        content = response.text
        
        # Extract sections (basic parsing - could be improved with regex)
        sections = {
            "description": "Information not available.",
            "symptoms": "Information not available.",
            "treatment": "Information not available.",
            "prevention": "Information not available.",
            "videos": "No resource suggestions available."
        }
        
        # Very basic section extraction - in a real app, use regex for better parsing
        if "Description" in content:
            description_start = content.find("Description")
            next_section = content.find("Causes", description_start)
            if next_section > 0:
                sections["description"] = content[description_start:next_section].replace("Description:", "").strip()
        
        if "Symptoms" in content:
            symptoms_start = content.find("Symptoms")
            next_section = content.find("Treatment", symptoms_start)
            if next_section > 0:
                sections["symptoms"] = content[symptoms_start:next_section].replace("Symptoms:", "").strip()
        
        if "Treatment" in content:
            treatment_start = content.find("Treatment")
            next_section = content.find("Prevention", treatment_start)
            if next_section > 0:
                sections["treatment"] = content[treatment_start:next_section].replace("Treatment:", "").strip()
        
        if "Prevention" in content:
            prevention_start = content.find("Prevention")
            next_section = content.find("Useful Resources", prevention_start)
            if next_section > 0:
                sections["prevention"] = content[prevention_start:next_section].replace("Prevention:", "").strip()
            else:
                sections["prevention"] = content[prevention_start:].replace("Prevention:", "").strip()
        
        if "Useful Resources" in content:
            resources_start = content.find("Useful Resources")
            sections["videos"] = content[resources_start:].replace("Useful Resources:", "").strip()
        
        return sections
        
    except Exception as e:
        return {
            "description": f"Error retrieving information: {str(e)}",
            "symptoms": "Information not available.",
            "treatment": "Information not available.",
            "prevention": "Information not available.",
            "videos": "No resource suggestions available."
        }

# -------------------------------------
# API Endpoints
# -------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plant Disease Recognition API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    """Process image and return disease prediction with information"""
    try:
        # Process image
        image_array = preprocess_image(request.image)
        
        # Make prediction
        predicted_index, confidence = predict_disease(image_array)
        disease_name = class_names[predicted_index]
        
        # Get disease information for non-healthy plants
        disease_info = {}
        if "healthy" not in disease_name.lower():
            disease_info = get_disease_info(disease_name)
        
        # Prepare response
        response = {
            "disease_name": disease_name,
            "confidence": confidence,
            **disease_info
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Return all possible disease classes"""
    return {"classes": class_names}

# -------------------------------------
# Server Startup
# -------------------------------------
if __name__ == "__main__":
    # Load model at startup
    load_model()
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
