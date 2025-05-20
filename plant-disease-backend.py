from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import base64
import io
from PIL import Image
from typing import Dict, Any, List, Optional
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# API Configuration
# -------------------------------
client = OpenAI(
    base_url="http://39.105.194.16:6691/v1/",  # 设置自定义API地址
    api_key="YOUR_API_KEY"  # 替换为您的API密钥
)

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
# Disease Information from OpenAI
# -------------------------------------
def get_disease_info(disease_name):
    """使用OpenAI获取详细的疾病信息"""
    try:
        # 清理疾病名称以获得更好的提示格式
        cleaned_name = disease_name.replace('___', ' - ').replace('_', ' ')
        
        # 创建结构化提示
        prompt = f"""
        提供关于植物疾病'{cleaned_name}'的详细信息，包含以下部分：
        
        1. 描述：该疾病的简要概述。
        2. 原因：导致这种疾病的原因（例如，真菌、细菌、病毒）。
        3. 症状：植物上出现的视觉症状。
        4. 治疗：推荐的治疗方法和控制措施。
        5. 预防：如何预防这种疾病。
        6. 有用资源：建议有帮助的视频类型（不包含实际链接）。
        
        请为每个部分添加清晰的标题。
        """

        # 使用新的API格式创建聊天完成
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {"role": "system", "content": "你是一个专业的植物病理学专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        if not response or not response.choices:
            return {
                "description": "信息不可用。",
                "symptoms": "信息不可用。",
                "treatment": "信息不可用。",
                "prevention": "信息不可用。",
                "videos": "没有可用的资源建议。"
            }
        
        # 获取生成的内容并清理思考过程标签
        content = response.choices[0].message.content
        
        # 清理<think></think>标签及其内容
        if "<think>" in content and "</think>" in content:
            think_start = content.find("<think>")
            think_end = content.find("</think>") + len("</think>")
            content = content[:think_start] + content[think_end:]
            content = content.strip()
        
        # 提取各个部分（基本解析 - 可以用regex改进）
        sections = {
            "description": "信息不可用。",
            "symptoms": "信息不可用。",
            "treatment": "信息不可用。",
            "prevention": "信息不可用。",
            "videos": "没有可用的资源建议。"
        }
        
        # 基本的部分提取 - 在实际应用中，建议使用regex进行更好的解析
        if "描述" in content:
            description_start = content.find("描述")
            next_section = content.find("原因", description_start)
            if next_section > 0:
                sections["description"] = content[description_start:next_section].replace("描述：", "").strip()
        
        if "症状" in content:
            symptoms_start = content.find("症状")
            next_section = content.find("治疗", symptoms_start)
            if next_section > 0:
                sections["symptoms"] = content[symptoms_start:next_section].replace("症状：", "").strip()
        
        if "治疗" in content:
            treatment_start = content.find("治疗")
            next_section = content.find("预防", treatment_start)
            if next_section > 0:
                sections["treatment"] = content[treatment_start:next_section].replace("治疗：", "").strip()
        
        if "预防" in content:
            prevention_start = content.find("预防")
            next_section = content.find("有用资源", prevention_start)
            if next_section > 0:
                sections["prevention"] = content[prevention_start:next_section].replace("预防：", "").strip()
            else:
                sections["prevention"] = content[prevention_start:].replace("预防：", "").strip()
        
        if "有用资源" in content:
            resources_start = content.find("有用资源")
            sections["videos"] = content[resources_start:].replace("有用资源：", "").strip()
        
        return sections
        
    except Exception as e:
        return {
            "description": f"获取信息时出错：{str(e)}",
            "symptoms": "信息不可用。",
            "treatment": "信息不可用。",
            "prevention": "信息不可用。",
            "videos": "没有可用的资源建议。"
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
        print("收到请求")
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
    uvicorn.run(app, host="0.0.0.0", port=8501)
