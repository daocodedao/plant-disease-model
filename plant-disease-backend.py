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
# API 配置
# -------------------------------
# 创建 OpenAI 客户端实例，设置自定义 API 地址和 API 密钥
client = OpenAI(
    base_url="http://39.105.194.16:6691/v1/",  # 设置自定义API地址
    api_key="YOUR_API_KEY"  # 替换为您的API密钥
)

# -------------------------------------
# FastAPI 应用设置
# -------------------------------------
# 创建 FastAPI 应用实例，设置标题、描述和版本
app = FastAPI(
    title="Plant Disease Recognition API",
    description="Backend API for the Plant Disease Recognition System",
    version="1.0.0"
)

# 添加 CORS 中间件，允许来自前端的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中，替换为特定的来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# 数据模型
# -------------------------------------
class ImageRequest(BaseModel):
    """
    图像请求数据模型，用于接收前端发送的图像信息。
    Attributes:
        image (str): 经过 Base64 编码的图像数据。
        filename (str): 图像文件的名称。
    """
    image: str  # Base64 编码的图像
    filename: str

class PredictionResponse(BaseModel):
    """
    预测响应数据模型，用于返回植物病害预测结果和相关信息。
    Attributes:
        disease_name (str): 预测的病害名称。
        confidence (float): 预测的置信度。
        description (Optional[str]): 病害描述，可选参数。
        symptoms (Optional[str]): 病害症状，可选参数。
        treatment (Optional[str]): 治疗建议，可选参数。
        prevention (Optional[str]): 预防措施，可选参数。
        videos (Optional[str]): 相关视频建议，可选参数。
    """
    disease_name: str
    confidence: float
    description: Optional[str] = None
    symptoms: Optional[str] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    videos: Optional[str] = None

# -------------------------------------
# 模型加载
# -------------------------------------
# 全局变量，用于存储加载的模型
model = None

def load_model():
    """
    加载植物病害识别模型。
    如果模型尚未加载，则从文件系统加载模型；如果已经加载，则直接返回。
    Returns:
        tf.keras.Model: 加载好的 TensorFlow 模型。
    """
    global model
    if model is None:
        # 原始的模型
        # model = tf.keras.models.load_model("new_trained_plant_disease_model.keras")
        # 自己训练的模型
        model = tf.keras.models.load_model("model/trained_plant_disease_model.keras")
    return model

# -------------------------------------
# 病害类别
# -------------------------------------
# 定义所有可能的植物病害类别
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
# 图像预处理和预测
# -------------------------------------
def preprocess_image(image_data):
    """
    处理 Base64 编码的图像，为模型预测做准备。
    Args:
        image_data (str): Base64 编码的图像数据。
    Returns:
        np.ndarray: 处理后的图像数组，可直接输入到模型中进行预测。
    Raises:
        HTTPException: 如果图像处理过程中出现错误，抛出 400 状态码的异常。
    """
    try:
        # 解码 Base64 图像
        decoded_image = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_image))
        
        # 调整图像大小并归一化
        image = image.resize((128, 128))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # 添加批次维度
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def predict_disease(image_array):
    """Make prediction using the model"""
    try:
        model = load_model()
        # 使用加载好的模型对预处理后的图像数组进行预测，得到每个类别的预测概率
        predictions = model.predict(image_array)
        # 找出预测概率数组中概率最大的元素的索引，该索引对应预测的病害类别
        predicted_index = np.argmax(predictions, axis=1)[0]
        # 获取预测概率数组中对应预测类别的概率值，并将其转换为百分比形式
        confidence = float(predictions[0][predicted_index] * 100)
        
        return predicted_index, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------------
# 从 OpenAI 获取病害信息
# -------------------------------------
def get_disease_info(disease_name):
    """
    使用 OpenAI 获取详细的植物病害信息。
    Args:
        disease_name (str): 病害名称。
    Returns:
        Dict[str, str]: 包含病害描述、症状、治疗、预防和相关视频建议的字典。
    """
    try:
        # 清理病害名称以获得更好的提示格式
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

        # 使用新的 API 格式创建聊天完成
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
        
        # 提取各个部分（基本解析 - 可以用 regex 改进）
        sections = {
            "description": "信息不可用。",
            "symptoms": "信息不可用。",
            "treatment": "信息不可用。",
            "prevention": "信息不可用。",
            "videos": "没有可用的资源建议。"
        }
        
        # 基本的部分提取 - 在实际应用中，建议使用 regex 进行更好的解析
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
# API 端点
# -------------------------------------
@app.get("/")
async def root():
    """
    根端点，返回 API 的基本信息。
    Returns:
        Dict[str, str]: 包含 API 消息、状态和版本的字典。
    """
    print("收到请求")
    return {
        "message": "Plant Disease Recognition API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    """
    处理图像并返回植物病害预测结果和相关信息。
    Args:
        request (ImageRequest): 包含 Base64 编码图像和文件名的请求数据。
    Returns:
        PredictionResponse: 包含病害名称、置信度和相关信息的响应数据。
    Raises:
        HTTPException: 如果处理请求过程中出现错误，抛出 500 状态码的异常。
    """
    try:
        print("收到请求")
        # 处理图像
        image_array = preprocess_image(request.image)
        
        # 进行预测
        predicted_index, confidence = predict_disease(image_array)
        disease_name = class_names[predicted_index]
        
        # 为非健康植物获取病害信息
        disease_info = {}
        if "healthy" not in disease_name.lower():
            disease_info = get_disease_info(disease_name)
        
        # 准备响应
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
    """
    返回所有可能的植物病害类别。
    Returns:
        Dict[str, List[str]]: 包含所有病害类别的字典。
    """
    return {"classes": class_names}

# -------------------------------------
# 服务器启动
# -------------------------------------
if __name__ == "__main__":
    # 在启动时加载模型
    load_model()
    # 运行服务器
    uvicorn.run(app, host="0.0.0.0", port=8503)
