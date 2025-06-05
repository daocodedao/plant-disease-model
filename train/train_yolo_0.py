# 导入YOLOv8库
from ultralytics import YOLO
import sys, os
import torch  # 导入 torch 库用于检测 GPU 数量
from datetime import datetime
from ultralytics.engine.results import Boxes
from PIL import Image
import cv2
# https://www.kaggle.com/code/naganithinreddy/yolov9

# ====================== 配置参数 ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 预训练模型路径（可选yolov8 n/s/m/l/x.pt）
# Nano（最小）、Small、Medium、Large、Extra Large（最大）

# MODEL_PATH = "yolo11n.pt"
# 数据集配置文件路径
# DATA_PATH = os.path.join(BASE_DIR,"runs/traindata/yolo/yolo_plant_diseases/dataset.yaml")  

MODEL_PATH = "yolo11n-cls.pt"
DATA_PATH = os.path.join(BASE_DIR,"runs/traindata/yolo/yolo_plant_diseases_classify")  

print("DATA_PATH: ", DATA_PATH)


EPOCHS = 100  # 训练轮数
# EPOCHS = 10  # 训练轮数
if sys.platform.lower() == "darwin":
    EPOCHS = 3

BATCH_SIZE = 16  # 批量大小（根据显存调整）
IMAGE_SIZE = 640  # 输入图像尺寸
VAL_FREQ = 10  # 每多少轮进行一次验证
SAVE_BEST_ONLY = True  # 仅保存最优模型
HALF_PRECISION = True  # 启用FP16混合精度


def get_device():
    """自动检测操作系统并返回设备类型"""
    os_name = sys.platform.lower()
    if "linux" in os_name:
        # Ubuntu 系统，检查 GPU 数量
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return "0"  # 1 个 GPU，指定设备编号 0
        elif gpu_count >= 2:
            # 2 个及以上 GPU，返回前两个 GPU 编号
            # 有的机器 2个 GPU 同时使用会死机
            # return [0, 1]  
            return "0" 
        else:
            return "cpu"  # 没有可用 GPU，使用 CPU
    elif "darwin" in os_name:  # MacOS
        return "cpu"
    else:
        return "cpu"  # 其他系统默认使用 CPU


# ====================== 训练函数 ======================
def train_yolov8():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 加载预训练模型
    model = YOLO(MODEL_PATH)
    # print(model.help)  # 显示所有可用参数及其默认值

    # 开始训练
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        # batch=BATCH_SIZE,
        # imgsz=IMAGE_SIZE,
        # val=True,  # 每VAL_FREQ轮进行验证（默认每轮验证）
        # save=not SAVE_BEST_ONLY,  # 是否保存所有检查点（False则仅保存best和last）
        # save_period=10,  # 每多少轮保存一次检查点（与save=True配合使用）
        # half=HALF_PRECISION,  # 启用混合精度
        # device=get_device(),  # 指定训练设备（如"0"或"0,1"多GPU，默认自动选择）
        project="runs/yolov8_train",  # 自定义训练结果保存目录
        # pretrained=True, # 使用预训练权重
        name=f"plant_diseases_{current_time}",  # 训练任务名称
    )

    # 打印训练结果摘要
    print(f"训练完成！最优模型保存路径：{results}")
    # scp -r  -P 10067  fxbox@frp.fxait.com:/data/work/plant-disease-model/yolov8_train/plant_diseases5/weights/last.pt ./model/yolo11n_train.pt 

# ====================== 验证函数 ======================
def validate_model():
    # 加载最优模型
    model = YOLO("model/yolo11n_train.pt")

    # 在验证集上评估
    results = model.val(
        data=DATA_PATH,
        imgsz=IMAGE_SIZE,
        device=get_device(),
        save_json=True,  # 保存验证结果为JSON文件
        save_conf=True,  # 保存预测置信度
    )

    # 打印mAP指标
    print(f"mAP@50: {results.box.map50:.3f}")
    print(f"mAP@50-95: {results.box.map:.3f}")


# ====================== 推理函数 ======================
def predict_image(image_path):
    # 加载最优模型
    model = YOLO("model/yolo11n_train.pt")
    # 推理
    results = model(image_path, conf=0.25)
    detected_objects = []

    # 遍历每个推理结果
    for r in results:
        # r.boxes 包含检测框信息
        # r.names 包含类别名称映射
        # r.boxes.data 包含每个检测框的原始数据 (x1, y1, x2, y2, confidence, class_id)

        # 遍历每个检测到的物体
        for box in r.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]

            detected_objects.append({
                'class_name': class_name,
                'confidence': confidence
            })
    return detected_objects


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 训练模型
    train_yolov8()

    # 2. 验证模型（可选，训练完成后自动验证，可注释此行）
    # validate_model()

    # 3. 推理示例（指定测试图像路径）
    # predict_image("images/apple_FREC_Scab.JPG")
