# 导入YOLOv8库
from ultralytics import YOLO
import sys
import torch  # 导入 torch 库用于检测 GPU 数量
from datetime import datetime

# https://www.kaggle.com/code/naganithinreddy/yolov9

# ====================== 配置参数 ======================
DATA_PATH = "./traindata/yolo/yolo_plant_diseases/dataset.yaml"  # 数据集配置文件路径
# 预训练模型路径（可选yolov8 n/s/m/l/x.pt）
# Nano（最小）、Small、Medium、Large、Extra Large（最大）
MODEL_PATH = "yolo11n.pt"
EPOCHS = 100  # 训练轮数
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
            return [0, 1]  # 2 个及以上 GPU，返回前两个 GPU 编号
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

    # 开始训练
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        val=True,  # 每VAL_FREQ轮进行验证（默认每轮验证）
        save=not SAVE_BEST_ONLY,  # 是否保存所有检查点（False则仅保存best和last）
        save_period=10,  # 每多少轮保存一次检查点（与save=True配合使用）
        half=HALF_PRECISION,  # 启用混合精度
        device=get_device(),  # 指定训练设备（如"0"或"0,1"多GPU，默认自动选择）
        project="yolov8_train",  # 自定义训练结果保存目录
        name=f"plant_diseases_{current_time}",  # 训练任务名称
    )

    # 打印训练结果摘要
    print(f"训练完成！最优模型保存路径：{results}")


# ====================== 验证函数 ======================
def validate_model():
    # 加载最优模型
    model = YOLO("runs/detect/plant_diseases/weights/best.pt")

    # 在验证集上评估
    results = model.val(
        data=DATA_PATH,
        imgsz=IMAGE_SIZE,
        device=get_device(),
        save_json=True,  # 保存验证结果为JSON文件
        save_conf=True,  # 保存预测置信度
    )

    # 打印mAP指标
    print(f"mAP@50: {results.metrics_map50:.3f}")
    print(f"mAP@50-95: {results.metrics_map:.3f}")


# ====================== 推理函数 ======================
def predict_image(image_path):
    # 加载最优模型
    model = YOLO("runs/detect/plant_diseases/weights/best.pt")

    # 对单张图像进行推理
    results = model.predict(
        source=image_path,
        imgsz=IMAGE_SIZE,
        device=get_device(),
        save=True,  # 保存推理结果图像
        conf=0.5,  # 置信度阈值（0-1）
    )

    # 打印预测结果
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            cls_name = model.names[cls_id]
            print(f"检测到：{cls_name}，置信度：{conf:.2f}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 训练模型
    train_yolov8()

    # 2. 验证模型（可选，训练完成后自动验证，可注释此行）
    # validate_model()

    # 3. 推理示例（指定测试图像路径）
    # predict_image("path/to/test_image.jpg")
