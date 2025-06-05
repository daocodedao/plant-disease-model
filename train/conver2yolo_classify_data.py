import os
import csv
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

# ====================== 配置参数 ======================
# 从 Kaggle Hub 下载植物病害数据集
# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
import kagglehub
tf_download_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", tf_download_path)
# 定义数据集路径
tf_dataset_path = f"{tf_download_path}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

INPUT_DATA_DIR = tf_dataset_path  # 输入数据集路径（解压后的根目录）
OUTPUT_YOLO_DIR = "./runs/traindata/yolo/yolo_plant_diseases_classify"        # 输出YOLO数据集路径
if os.path.exists(OUTPUT_YOLO_DIR):
    shutil.rmtree(OUTPUT_YOLO_DIR)
os.makedirs(OUTPUT_YOLO_DIR, exist_ok=True)

TRAIN_SIZE = 0.8                                 # 训练集比例
IMAGE_EXTENSIONS = [".JPG", ".jpg", ".jpeg", ".png"]     # 支持的图像扩展名

# ====================== 类别映射（需根据实际数据集调整） ======================
# 从原数据集的类别名称生成映射（示例：假设病害类别为文件夹名）
def get_class_mapping(data_dir):
    class_names = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith("."):
            class_names.append(folder)
    class_names.sort()  # 按字母序排序，确保类别编号固定
    return {cls: idx for idx, cls in enumerate(class_names)}

# ====================== 划分数据集并保存 ======================
def save_dataset(annotations, class_map, output_dir, train_size=0.8):
    # 划分训练集和验证集
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_size)
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]
    
    # 创建目录结构
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    for cls in class_map.keys():
        os.makedirs(os.path.join(output_dir, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", cls), exist_ok=True)
    
    # 保存训练集
    for data in train_data:
        img_path = data["image_path"]
        cls = data["class_name"]
        try:
            shutil.copy2(img_path, os.path.join(output_dir, "train", cls))
            print(f"图像 {img_path} 复制到训练集 {cls} 类成功")
        except Exception as e:
            print(f"图像 {img_path} 复制到训练集 {cls} 类失败，错误信息: {e}")
    
    # 保存验证集
    for data in val_data:
        img_path = data["image_path"]
        cls = data["class_name"]
        try:
            shutil.copy2(img_path, os.path.join(output_dir, "val", cls))
            print(f"图像 {img_path} 复制到验证集 {cls} 类成功")
        except Exception as e:
            print(f"图像 {img_path} 复制到验证集 {cls} 类失败，错误信息: {e}")
    
    # 生成类别名文件（classes.names）
    with open(os.path.join(output_dir, "classes.names"), "w") as f:
        for cls in class_map.keys():
            f.write(f"{cls}\n")
    
    # 生成数据集配置文件（dataset.yaml）
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")  # 数据集根路径
        f.write(f"train: train\n")  # 训练集路径（相对于path）
        f.write(f"val: val\n")      # 验证集路径
        # f.write(f"test: images/test\n")   # 测试集路径（如果有）
        f.write(f"nc: {len(class_map)}\n")  # 类别数
        # 修改 names 字段输出格式
        class_names = list(class_map.keys())
        f.write(f"names: {class_names}\n")
        
    return train_data, val_data

# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 检查输入路径是否存在
    if not os.path.exists(INPUT_DATA_DIR):
        raise FileNotFoundError(f"请先下载数据集并解压到路径：{INPUT_DATA_DIR}")
    
    # 2. 获取类别映射（假设图像按类别存放在子文件夹中）
    class_map = get_class_mapping(os.path.join(INPUT_DATA_DIR, "train"))  # 假设训练集图像在train子文件夹中，每个子文件夹为一个类别
    
    # 3. 解析标注（仅按文件夹分类）
    annotations = []
    for cls, idx in class_map.items():
        cls_dir = os.path.join(INPUT_DATA_DIR, "train", cls)  # 假设类别文件夹路径为train/类别名
        for img_file in os.listdir(cls_dir):
            if any(img_file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_path = os.path.join(cls_dir, img_file)
                annotations.append({
                    "image_path": img_path,
                    "class_name": cls
                })
    
    # 4. 保存为YOLO格式
    train_data, val_data = save_dataset(annotations, class_map, OUTPUT_YOLO_DIR, train_size=TRAIN_SIZE)
    
    print(f"✅ 转换完成！YOLO数据集已保存至：{OUTPUT_YOLO_DIR}")
    print(f"类别数：{len(class_map)}，训练集样本数：{len(train_data)}，验证集样本数：{len(val_data)}")