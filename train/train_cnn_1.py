import tensorflow as tf
from tensorflow import keras

from keras import layers, models
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import time
from tqdm import tqdm

import kagglehub

# 从 Kaggle 下载植物病害数据集
# 数据集链接: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
download_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", download_path)
# 定义数据集路径
dataset_path = f"{download_path}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# 打印可用的 GPU 数量
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# 检查 GPU 是否可用
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
# GPU 配置指南: https://www.tensorflow.org/guide/gpu
if gpus:
    # 限制 TensorFlow 仅使用第一块 GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        # 当前，所有 GPU 的内存增长设置需要保持一致
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # 必须在 GPU 初始化之前设置可见设备
        print(e)

# 开启设备放置日志，方便调试 TensorFlow 运算在哪个设备上执行
tf.debugging.set_log_device_placement(True)

# 数据预处理，使用进度条显示处理进度
# 创建图像数据生成器，进行数据增强和归一化操作
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 归一化，将像素值缩放到 0-1 之间
    validation_split=0.2,  # 划分 20% 的数据作为验证集
    rotation_range=20,  # 随机旋转图像的角度范围
    zoom_range=0.2,  # 随机缩放图像的范围
    horizontal_flip=True,  # 随机水平翻转图像
)
# dataset_path = 'dataset/'
# print("Preparing dataset...")

# 使用 tqdm 库创建进度条，显示数据集准备进度
with tqdm(total=100, desc="Dataset Preparation", unit="%") as pbar:
    # 生成训练集数据
    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),  # 将图像大小调整为 150x150
        batch_size=32,  # 每个批次包含 32 张图像
        class_mode="binary",  # 二分类模式
        subset="training",  # 指定为训练集
    )
    time.sleep(1)  # 模拟耗时操作
    pbar.update(50)  # 进度条更新 50%

    # 生成验证集数据
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        subset="validation",  # 指定为验证集
    )
    time.sleep(1)
    pbar.update(50)  # 进度条更新到 100%

# 构建模型
print("Building model...")
# 创建一个顺序模型
model = models.Sequential(
    [
        # 第一层卷积层，32 个卷积核，卷积核大小为 3x3，激活函数为 ReLU
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        # 最大池化层，池化窗口大小为 2x2
        layers.MaxPooling2D((2, 2)),
        # 第二层卷积层，64 个卷积核，卷积核大小为 3x3，激活函数为 ReLU
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        # 第三层卷积层，128 个卷积核，卷积核大小为 3x3，激活函数为 ReLU
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        # 将多维数据展平为一维向量
        layers.Flatten(),
        # 全连接层，256 个神经元，激活函数为 ReLU
        layers.Dense(256, activation="relu"),
        # Dropout 层，防止过拟合，丢弃 50% 的神经元
        layers.Dropout(0.5),
        # 输出层，1 个神经元，激活函数为 Sigmoid，用于二分类
        layers.Dense(1, activation="sigmoid"),
    ]
)

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 记录最佳验证准确率
best_acc = 0

# 定义最佳模型的保存路径
# scp -r  -P 10067 fxbox@frp.fxait.com:/data/work/plant-disease-model/model/model.h5 ./model/ 
best_model_path = "model/trained_plant_disease_model.h5"

# 开始训练模型
print("Training model...")
# 使用 tqdm 库创建进度条，显示训练进度
with tqdm(total=10, desc="Training Progress", unit="epoch") as pbar:
    for epoch in range(10):
        # 训练一个 epoch
        history = model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=0)
        # 获取当前 epoch 的验证准确率
        current_acc = history.history["val_accuracy"][0]

        # 如果当前验证准确率高于之前的最佳准确率，则保存新的最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            model.save(best_model_path)
            print(f"New best model saved with validation accuracy: {best_acc}")

        # 进度条更新一个 epoch
        pbar.update(1)

# 打印模型训练完成信息
print("Model Training Complete!")
