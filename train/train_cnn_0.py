"""
此脚本用于从 Kaggle 下载植物病害数据集，构建并训练一个卷积神经网络（CNN）模型，
对植物病害进行分类。训练完成后，会保存模型，记录训练历史，并可视化训练结果和评估指标。
"""
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import kagglehub
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras import utils as keras_utils

# 从 Kaggle Hub 下载植物病害数据集
# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
download_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", download_path)

# 定义数据集路径
dataset_path = f"{download_path}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
# 定义训练集目录
trainDir = os.path.join(dataset_path, "train")

# 开启 TensorFlow 设备放置日志，方便调试时查看运算在哪个设备上执行
# tf.debugging.set_log_device_placement(True)

print("trainDir:", trainDir)
# 从训练集目录加载图像数据集
training_set = keras_utils.image_dataset_from_directory(
    trainDir,
    labels="inferred",  # 从目录结构推断图像标签
    label_mode="categorical",  # 使用独热编码的标签
    class_names=None,  # 自动推断类别名称
    color_mode="rgb",  # 处理 RGB 图像
    batch_size=32,  # 每个批次包含 32 张图像
    image_size=(128, 128),  # 将图像大小调整为 128x128
    shuffle=True,  # 打乱数据集
    seed=None,  # 不设置随机种子
    validation_split=None,  # 不进行数据集划分
    subset=None,  # 不指定子集
    interpolation="bilinear",  # 使用双线性插值调整图像大小
    follow_links=False,  # 不跟随符号链接
    crop_to_aspect_ratio=False,  # 不按纵横比裁剪图像
)

# 定义验证集目录
validDir = os.path.join(dataset_path, "valid")
print("validDir:", validDir)
# 从验证集目录加载图像数据集
validation_set = keras_utils.image_dataset_from_directory(
    validDir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# 创建一个顺序模型
cnn = Sequential()

# 添加卷积层和池化层构建 CNN 模型
# 添加第一个卷积层，输入层接收 128x128 大小的 RGB 图像
cnn.add(
    Conv2D(
        filters=32,  # 使用 32 个卷积核，用于提取图像特征
        kernel_size=3,  # 卷积核大小为 3x3
        padding="same",  # 填充图像，使卷积层输出尺寸与输入尺寸相同
        activation="relu",  # 使用 ReLU 激活函数，引入非线性
        input_shape=[128, 128, 3],  # 输入图像的形状为 128x128x3
    )
)
# 添加第二个卷积层，进一步提取图像特征
cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
# 最大池化层，池化窗口 2x2，步长为 2，用于降低特征图的维度
cnn.add(MaxPool2D(pool_size=2, strides=2))  

# 添加第三组卷积层，卷积核数量增加到 64，增强特征提取能力
cnn.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
# 最大池化层，进一步降低特征图维度
cnn.add(MaxPool2D(pool_size=2, strides=2))

# 添加第四组卷积层，卷积核数量增加到 128
cnn.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
cnn.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
# 最大池化层
cnn.add(MaxPool2D(pool_size=2, strides=2))

# 添加第五组卷积层，卷积核数量增加到 256
cnn.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
cnn.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
# 最大池化层
cnn.add(MaxPool2D(pool_size=2, strides=2))

# 添加第六组卷积层，卷积核数量增加到 512
cnn.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
cnn.add(Conv2D(filters=512, kernel_size=3, activation="relu"))
# 最大池化层
cnn.add(MaxPool2D(pool_size=2, strides=2))

# 添加 Dropout 层防止过拟合
cnn.add(Dropout(0.25))
# 将多维数据展平为一维向量
cnn.add(Flatten())
# 添加全连接层
cnn.add(Dense(units=1500, activation="relu"))
# 再次添加 Dropout 层防止过拟合
cnn.add(Dropout(0.4))

# 输出层，38 个神经元对应 38 个类别，使用 softmax 激活函数
cnn.add(Dense(units=38, activation="softmax"))


from keras.optimizers import Adam
# 编译模型，指定优化器、损失函数和评估指标
cnn.compile(
    optimizer=Adam(learning_rate=0.0001),  # 使用 Adam 优化器，学习率为 0.0001
    loss="categorical_crossentropy",  # 多分类问题使用交叉熵损失函数
    metrics=["accuracy"],  # 评估指标为准确率
)

# 打印模型结构信息
cnn.summary()

# 训练模型，指定训练集、验证集和训练轮数
training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)

# 评估训练集准确率
train_loss, train_acc = cnn.evaluate(training_set)
print("Training accuracy:", train_acc)

# 评估验证集准确率
val_loss, val_acc = cnn.evaluate(validation_set)
print("Validation accuracy:", val_acc)

# 保存训练好的模型
cnn.save("model/trained_plant_disease_model.keras")

# 记录训练历史到 JSON 文件
import json

with open("training_hist.json", "w") as f:
    json.dump(training_history.history, f)

# 打印训练历史的键
print(training_history.history.keys())

# 生成训练轮数列表
epochs = [i for i in range(1, 11)]
# 绘制训练准确率曲线
plt.plot(epochs, training_history.history["accuracy"], color="red", label="Training Accuracy")
# 绘制验证准确率曲线
plt.plot(
    epochs,
    training_history.history["val_accuracy"],
    color="blue",
    label="Validation Accuracy",
)
plt.xlabel("No. of Epochs")  # 设置 x 轴标签
plt.title("Visualization of Accuracy Result")  # 设置图表标题
plt.legend()  # 显示图例
import sys
# 判断系统类型
if sys.platform == 'darwin':  # macOS 系统
    plt.show()  # 显示图表
else:
    accuracy_path = "model/accuracy_plot.png"
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {accuracy_path}")


# 获取验证集的类别名称
class_name = validation_set.class_names
# 从验证集目录加载测试数据集
test_set = keras_utils.image_dataset_from_directory(
    validDir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,  # 每个批次包含 1 张图像
    image_size=(128, 128),
    shuffle=False,  # 不打乱数据集
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# 使用模型对测试集进行预测
y_pred = cnn.predict(test_set)
# 获取预测结果中概率最大的类别索引
predicted_categories = tf.argmax(y_pred, axis=1)

# 获取测试集的真实标签
true_categories = tf.concat([y for x, y in test_set], axis=0)
# 获取真实标签中概率最大的类别索引
Y_true = tf.argmax(true_categories, axis=1)

# 从 sklearn 导入混淆矩阵和分类报告函数
from sklearn.metrics import confusion_matrix, classification_report

# 计算混淆矩阵
cm = confusion_matrix(Y_true, predicted_categories)
# 打印分类报告，包含精确率、召回率和 F1 值
print(classification_report(Y_true, predicted_categories, target_names=class_name))

# 绘制混淆矩阵热力图
plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=True, annot_kws={"size": 10})
plt.xlabel("Predicted Class", fontsize=20)  # 设置 x 轴标签
plt.ylabel("Actual Class", fontsize=20)  # 设置 y 轴标签
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)  # 设置图表标题
# plt.show()  # 显示图表
if sys.platform == 'darwin':  # macOS 系统
    plt.show()  # 显示图表
else:
    confusion_matrix_path = "model/confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {confusion_matrix_path}")

# scp -r  -P 10067 fxbox@frp.fxait.com:/data/work/plant-disease-model/model/trained_plant_disease_model.keras ./model/ 
# scp -r  -P 10067 fxbox@frp.fxait.com:/data/work/plant-disease-model/model/confusion_matrix.png ./model/ 
# scp -r  -P 10067 fxbox@frp.fxait.com:/data/work/plant-disease-model/model/accuracy_plot.png ./model/ 