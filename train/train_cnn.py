import tensorflow as tf
from tensorflow import keras

from keras import layers, models
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import time
from tqdm import tqdm

import kagglehub

# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
download_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", download_path)
dataset_path = f"{download_path}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# 检查 GPU 是否可用
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
# exit(0)
# https://www.tensorflow.org/guide/gpu
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

tf.debugging.set_log_device_placement(True)


# Data Preparation with Progress Bar
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)
# dataset_path = 'dataset/'
# print("Preparing dataset...")

with tqdm(total=100, desc="Dataset Preparation", unit="%") as pbar:
    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        subset="training",
    )
    time.sleep(1)
    pbar.update(50)

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        subset="validation",
    )
    time.sleep(1)
    pbar.update(50)

# Model Architecture
print("Building model...")
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

best_acc = 0
best_model_path = "model/model.h5"

print("Training model...")
with tqdm(total=10, desc="Training Progress", unit="epoch") as pbar:
    for epoch in range(10):
        history = model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=0)
        current_acc = history.history["val_accuracy"][0]

        if current_acc > best_acc:
            best_acc = current_acc
            model.save(best_model_path)
            print(f"New best model saved with validation accuracy: {best_acc}")

        pbar.update(1)

print("Model Training Complete!")
