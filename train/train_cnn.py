import tensorflow as tf
from tensorflow import keras

from keras import layers, models
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import time
from tqdm import tqdm

# Download Dataset using Kaggle API
os.environ['KAGGLE_CONFIG_DIR'] = './secrets/.kaggle'
print("Downloading dataset...")
import kaggle
kaggle.api.dataset_download_files('vipoooool/new-plant-diseases-dataset', path='dataset', unzip=True)
print("Dataset Downloaded Successfully!")

# Data Preparation with Progress Bar
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
dataset_path = 'dataset/'
print("Preparing dataset...")

with tqdm(total=100, desc="Dataset Preparation", unit="%") as pbar:
    train_gen = datagen.flow_from_directory(dataset_path, target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')
    time.sleep(1)
    pbar.update(50)
    val_gen = datagen.flow_from_directory(dataset_path, target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')
    time.sleep(1)
    pbar.update(50)

# Model Architecture
print("Building model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

best_acc = 0
best_model_path = 'model/model.h5'

print("Training model...")
with tqdm(total=10, desc="Training Progress", unit="epoch") as pbar:
    for epoch in range(10):
        history = model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=0)
        current_acc = history.history['val_accuracy'][0]

        if current_acc > best_acc:
            best_acc = current_acc
            model.save(best_model_path)
            print(f"New best model saved with validation accuracy: {best_acc}")
        
        pbar.update(1)

print("Model Training Complete!")