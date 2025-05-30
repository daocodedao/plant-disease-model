import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import kagglehub

# Download latest version
download_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

print("Path to dataset files:", download_path)

import os

dataset_path = f"{download_path}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
trainDir = os.path.join(dataset_path, "train")
print("trainDir:", trainDir)
training_set = tf.keras.utils.image_dataset_from_directory(
    trainDir,
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

validDir = os.path.join(dataset_path, "valid")
print("validDir:", validDir)
validation_set = tf.keras.utils.image_dataset_from_directory(
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

cnn = tf.keras.models.Sequential()

cnn.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=[128, 128, 3],
    )
)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
)
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding="same", activation="relu"
    )
)
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(
    tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, padding="same", activation="relu"
    )
)
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(
    tf.keras.layers.Conv2D(
        filters=512, kernel_size=3, padding="same", activation="relu"
    )
)
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=1500, activation="relu"))

cnn.add(tf.keras.layers.Dropout(0.4))  # To avoid overfitting

# Output Layer
cnn.add(tf.keras.layers.Dense(units=38, activation="softmax"))

from tensorflow.keras.optimizers import Adam

cnn.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cnn.summary()

training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)

# #Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print("Training accuracy:", train_acc)

# #Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print("Validation accuracy:", val_acc)

cnn.save("trained_plant_disease_model.keras")

# training_history.history #Return Dictionary of history

# #Recording History in json
import json

with open("training_hist.json", "w") as f:
    json.dump(training_history.history, f)

print(training_history.history.keys())

epochs = [i for i in range(1, 11)]
plt.plot(
    epochs, training_history.history["accuracy"], color="red", label="Training Accuracy"
)
plt.plot(
    epochs,
    training_history.history["val_accuracy"],
    color="blue",
    label="Validation Accuracy",
)
plt.xlabel("No. of Epochs")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()

class_name = validation_set.class_names
test_set = tf.keras.utils.image_dataset_from_directory(
    validDir,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)

true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

Y_true

predicted_categories

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_true, predicted_categories)
# Precision Recall Fscore
print(classification_report(Y_true, predicted_categories, target_names=class_name))

plt.figure(figsize=(40, 40))
sns.heatmap(cm, annot=True, annot_kws={"size": 10})
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("Actual Class", fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()
