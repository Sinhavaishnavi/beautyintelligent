import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Dataset Paths
train_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\pigmentation\dataset\train'
test_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\pigmentation\dataset\test'
valid_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\pigmentation\dataset\valid'

# Image Size and Batch Size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data Generators with Augmentation for Training
train_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data Generators for Validation and Testing (No Augmentation)
test_gen = ImageDataGenerator(rescale=1.0/255.0)
valid_gen = ImageDataGenerator(rescale=1.0/255.0)

# Load Data from Directories
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_data = valid_gen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (clear_skin or pigmented_skin)
])

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10,
    batch_size=BATCH_SIZE
)

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Save the Model
model.save("pigmentation_detector.h5")
print("Model saved as pigmentation_detector.h5")

