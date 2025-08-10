import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import json # For saving training history


# --- Define F1-Score Metric ---
# Using tensorflow_addons for F1Score is generally more robust
# However, if tensorflow_addons is not installed, your custom metric is okay.
# For simplicity and broad compatibility, sticking with your custom one but noting the alternative.
# If you can install it: from tensorflow_addons.metrics import F1Score
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

# --- Paths ---
# Use absolute paths or check if the dataset path is correct for your environment.
# Make sure these directories exist and contain your image data.
train_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\skintype\dataset\train'
valid_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\skintype\dataset\valid'
output_dir = 'model_output' # Directory to save plots, logs, and models
os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

# --- Parameters ---
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5 # Increased epochs for initial training
EPOCHS_PHASE2 = 5 # Increased epochs for fine-tuning
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2
NUM_CLASSES = 3 # Ensure this matches your dataset classes

# --- Data Augmentation ---
# Reduced the intensity of some augmentation parameters to prevent over-distortion.
# Added brightness_range to simulate different lighting conditions.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.15, # Reduced from 0.3
    rotation_range=25, # Reduced from 40
    shear_range=0.15, # Reduced from 0.2
    width_shift_range=0.1, # Reduced from 0.15
    height_shift_range=0.1, # Reduced from 0.15
    brightness_range=[0.7, 1.3], # Added brightness augmentation
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Keep shuffle=False for validation/testing to ensure consistent order
)

# --- Class Weights ---
train_labels = train_generator.classes
# Get actual class names for better readability
class_names = list(train_generator.class_indices.keys())
print("Found classes in training data:", class_names)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights_dict = dict(enumerate(class_weights))
print("Computed class weights:", class_weights_dict)

# Print class distribution (helps identify imbalanced classes)
print("\nTraining Class Distribution:")
unique_train_labels, counts_train = np.unique(train_labels, return_counts=True)
for label, count in zip(unique_train_labels, counts_train):
    print(f"  Class '{class_names[label]}': {count} images")

print("\nValidation Class Distribution:")
unique_valid_labels, counts_valid = np.unique(valid_generator.classes, return_counts=True)
for label, count in zip(unique_valid_labels, counts_valid):
    print(f"  Class '{class_names[label]}': {count} images")


# --- Build Model ---
base_model = EfficientNetB3(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                            include_top=False,
                            weights='imagenet')
base_model.trainable = False # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Helps with training stability
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x) # Helps with training stability
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x) # Slightly reduced dropout
x = BatchNormalization()(x) # Helps with training stability
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Use a slightly lower initial learning rate and potentially experiment with more
# aggressive ReduceLROnPlateau settings.
model.compile(optimizer=Adam(learning_rate=5e-5), # Slightly lower initial LR
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_score])

model.summary()

# --- Callbacks ---
checkpoint_path = os.path.join(output_dir, 'best_model_b3.h5')
csv_logger_path = os.path.join(output_dir, 'training_log.csv')

callbacks = [
    # Increased patience to give the model more time to improve
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    # More aggressive learning rate reduction
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7),
    CSVLogger(csv_logger_path, append=True) # Log training metrics to a CSV file
]

print("\nStarting Phase 1: Feature Extraction Training...")
# --- Train (Phase 1: Feature Extraction) ---
history = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=valid_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Load the best model from phase 1 before fine-tuning
model.load_weights(checkpoint_path)

print("\nStarting Phase 2: Fine-Tuning...")
# --- Fine-Tuning ---
base_model.trainable = True # Unfreeze base model

# Fine-tune a specific number of layers from the end
# Instead of //2, let's try unfreezing the last few blocks.
# EfficientNet has blocks of layers. You can inspect model.summary() to see layer names.
# For B3, roughly the last 30-50% of layers might be a good starting point.
# A common strategy is to unfreeze the last few blocks (e.g., 'block6', 'block7', 'block8')
# You need to find the layer names by printing base_model.summary()
fine_tune_from_layer = None
for layer in reversed(base_model.layers):
    # This is a heuristic. Adjust based on inspecting base_model.summary()
    if 'block7' in layer.name or 'block6' in layer.name: # Example blocks to unfreeze
        fine_tune_from_layer = layer.name
        break
if fine_tune_from_layer:
    print(f"Fine-tuning from layer: {fine_tune_from_layer}")
    for layer in base_model.layers:
        if layer.name == fine_tune_from_layer:
            break
        layer.trainable = False
else: # If specific blocks not found, fall back to proportion (adjust as needed)
    fine_tune_at = len(base_model.layers) // 3 # Unfreeze last 2/3 of layers (more aggressive)
    print(f"Fine-tuning {len(base_model.layers) - fine_tune_at} layers (from index {fine_tune_at})")
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False


# Recompile model after unfreezing layers with a very low learning rate
model.compile(optimizer=Adam(learning_rate=1e-6), # Very low learning rate for fine-tuning
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy', f1_score])

# Continue training from where the previous phase left off
history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS, # Total epochs include initial + fine-tuning epochs
    initial_epoch=history.epoch[-1] + 1 if history.epoch else 0, # Start from the next epoch
    validation_data=valid_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# --- Save Model ---
final_model_path = os.path.join(output_dir, "final_skin_type_model_b3.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# =============================
# ✅ EVALUATION + VISUALIZATION
# =============================

# Load the best saved model for final evaluation
# This ensures you evaluate the model that performed best on validation accuracy
print(f"\nLoading best model from {checkpoint_path} for final evaluation...")
best_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'f1_score': f1_score})


# --- Classification Report ---
print("\nGenerating Classification Report...")
# Predict on validation data to get a comprehensive report
# Reset generator to ensure predictions start from the beginning
valid_generator.reset()
Y_pred = best_model.predict(valid_generator, steps=valid_generator.samples // BATCH_SIZE + (1 if valid_generator.samples % BATCH_SIZE else 0))
y_pred = np.argmax(Y_pred, axis=1)
y_true = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

# Ensure all classes are present in y_true, if not, handle with target_names
# This handles cases where a class might be missing in valid_generator.classes but expected in report
unique_true_classes = np.unique(y_true)
target_names_for_report = [class_labels[i] for i in unique_true_classes]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

# --- Confusion Matrix ---
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 7)) # Increased figure size for better readability
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues", cbar=True)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"Confusion Matrix saved to: {cm_path}")

# --- Combine History ---
def combine_history(h1, h2):
    combined = {}
    for key in h1.history.keys():
        combined[key] = h1.history[key]
        if key in h2.history:
            combined[key].extend(h2.history[key])
    return combined

# Check if history_fine exists (might not if early stopping occurred very early)
if 'history_fine' in locals() and history_fine:
    combined_history = combine_history(history, history_fine)
else:
    combined_history = history.history # If only phase 1 ran significantly

# Save combined history for later analysis
history_path = os.path.join(output_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(combined_history, f)
print(f"Training history saved to: {history_path}")

epochs_range = range(len(combined_history['accuracy']))

# --- Plot Accuracy and F1-Score ---
plt.figure(figsize=(16, 6)) # Increased figure size

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, combined_history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(epochs_range, combined_history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# F1 Score Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, combined_history['f1_score'], label='Train F1 Score', color='blue')
plt.plot(epochs_range, combined_history['val_f1_score'], label='Val F1 Score', color='orange')
plt.title('Training & Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plot_path = os.path.join(output_dir, "accuracy_f1_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"Accuracy and F1-Score plots saved to: {plot_path}")

# --- Plot Loss ---
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, combined_history['loss'], label='Train Loss', color='red')
plt.plot(epochs_range, combined_history['val_loss'], label='Val Loss', color='green')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(output_dir, "loss_plot.png")
plt.savefig(loss_plot_path)
plt.show()
print(f"Loss plot saved to: {loss_plot_path}")