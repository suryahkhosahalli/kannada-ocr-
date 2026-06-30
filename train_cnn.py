# =====================================
# Import libraries
# =====================================
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =====================================
# Load dataset
# =====================================
print("Loading dataset...")
dataset = pd.read_csv("pixel_dataset_rebuilt.csv")

print(f"Dataset shape: {dataset.shape}")

# class column
y = dataset.iloc[:, 0].values

# pixel columns
X = dataset.iloc[:, 1:].values

print(f"Pixel feature shape: {X.shape}")

# Normalize pixels (0–255 → 0–1)
X = X / 255.0

print(f"Pixel range: {X.min():.4f} to {X.max():.4f}")

# =====================================
# Reshape pixels to 32x32 grayscale images
# =====================================
IMG_SIZE = 32
X_reshaped = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Reshaped to: {X_reshaped.shape}")

# =====================================
# Split into train/validation sets
# =====================================
print("\nSplitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(
    X_reshaped, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Convert labels to categorical (0-655 for 656 classes)
num_classes = 656
y_train_cat = keras.utils.to_categorical(y_train - 1, num_classes)
y_val_cat = keras.utils.to_categorical(y_val - 1, num_classes)

print(f"Number of classes: {num_classes}")

# =====================================
# Build CNN Model
# =====================================
print("\nBuilding CNN model...")

model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# =====================================
# Compile model
# =====================================
print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================
# Train model
# =====================================
print("\nTraining model...")
print("-" * 60)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

start_time = time.time()

history = model.fit(
    X_train, y_train_cat,
    batch_size=64,
    epochs=30,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

training_time = time.time() - start_time

print("-" * 60)
print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# =====================================
# Evaluate model
# =====================================
print("\nEvaluating model...")
train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# =====================================
# Generate predictions for confusion matrix
# =====================================
print("\nGenerating predictions...")
y_pred_proba = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1) + 1  # Convert back to 1-656

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

print(f"Confusion matrix shape: {conf_matrix.shape}")

# =====================================
# Save model
# =====================================
model_path = "kannada_cnn_model.h5"
print(f"\nSaving model to {model_path}...")
model.save(model_path)

# Get model size
model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
print(f"Model size: {model_size:.2f} MB")

# =====================================
# Plot training history
# =====================================
print("\nGenerating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
print("Saved training history plot to cnn_training_history.png")

# =====================================
# Plot confusion matrix (subset)
# =====================================
print("Generating confusion matrix plot...")
# Plot subset of confusion matrix (first 50 classes for readability)
subset_size = min(50, num_classes)
conf_subset = conf_matrix[:subset_size, :subset_size]

plt.figure(figsize=(12, 10))
sns.heatmap(conf_subset, annot=False, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix (First {subset_size} Classes)')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved confusion matrix plot to cnn_confusion_matrix.png")

# =====================================
# Generate report
# =====================================
print("\n" + "=" * 80)
print("CNN TRAINING REPORT")
print("=" * 80)
print(f"\nDataset Information:")
print(f"  Total samples: {len(X)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Number of classes: {num_classes}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}x1")

print(f"\nModel Architecture:")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Model size: {model_size:.2f} MB")

print(f"\nTraining Results:")
print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"  Epochs completed: {len(history.history['accuracy'])}")

print(f"\nConfusion Matrix:")
print(f"  Shape: {conf_matrix.shape}")
print(f"  Correct predictions: {np.trace(conf_matrix)}")
print(f"  Total predictions: {len(y_val)}")

print("\n" + "=" * 80)
print("CNN TRAINING COMPLETE")
print("=" * 80)

# Save report to file
with open('cnn_training_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CNN TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Dataset Information:\n")
    f.write(f"  Total samples: {len(X)}\n")
    f.write(f"  Training samples: {len(X_train)}\n")
    f.write(f"  Validation samples: {len(X_val)}\n")
    f.write(f"  Number of classes: {num_classes}\n")
    f.write(f"  Image size: {IMG_SIZE}x{IMG_SIZE}x1\n\n")
    
    f.write("Model Architecture:\n")
    f.write(f"  Total parameters: {model.count_params():,}\n")
    f.write(f"  Model size: {model_size:.2f} MB\n\n")
    
    f.write("Training Results:\n")
    f.write(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
    f.write(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    f.write(f"  Epochs completed: {len(history.history['accuracy'])}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write(f"  Shape: {conf_matrix.shape}\n")
    f.write(f"  Correct predictions: {np.trace(conf_matrix)}\n")
    f.write(f"  Total predictions: {len(y_val)}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("CNN TRAINING COMPLETE\n")
    f.write("=" * 80 + "\n")

print("Saved report to cnn_training_report.txt")
