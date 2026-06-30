# =====================================
# Import libraries
# =====================================
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

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
# Split into train/validation sets
# =====================================
print("\nSplitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

num_classes = 656
print(f"Number of classes: {num_classes}")

# =====================================
# Build MLP Model
# =====================================
print("\nBuilding MLP model...")

# Multi-Layer Perceptron with 3 hidden layers
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)

print(f"MLP Architecture: {mlp.hidden_layer_sizes}")
print(f"Total parameters: {mlp.hidden_layer_sizes[0] * 1024 + sum(mlp.hidden_layer_sizes[i] * mlp.hidden_layer_sizes[i+1] for i in range(len(mlp.hidden_layer_sizes)-1)) + mlp.hidden_layer_sizes[-1] * num_classes:,}")

# =====================================
# Train model
# =====================================
print("\nTraining model...")
print("-" * 60)

start_time = time.time()

mlp.fit(X_train, y_train)

training_time = time.time() - start_time

print("-" * 60)
print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Epochs completed: {mlp.n_iter_}")

# =====================================
# Evaluate model
# =====================================
print("\nEvaluating model...")
train_pred = mlp.predict(X_train)
val_pred = mlp.predict(X_val)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# =====================================
# Generate confusion matrix
# =====================================
print("\nGenerating confusion matrix...")
conf_matrix = confusion_matrix(y_val, val_pred)

print(f"Confusion matrix shape: {conf_matrix.shape}")

# =====================================
# Save model
# =====================================
model_path = "kannada_mlp_model.pkl"
print(f"\nSaving model to {model_path}...")
joblib.dump(mlp, model_path)

# Get model size
model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
print(f"Model size: {model_size:.2f} MB")

# =====================================
# Plot training loss curve
# =====================================
print("\nGenerating training plots...")
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('MLP Training Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_training_loss.png', dpi=150, bbox_inches='tight')
print("Saved training loss plot to mlp_training_loss.png")

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
plt.savefig('mlp_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved confusion matrix plot to mlp_confusion_matrix.png")

# =====================================
# Generate report
# =====================================
print("\n" + "=" * 80)
print("MLP TRAINING REPORT")
print("=" * 80)
print(f"\nDataset Information:")
print(f"  Total samples: {len(X)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Number of classes: {num_classes}")
print(f"  Input features: {X.shape[1]} (32x32 pixels flattened)")

print(f"\nModel Architecture:")
print(f"  Hidden layers: {mlp.hidden_layer_sizes}")
print(f"  Activation: {mlp.activation}")
print(f"  Solver: {mlp.solver}")
print(f"  Model size: {model_size:.2f} MB")

print(f"\nTraining Results:")
print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"  Iterations completed: {mlp.n_iter_}")
print(f"  Final loss: {mlp.loss_:.6f}")

print(f"\nConfusion Matrix:")
print(f"  Shape: {conf_matrix.shape}")
print(f"  Correct predictions: {np.trace(conf_matrix)}")
print(f"  Total predictions: {len(y_val)}")

print("\n" + "=" * 80)
print("MLP TRAINING COMPLETE")
print("=" * 80)

# Save report to file
with open('mlp_training_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MLP TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Dataset Information:\n")
    f.write(f"  Total samples: {len(X)}\n")
    f.write(f"  Training samples: {len(X_train)}\n")
    f.write(f"  Validation samples: {len(X_val)}\n")
    f.write(f"  Number of classes: {num_classes}\n")
    f.write(f"  Input features: {X.shape[1]} (32x32 pixels flattened)\n\n")
    
    f.write("Model Architecture:\n")
    f.write(f"  Hidden layers: {mlp.hidden_layer_sizes}\n")
    f.write(f"  Activation: {mlp.activation}\n")
    f.write(f"  Solver: {mlp.solver}\n")
    f.write(f"  Model size: {model_size:.2f} MB\n\n")
    
    f.write("Training Results:\n")
    f.write(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
    f.write(f"  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    f.write(f"  Iterations completed: {mlp.n_iter_}\n")
    f.write(f"  Final loss: {mlp.loss_:.6f}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write(f"  Shape: {conf_matrix.shape}\n")
    f.write(f"  Correct predictions: {np.trace(conf_matrix)}\n")
    f.write(f"  Total predictions: {len(y_val)}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("MLP TRAINING COMPLETE\n")
    f.write("=" * 80 + "\n")

print("Saved report to mlp_training_report.txt")
