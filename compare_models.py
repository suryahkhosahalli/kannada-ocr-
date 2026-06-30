# =====================================
# Import libraries
# =====================================
import numpy as np
import pandas as pd
import time
import os
import joblib
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =====================================
# Load dataset for comparison
# =====================================
print("Loading dataset...")
dataset = pd.read_csv("pixel_dataset_rebuilt.csv")

y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:].values

# Normalize pixels
X = X / 255.0

print(f"Dataset shape: {X.shape}")

# Use a subset for faster comparison
subset_size = 1000
X_subset = X[:subset_size]
y_subset = y[:subset_size]

print(f"Using subset of {subset_size} samples for comparison")

# =====================================
# Load HOG + LinearSVC Model
# =====================================
print("\n" + "=" * 80)
print("HOG + LinearSVC Model")
print("=" * 80)

svm_model_path = "hog_svm_model_normalized.pkl"
print(f"Loading SVM model from {svm_model_path}...")
svm_model = joblib.load(svm_model_path)

# Get SVM model size
svm_size = os.path.getsize(svm_model_path) / (1024 * 1024)  # MB
print(f"SVM Model size: {svm_size:.2f} MB")

# Extract HOG features for subset
print("Extracting HOG features...")
IMG_SIZE = 32
hog_features = []

for pixels in X_subset:
    img = pixels.reshape(IMG_SIZE, IMG_SIZE)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2'
    )
    hog_features.append(features)

X_hog = np.array(hog_features)
print(f"HOG features shape: {X_hog.shape}")

# Measure SVM inference time
print("Measuring SVM inference time...")
svm_inference_times = []
for _ in range(10):
    start = time.time()
    svm_pred = svm_model.predict(X_hog)
    svm_inference_times.append(time.time() - start)

svm_avg_inference_time = np.mean(svm_inference_times) * 1000  # ms
print(f"SVM average inference time: {svm_avg_inference_time:.2f} ms")

# Calculate SVM accuracy
svm_accuracy = np.mean(svm_pred == y_subset)
print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")

# =====================================
# Load MLP Model
# =====================================
print("\n" + "=" * 80)
print("MLP Model")
print("=" * 80)

mlp_model_path = "kannada_mlp_model.pkl"

if os.path.exists(mlp_model_path):
    print(f"Loading MLP model from {mlp_model_path}...")
    mlp_model = joblib.load(mlp_model_path)
    
    # Get MLP model size
    mlp_size = os.path.getsize(mlp_model_path) / (1024 * 1024)  # MB
    print(f"MLP Model size: {mlp_size:.2f} MB")
    
    # Measure MLP inference time
    print("Measuring MLP inference time...")
    mlp_inference_times = []
    for _ in range(10):
        start = time.time()
        mlp_pred = mlp_model.predict(X_subset)
        mlp_inference_times.append(time.time() - start)
    
    mlp_avg_inference_time = np.mean(mlp_inference_times) * 1000  # ms
    print(f"MLP average inference time: {mlp_avg_inference_time:.2f} ms")
    
    # Calculate MLP accuracy
    mlp_accuracy = accuracy_score(y_subset, mlp_pred)
    print(f"MLP Accuracy: {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)")
    
    # Get MLP parameters
    mlp_params = sum(mlp_model.coefs_[i].size + mlp_model.intercepts_[i].size for i in range(len(mlp_model.coefs_)))
    print(f"MLP Total parameters: {mlp_params:,}")
    
else:
    print(f"MLP model not found at {mlp_model_path}")
    print("Please run train_mlp.py first to generate the MLP model.")
    mlp_size = 0
    mlp_avg_inference_time = 0
    mlp_accuracy = 0
    mlp_params = 0

# =====================================
# Comparison Report
# =====================================
print("\n" + "=" * 80)
print("MODEL COMPARISON REPORT")
print("=" * 80)

print("\n" + "-" * 80)
print("ACCURACY COMPARISON")
print("-" * 80)
print(f"HOG + LinearSVC:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
if os.path.exists(mlp_model_path):
    print(f"MLP:              {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)")
    acc_diff = mlp_accuracy - svm_accuracy
    print(f"Difference:      {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")

print("\n" + "-" * 80)
print("INFERENCE TIME COMPARISON (per 1000 samples)")
print("-" * 80)
print(f"HOG + LinearSVC:  {svm_avg_inference_time:.2f} ms")
if os.path.exists(mlp_model_path):
    print(f"MLP:              {mlp_avg_inference_time:.2f} ms")
    time_diff = mlp_avg_inference_time - svm_avg_inference_time
    print(f"Difference:      {time_diff:+.2f} ms")
    if time_diff > 0:
        print(f"MLP is {time_diff/svm_avg_inference_time:.2f}x slower")
    else:
        print(f"MLP is {abs(time_diff)/mlp_avg_inference_time:.2f}x faster")

print("\n" + "-" * 80)
print("MODEL SIZE COMPARISON")
print("-" * 80)
print(f"HOG + LinearSVC:  {svm_size:.2f} MB")
if os.path.exists(mlp_model_path):
    print(f"MLP:              {mlp_size:.2f} MB")
    size_diff = mlp_size - svm_size
    print(f"Difference:      {size_diff:+.2f} MB")
    print(f"MLP is {mlp_size/svm_size:.2f}x larger")

print("\n" + "-" * 80)
print("MODEL COMPLEXITY")
print("-" * 80)
print(f"HOG + LinearSVC:  Linear classifier with 656 classes")
if os.path.exists(mlp_model_path):
    print(f"MLP:              {mlp_params:,} parameters")

print("\n" + "-" * 80)
print("TRAINING TIME (from reports)")
print("-" * 80)
print(f"HOG + LinearSVC:  ~5-10 minutes (5-fold CV)")
if os.path.exists(mlp_model_path):
    # Try to read training time from report
    if os.path.exists('mlp_training_report.txt'):
        with open('mlp_training_report.txt', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Training Time:' in line:
                    print(f"MLP:              {line.split(':')[1].strip()}")
                    break
    else:
        print(f"MLP:              (see mlp_training_report.txt)")

print("\n" + "=" * 80)
print("REVIEW")
print("=" * 80)

if os.path.exists(mlp_model_path):
    if mlp_accuracy > svm_accuracy:
        print("✓ MLP achieves higher accuracy")
    elif svm_accuracy > mlp_accuracy:
        print("✓ HOG+SVM achieves higher accuracy")
    else:
        print("= Both models achieve similar accuracy")
    
    if svm_avg_inference_time < mlp_avg_inference_time:
        print("✓ HOG+SVM has faster inference")
    else:
        print("✓ MLP has faster inference")
    
    if svm_size < mlp_size:
        print("✓ HOG+SVM has smaller model size")
    else:
        print("✓ MLP has smaller model size")
else:
    print("MLP model not available for comparison")

print("\n" + "=" * 80)

# =====================================
# Generate comparison plot
# =====================================
if os.path.exists(mlp_model_path):
    print("\nGenerating comparison plot...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    models = ['HOG+SVM', 'MLP']
    accuracies = [svm_accuracy * 100, mlp_accuracy * 100]
    colors = ['#3b82f6', '#10b981']
    
    bars = ax1.bar(models, accuracies, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim([min(accuracies) * 0.95, 100])
    ax1.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Inference time comparison
    times = [svm_avg_inference_time, mlp_avg_inference_time]
    bars = ax2.bar(models, times, color=colors)
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Inference Time Comparison')
    ax2.grid(True, alpha=0.3)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    # Model size comparison
    sizes = [svm_size, mlp_size]
    bars = ax3.bar(models, sizes, color=colors)
    ax3.set_ylabel('Model Size (MB)')
    ax3.set_title('Model Size Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, s in zip(bars, sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{s:.2f} MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to model_comparison.png")

# Save report to file
with open('model_comparison_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MODEL COMPARISON REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ACCURACY COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(f"HOG + LinearSVC:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)\n")
    if os.path.exists(mlp_model_path):
        f.write(f"MLP:              {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)\n")
        acc_diff = mlp_accuracy - svm_accuracy
        f.write(f"Difference:      {acc_diff:+.4f} ({acc_diff*100:+.2f}%)\n\n")
    
    f.write("INFERENCE TIME COMPARISON (per 1000 samples)\n")
    f.write("-" * 80 + "\n")
    f.write(f"HOG + LinearSVC:  {svm_avg_inference_time:.2f} ms\n")
    if os.path.exists(mlp_model_path):
        f.write(f"MLP:              {mlp_avg_inference_time:.2f} ms\n")
        time_diff = mlp_avg_inference_time - svm_avg_inference_time
        f.write(f"Difference:      {time_diff:+.2f} ms\n\n")
    
    f.write("MODEL SIZE COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(f"HOG + LinearSVC:  {svm_size:.2f} MB\n")
    if os.path.exists(mlp_model_path):
        f.write(f"MLP:              {mlp_size:.2f} MB\n")
        size_diff = mlp_size - svm_size
        f.write(f"Difference:      {size_diff:+.2f} MB\n")
        f.write(f"MLP is {mlp_size/svm_size:.2f}x larger\n\n")
    
    f.write("MODEL COMPLEXITY\n")
    f.write("-" * 80 + "\n")
    f.write(f"HOG + LinearSVC:  Linear classifier with 656 classes\n")
    if os.path.exists(mlp_model_path):
        f.write(f"MLP:              {mlp_params:,} parameters\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("COMPARISON COMPLETE\n")
    f.write("=" * 80 + "\n")

print("Saved report to model_comparison_report.txt")
print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)
