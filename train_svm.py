# =====================================
# Import libraries
# =====================================
import numpy as np
import pandas as pd
import joblib
from skimage.feature import hog
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# =====================================
# Load dataset
# =====================================
dataset = pd.read_csv("/content/pixel_dataset.csv")

print("Dataset shape:", dataset.shape)

# class column
y = dataset.iloc[:,0].values

# pixel columns
X = dataset.iloc[:,1:].values

print("Pixel feature shape:", X.shape)

# =====================================
# Normalize pixels (0–255 → 0–1)
# =====================================
X = X / 255.0

print("Pixel range:", X.min(), X.max())

# =====================================
# Convert pixels → HOG features
# =====================================
IMG_SIZE = 32
hog_features = []

print("Extracting HOG features...")

for pixels in X:

    img = pixels.reshape(IMG_SIZE, IMG_SIZE)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2'
    )

    hog_features.append(features)

X_hog = np.array(hog_features)

print("HOG feature shape:", X_hog.shape)

# =====================================
# 5 Fold Cross Validation
# =====================================
kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold = 1
accuracies = []

for train_idx, test_idx in kfold.split(X_hog, y):

    print("\n===== Fold", fold, "=====")

    X_train = X_hog[train_idx]
    X_test = X_hog[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Train SVM
    model = LinearSVC(max_iter=10000)

    print("Training model...")

    model.fit(X_train, y_train)

    print("Training complete")

    # Evaluate
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print("Fold Accuracy:", accuracy)

    accuracies.append(accuracy)

    fold += 1

# =====================================
# Final results
# =====================================
print("\n=================================")
print("Cross Validation Accuracy:", np.mean(accuracies))
print("=================================")

# =====================================
# Train final model on FULL dataset
# =====================================
print("\nTraining final model on full dataset...")

final_model = LinearSVC(max_iter=10000)

final_model.fit(X_hog, y)

joblib.dump(final_model, "hog_svm_model_normalized.pkl")

print("Model saved successfully")