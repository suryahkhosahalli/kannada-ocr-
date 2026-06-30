import os
import io
import glob
import random
import base64
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.filters import threshold_otsu

# Load model relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hog_svm_model_normalized.pkl")

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

TARGET_SIZE = (64, 64)

def preprocess(img_np):
    """
    Preprocessing pipeline matching predict_colab.py:
    - Resizes to 64x64
    - Applies Otsu thresholding with inversion (to get white foreground)
    - Crops bounding box of the non-zero pixels
    - Centers in a square canvas
    - Resizes to 32x32
    """
    # 1. Resize to 64x64
    img_pil = Image.fromarray(img_np).resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_resized = np.array(img_pil)
    
    # 2. Otsu thresholding + inversion
    try:
        thresh = threshold_otsu(img_resized)
        img_bin = (img_resized < thresh).astype(np.uint8) * 255
    except ValueError:
        # Fallback if image is uniform
        img_bin = (img_resized < 127).astype(np.uint8) * 255

    # 3. Crop bounding box of non-zero pixels
    coords = np.argwhere(img_bin > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img_bin = img_bin[y_min:y_max+1, x_min:x_max+1]
        
    # 4. Center in a square canvas
    h, w = img_bin.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img_bin
    
    # 5. Resize to 32x32
    padded_pil = Image.fromarray(padded).resize((32, 32), Image.Resampling.BILINEAR)
    return np.array(padded_pil)

def extract_features(img):
    """
    Feature extraction matching model parameters:
    - Normalizes pixels to 0-1
    - Computes HOG features
    """
    img = img / 255.0
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2'
    )
    return features

def img_to_base64(img_np):
    """Converts a numpy array image to base64 data URI."""
    pil_img = Image.fromarray(img_np)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def softmax(x):
    """Computes softmax values for decision function scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_kannada(image_path):
    """
    Runs prediction pipeline on an input image.
    Returns a dictionary containing predicted class, confidence, top 5 predictions,
    and base64 images.
    """
    # 1. Load image in grayscale
    img_pil = Image.open(image_path).convert('L')
    img_np = np.array(img_pil)
    
    # 2. Preprocess and extract features
    img_processed = preprocess(img_np)
    features = extract_features(img_processed).reshape(1, -1)
    
    # 3. Predict class using model
    pred_class = int(model.predict(features)[0])
    
    # 4. Compute probabilities over all classes via decision function
    scores = model.decision_function(features)[0]
    probs = softmax(scores)
    
    # Find predicted class probability
    pred_class_idx = np.where(model.classes_ == pred_class)[0]
    if len(pred_class_idx) > 0:
        confidence = float(probs[pred_class_idx[0]])
    else:
        confidence = 0.0
        
    # 5. Extract top 5 predictions
    top5_indices = np.argsort(scores)[::-1][:5]
    top5 = []
    for idx in top5_indices:
        top5.append({
            "class": int(model.classes_[idx]),
            "score": float(scores[idx]),
            "probability": float(probs[idx])
        })
        
    # 6. Generate base64 representations
    processed_base64 = img_to_base64(img_processed)
    
    # 7. Find a random sample image from predicted class for visual comparison
    class_prefix = f"img{pred_class:03d}"
    sample_images_dir = os.path.join(BASE_DIR, "sample images", "Img_26_100")
    matching_files = glob.glob(os.path.join(sample_images_dir, f"{class_prefix}-*.png"))
    
    sample_base64 = None
    sample_name = None
    if matching_files:
        sample_path = random.choice(matching_files)
        sample_name = os.path.basename(sample_path)
        try:
            sample_pil = Image.open(sample_path).convert('L')
            sample_np = np.array(sample_pil)
            sample_base64 = img_to_base64(sample_np)
        except Exception as e:
            print(f"Error loading sample image {sample_path}: {e}")
            
    return {
        "class": pred_class,
        "confidence": confidence,
        "top5": top5,
        "processed_img": processed_base64,
        "sample_img": sample_base64,
        "sample_name": sample_name
    }
