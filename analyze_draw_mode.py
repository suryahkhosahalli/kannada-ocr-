import os
import glob
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set paths
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample images", "Img_26_100")
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = (64, 64)
FINAL_SIZE = 32

def preprocess(img_np):
    """Preprocessing pipeline matching predict.py"""
    # 1. Resize to 64x64
    img_pil = Image.fromarray(img_np).resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_resized = np.array(img_pil)
    
    # 2. Otsu thresholding + inversion
    try:
        thresh = threshold_otsu(img_resized)
        img_bin = (img_resized < thresh).astype(np.uint8) * 255
    except ValueError:
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
    padded_pil = Image.fromarray(padded).resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.BILINEAR)
    return np.array(padded_pil)

def extract_features(img):
    """Extract HOG features"""
    img = img / 255.0
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2'
    )
    return features

def analyze_image(img_np, name):
    """Analyze image properties"""
    results = {
        'name': name,
        'shape': img_np.shape,
        'size': img_np.size,
        'mean': np.mean(img_np),
        'std': np.std(img_np),
        'min': np.min(img_np),
        'max': np.max(img_np),
        'non_zero_ratio': np.count_nonzero(img_np) / img_np.size,
    }
    
    # Calculate stroke thickness estimate
    if len(img_np.shape) == 2:
        # Simple estimate: ratio of edge pixels to total
        from scipy import ndimage
        sobel_x = ndimage.sobel(img_np, axis=0)
        sobel_y = ndimage.sobel(img_np, axis=1)
        edge_ratio = np.count_nonzero(sobel_x + sobel_y) / img_np.size
        results['edge_ratio'] = edge_ratio
    
    return results

def create_canvas_drawing():
    """Create a sample canvas drawing simulating user input"""
    # Create a 250x250 white canvas
    canvas = Image.new('L', (250, 250), 255)
    draw = ImageDraw.Draw(canvas)
    
    # Draw a simple Kannada-like character (similar to 'ka' - ಕ)
    # Using thick strokes as in the web canvas
    draw.line([(80, 80), (170, 80)], fill=0, width=14)  # Top horizontal
    draw.line([(80, 80), (80, 170)], fill=0, width=14)  # Left vertical
    draw.line([(80, 170), (170, 170)], fill=0, width=14)  # Bottom horizontal
    draw.line([(170, 80), (170, 170)], fill=0, width=14)  # Right vertical
    draw.arc([(100, 100), (150, 150)], 0, 360, fill=0, width=14)  # Circle
    
    return np.array(canvas)

def client_side_preprocess_canvas(img_np):
    """Simulate client-side canvas preprocessing from index.html"""
    # Find bounding box of non-white pixels
    coords = np.argwhere(img_np < 250)
    if coords.size == 0:
        return img_np
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop to content
    cropped = img_np[y_min:y_max+1, x_min:x_max+1]
    
    h, w = cropped.shape
    max_dim = max(h, w)
    
    # Scale to 200x200 max, preserving aspect ratio
    scale = 200 / max_dim
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    img_pil = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.BILINEAR)
    cropped = np.array(img_pil)
    
    # Center in 250x250 canvas with 25px padding
    canvas = np.full((250, 250), 255, dtype=np.uint8)
    y_offset = (250 - cropped.shape[0]) // 2
    x_offset = (250 - cropped.shape[1]) // 2
    canvas[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
    
    return canvas

# =====================================
# Main Analysis
# =====================================

print("=" * 60)
print("DRAW MODE CONFIDENCE ANALYSIS")
print("=" * 60)

# 1. Load training images
print("\n1. Loading training images...")
training_files = glob.glob(os.path.join(SAMPLE_DIR, "img001-*.png"))[:5]  # Sample from class 1
training_images = []
for f in training_files:
    img = Image.open(f).convert('L')
    img_np = np.array(img)
    training_images.append(img_np)
print(f"   Loaded {len(training_images)} training images")

# 2. Load upload images (if any)
print("\n2. Loading upload images...")
upload_files = glob.glob(os.path.join(UPLOADS_DIR, "*.png"))[:5]
upload_images = []
for f in upload_files:
    img = Image.open(f).convert('L')
    img_np = np.array(img)
    upload_images.append(img_np)
print(f"   Loaded {len(upload_images)} upload images")

# 3. Create canvas drawings
print("\n3. Creating canvas drawings...")
canvas_images = []
for i in range(5):
    canvas_img = create_canvas_drawing()
    canvas_images.append(canvas_img)
print(f"   Created {len(canvas_images)} canvas drawings")

# 4. Analyze each type
print("\n4. Analyzing image properties...")

results = {
    'training': [],
    'upload': [],
    'canvas': []
}

# Analyze training images
for i, img in enumerate(training_images):
    orig_analysis = analyze_image(img, f"training_{i}")
    processed = preprocess(img)
    proc_analysis = analyze_image(processed, f"training_{i}_processed")
    features = extract_features(processed)
    results['training'].append({
        'original': orig_analysis,
        'processed': proc_analysis,
        'features': features,
        'original_img': img,
        'processed_img': processed
    })

# Analyze upload images
for i, img in enumerate(upload_images):
    orig_analysis = analyze_image(img, f"upload_{i}")
    processed = preprocess(img)
    proc_analysis = analyze_image(processed, f"upload_{i}_processed")
    features = extract_features(processed)
    results['upload'].append({
        'original': orig_analysis,
        'processed': proc_analysis,
        'features': features,
        'original_img': img,
        'processed_img': processed
    })

# Analyze canvas images
for i, img in enumerate(canvas_images):
    orig_analysis = analyze_image(img, f"canvas_{i}")
    # Apply client-side preprocessing first
    client_processed = client_side_preprocess_canvas(img)
    client_analysis = analyze_image(client_processed, f"canvas_{i}_client")
    # Then apply backend preprocessing
    processed = preprocess(client_processed)
    proc_analysis = analyze_image(processed, f"canvas_{i}_processed")
    features = extract_features(processed)
    results['canvas'].append({
        'original': orig_analysis,
        'client_processed': client_analysis,
        'processed': proc_analysis,
        'features': features,
        'original_img': img,
        'client_img': client_processed,
        'processed_img': processed
    })

# 5. Compare HOG features
print("\n5. Comparing HOG features...")

if results['training'] and results['canvas']:
    train_features = np.array([r['features'] for r in results['training']])
    canvas_features = np.array([r['features'] for r in results['canvas']])
    
    # Calculate feature statistics
    train_mean = np.mean(train_features, axis=0)
    canvas_mean = np.mean(canvas_features, axis=0)
    feature_diff = np.abs(train_mean - canvas_mean)
    
    print(f"   Training feature mean: {np.mean(train_features):.4f}")
    print(f"   Canvas feature mean: {np.mean(canvas_features):.4f}")
    print(f"   Feature difference: {np.mean(feature_diff):.4f}")
    print(f"   Max feature difference: {np.max(feature_diff):.4f}")

# 6. Generate visual comparisons
print("\n6. Generating visual comparisons...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Image Comparison: Training vs Canvas', fontsize=16)

# Training images
if results['training']:
    axes[0, 0].imshow(results['training'][0]['original_img'], cmap='gray')
    axes[0, 0].set_title('Training: Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['training'][0]['processed_img'], cmap='gray')
    axes[0, 1].set_title('Training: Processed')
    axes[0, 1].axis('off')
    
    axes[0, 2].hist(results['training'][0]['original_img'].flatten(), bins=50, color='blue', alpha=0.7)
    axes[0, 2].set_title('Training: Pixel Dist')
    axes[0, 2].set_xlabel('Pixel Value')
    axes[0, 2].set_ylabel('Frequency')

# Canvas images
if results['canvas']:
    axes[1, 0].imshow(results['canvas'][0]['original_img'], cmap='gray')
    axes[1, 0].set_title('Canvas: Original')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['canvas'][0]['client_img'], cmap='gray')
    axes[1, 1].set_title('Canvas: Client Preprocessed')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(results['canvas'][0]['processed_img'], cmap='gray')
    axes[1, 2].set_title('Canvas: Backend Processed')
    axes[1, 2].axis('off')
    
    axes[1, 3].hist(results['canvas'][0]['processed_img'].flatten(), bins=50, color='red', alpha=0.7)
    axes[1, 3].set_title('Canvas: Pixel Dist')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')

# Feature comparison
if results['training'] and results['canvas']:
    axes[2, 0].bar(range(len(train_features[0])), train_features[0], alpha=0.5, label='Training')
    axes[2, 0].set_title('Training: HOG Features')
    axes[2, 0].set_xlabel('Feature Index')
    axes[2, 0].set_ylabel('Value')
    
    axes[2, 1].bar(range(len(canvas_features[0])), canvas_features[0], alpha=0.5, label='Canvas', color='red')
    axes[2, 1].set_title('Canvas: HOG Features')
    axes[2, 1].set_xlabel('Feature Index')
    axes[2, 1].set_ylabel('Value')
    
    axes[2, 2].bar(range(len(feature_diff)), feature_diff, alpha=0.5, color='green')
    axes[2, 2].set_title('Feature Difference')
    axes[2, 2].set_xlabel('Feature Index')
    axes[2, 2].set_ylabel('Absolute Difference')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'visual_comparison.png'), dpi=150)
print(f"   Saved visual comparison to {OUTPUT_DIR}/visual_comparison.png")

# 7. Generate detailed statistics report
print("\n7. Generating statistics report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("DRAW MODE CONFIDENCE ANALYSIS REPORT")
report_lines.append("=" * 80)
report_lines.append("")

# Training image statistics
report_lines.append("TRAINING IMAGES STATISTICS")
report_lines.append("-" * 80)
if results['training']:
    for key in ['shape', 'size', 'mean', 'std', 'min', 'max', 'non_zero_ratio', 'edge_ratio']:
        values = [r['processed'][key] for r in results['training'] if key in r['processed']]
        if values:
            report_lines.append(f"{key:20s}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")
report_lines.append("")

# Canvas image statistics
report_lines.append("CANVAS DRAWINGS STATISTICS")
report_lines.append("-" * 80)
if results['canvas']:
    for key in ['shape', 'size', 'mean', 'std', 'min', 'max', 'non_zero_ratio', 'edge_ratio']:
        values = [r['processed'][key] for r in results['canvas'] if key in r['processed']]
        if values:
            report_lines.append(f"{key:20s}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")
report_lines.append("")

# Comparison
report_lines.append("KEY DIFFERENCES")
report_lines.append("-" * 80)
if results['training'] and results['canvas']:
    train_mean = np.mean([r['processed']['mean'] for r in results['training']])
    canvas_mean = np.mean([r['processed']['mean'] for r in results['canvas']])
    train_std = np.mean([r['processed']['std'] for r in results['training']])
    canvas_std = np.mean([r['processed']['std'] for r in results['canvas']])
    
    report_lines.append(f"Pixel Mean Difference: {abs(train_mean - canvas_mean):.4f}")
    report_lines.append(f"Pixel Std Difference: {abs(train_std - canvas_std):.4f}")
    
    train_nz = np.mean([r['processed']['non_zero_ratio'] for r in results['training']])
    canvas_nz = np.mean([r['processed']['non_zero_ratio'] for r in results['canvas']])
    report_lines.append(f"Non-zero Ratio Difference: {abs(train_nz - canvas_nz):.4f}")

report_lines.append("")

# Save report
with open(os.path.join(OUTPUT_DIR, 'analysis_report.txt'), 'w') as f:
    f.write('\n'.join(report_lines))

print(f"   Saved analysis report to {OUTPUT_DIR}/analysis_report.txt")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
