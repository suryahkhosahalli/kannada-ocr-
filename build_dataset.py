import os
import glob
import csv
import numpy as np
from PIL import Image

# Directory containing sample images
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample images", "Img_26_100")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixel_dataset_rebuilt.csv")

# Get all PNG files
image_files = glob.glob(os.path.join(SAMPLE_DIR, "*.png"))
image_files.sort()

print(f"Found {len(image_files)} images in {SAMPLE_DIR}")

# Prepare data storage
data_rows = []
IMG_SIZE = 32

for img_path in image_files:
    # Extract class ID from filename (e.g., img001-001.png -> 1)
    filename = os.path.basename(img_path)
    # Format: imgXXX-YYY.png
    class_id = int(filename[3:6])
    
    # Load image as grayscale
    img = Image.open(img_path).convert('L')
    
    # Resize to 32x32
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    
    # Convert to numpy array and flatten
    img_array = np.array(img_resized)
    pixels = img_array.flatten().tolist()
    
    # Create row: class_id + 1024 pixel values
    row = [class_id] + pixels
    data_rows.append(row)
    
    print(f"Processed {filename}: Class ID {class_id}")

# Sort by class ID to ensure consistent ordering
data_rows.sort(key=lambda x: x[0])

# Write to CSV
print(f"\nWriting to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['class'] + [f'pixel_{i}' for i in range(IMG_SIZE * IMG_SIZE)])
    writer.writerows(data_rows)

print(f"Dataset built successfully!")
print(f"Total rows: {len(data_rows)}")
print(f"Columns: 1 (class) + {IMG_SIZE * IMG_SIZE} (pixels) = {IMG_SIZE * IMG_SIZE + 1}")
print(f"Output file: {OUTPUT_CSV}")
