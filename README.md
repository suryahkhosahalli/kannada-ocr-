# Handwritten Kannada Character Recognition using HOG and Linear SVM

A modern, high-performance web portal for Optical Character Recognition (OCR) of handwritten Kannada characters. This project utilizes Histogram of Oriented Gradients (HOG) for robust feature extraction and a Linear Support Vector Machine (Linear SVM) classifier trained on a comprehensive Kannada handwriting dataset. The application features a premium dark-themed web dashboard where users can either upload scanned character images or sketch characters in real-time.

---

## Project Overview

Kannada is a major Dravidian language spoken by millions of people primarily in the state of Karnataka, India. Recognizing handwritten Kannada characters presents significant challenges due to the complex curves, strokes, and structural similarity between characters. 

This project delivers a complete pipeline to solve this task:
1. **Frontend Input**: Supports image uploading (drag-and-drop or select) or digital canvas drawing (desktop mouse + mobile touch controls).
2. **Client-Side Preprocessing (Draw Mode)**: Crops drawings to their exact bounding box, normalizes size, and center-pads preserving aspect ratio to maximize classifier precision.
3. **Backend Preprocessing**: Standardizes all inputs to grayscale, applies Otsu's binarization, crops bounding boxes, center-pads, and resizes to 32x32 pixels.
4. **Feature Extraction**: Extracts 324-dimensional HOG feature vectors from the preprocessed 32x32 image (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)).
5. **Classification**: Evaluates the features using a trained `LinearSVC` model, calculating decision margins to obtain prediction ranks and confidence scores.
6. **Visualization**: Returns the results instantly via AJAX, rendering the main class, confidence progress bar, top 5 predictions, HOG feature visual, and a matching database training sample.

---

## Features

- **Upload Mode**: Upload images from your local system with instant drag-and-drop support, real-time preview, and upload validation.
- **Draw Mode**: Draw characters directly on a 250x250 HTML5 canvas with responsive mouse/touch drawing tracking.
- **Advanced Canvas Preprocessing**: Client-side canvas preprocessing scans pixels, crops blank borders, scales preserving aspect ratio, and centers characters with 25px uniform padding.
- **HOG Preprocessing Visualization**: Renders the final 32x32 preprocessed image alongside the original input with pixelated scaling so you can see exactly what the model sees.
- **Confidence Scoring & Top 5 Rankings**: Uses softmax normalization over Linear SVM decision boundaries to display predictions with confidence ratings and a visual ranking bar chart of the Top 5 candidate classes.
- **Visual Dataset Comparison**: Automatically queries a random training sample from the same predicted class in the local dataset so you can visually verify classification accuracy.
- **Interactive Glassmorphism UI**: Built with a dark gradient aesthetic, responsive grids, loading state overlays, smooth micro-animations, and error banners.

---

## Technology Stack

- **Web Framework**: Flask (Python 3)
- **Machine Learning**: Scikit-Learn (LinearSVC)
- **Image Processing**: Scikit-Image (HOG extraction), Pillow (PIL)
- **Scientific Computing**: NumPy
- **Frontend Design**: Vanilla CSS (CSS Grid, Flexbox, Custom Variables, Backdrop Filters)
- **Frontend Logic**: Vanilla JavaScript (AJAX Fetch API, HTML5 Canvas API, Drag and Drop API)
- **Typography & Icons**: FontAwesome, Google Fonts (Outfit & Inter)

---

## Folder Structure

```text
d:/kan ocr/
├── app.py                      # Flask server application (routing and uploads)
├── predict.py                  # Backend prediction pipeline & preprocessing
├── requirements.txt            # Python dependencies
├── train_svm.py                # Original model cross-validation & training code
├── predict_colab.py            # Colab drawing interface and prediction testing reference
├── hog_svm_model_normalized.pkl # Pre-trained LinearSVC model (656 classes)
├── MODEL_SPECIFICATION.txt     # Parameters and specifications for the SVM & HOG
├── templates/
│   └── index.html              # Single-page premium dashboard template
├── uploads/                    # Stores uploaded user files (created automatically)
└── sample images/              # Local training sample dataset
    └── Img_26_100/             # Images containing class prefixes (img001-img656)
```

---

## Installation Instructions

1. **Prerequisites**: Ensure you have Python (version 3.8 to 3.12 recommended) installed on your system.
2. **Clone/Open Project Directory**:
   ```bash
   cd "d:\kan ocr"
   ```
3. **Install Dependencies**:
   Install all python packages defined in requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. **Start the Flask Application**:
   Execute the server script:
   ```bash
   python app.py
   ```
2. **Access the Portal**:
   Once started, open your web browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```

---

## Usage Guide

### Upload Mode
1. Ensure the **Upload Image** tab is active.
2. Drag and drop an image of a handwritten Kannada character into the dashed area, or click to browse files.
3. Verify the upload preview appears.
4. Click **Analyze Character**.
5. The portal will process and display the classification, confidence score, Top 5 predictions, and a database sample. Click the `X` button on the preview to clear and upload a new image.

### Draw Mode
1. Click the **Draw Character** tab.
2. Draw a character inside the white 250x250 square canvas. Draw thick, clear strokes for better accuracy.
3. Click **Analyze Character** to view predictions.
4. Click **Clear Canvas** to wipe the canvas and start a new sketch.

---

## Screenshots

*Below are placeholders for screenshots of the dashboard interface:*

#### 1. Home Dashboard (Upload Mode)
`[Placeholder: Screenshot showing the drag-and-drop area, premium glassmorphism card layouts, and header elements.]`

#### 2. Canvas Drawing Interface (Draw Mode)
`[Placeholder: Screenshot showing the 250x250 digital white canvas with touch-support and character drawn.]`

#### 3. Prediction Results & Data Visualizations
`[Placeholder: Screenshot showing the predicted class ID, circular confidence meter, preprocessed image comparison, and the Top 5 predictions horizontal bar chart.]`

---

## Future Enhancements

1. **Real-Time Feature Visualizer**: Render the extracted HOG feature orientations directly on the UI as a grid overlay.
2. **Batch File OCR**: Support uploading a PDF or a larger image containing multiple characters, segmenting them, and predicting them in sequence.
3. **Model Fine-Tuning**: Extend classification with a Deep Convolutional Neural Network (CNN) such as ResNet or MobileNet to compare accuracy against the HOG+SVM baseline.
4. **Speech Synthesis**: Add text-to-speech support to vocalize the Kannada character name or class when classified.
