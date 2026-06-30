# =====================================
# Import libraries
# =====================================
import cv2
import numpy as np
import joblib
import io
import os
import glob
import random
from PIL import Image
from base64 import b64decode
from google.colab import output
from IPython.display import Javascript, display
from skimage.feature import hog
import matplotlib.pyplot as plt

# =====================================
# Load trained model
# =====================================
model = joblib.load("hog_svm_model_normalized.pkl")

print("Model loaded successfully")

# =====================================
# Folder containing dataset images
# =====================================
IMG_BASE = "/content/Sample_Images/Img_26_100"

print("Dataset folder:", IMG_BASE)

TARGET_SIZE = (64,64)

# =====================================
# SAME PREPROCESSING USED IN DATASET
# =====================================
def preprocess(img):

    img_resized = cv2.resize(img, TARGET_SIZE)

    _, img_bin = cv2.threshold(
        img_resized,
        0,255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(img_bin)

    if coords is not None:
        x,y,w,h = cv2.boundingRect(coords)
        img_bin = img_bin[y:y+h, x:x+w]

    h,w = img_bin.shape
    size = max(h,w)

    padded = np.zeros((size,size), dtype=np.uint8)

    y_offset = (size-h)//2
    x_offset = (size-w)//2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img_bin

    return cv2.resize(padded, (32,32))


# =====================================
# FEATURE EXTRACTION (Same as training)
# =====================================
def extract_features(img):

    img = img / 255.0

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2'
    )

    return features


# =====================================
# Prediction function
# =====================================
def predict_character(data):

    data = data.split(',')[1]

    img = Image.open(io.BytesIO(b64decode(data))).convert('L')
    img = np.array(img)

    # preprocessing
    img_processed = preprocess(img)

    plt.figure(figsize=(4,4))
    plt.imshow(img_processed, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")
    plt.show()

    # extract features
    features = extract_features(img_processed)
    features = features.reshape(1,-1)

    # predict
    pred = model.predict(features)
    pred_class = int(pred[0])

    print("Predicted Class:", pred_class)

    # =====================================
    # FIND SAMPLE IMAGE FROM SAME CLASS
    # Example:
    # class 1 -> img001-xxx.png
    # class 2 -> img002-xxx.png
    # =====================================
    class_prefix = f"img{pred_class:03d}"

    matching_files = glob.glob(
        os.path.join(IMG_BASE, f"{class_prefix}-*.png")
    )

    if len(matching_files) == 0:
        print("No matching images found for this class")
        return

    sample_img_path = random.choice(matching_files)

    print("Class Name:", class_prefix)
    print("Sample Image:", os.path.basename(sample_img_path))

    sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)

    if sample_img is not None:
        plt.figure(figsize=(4,4))
        plt.imshow(sample_img, cmap="gray")
        plt.title(f"Example Image from {class_prefix}")
        plt.axis("off")
        plt.show()
    else:
        print("Could not load sample image")


output.register_callback(
    'notebook.predict_character',
    predict_character
)


# =====================================
# Drawing board UI
# =====================================
display(Javascript("""

const canvas=document.createElement('canvas');
canvas.width=250;
canvas.height=250;
canvas.style.border="2px solid black";
canvas.style.touchAction="none";

document.body.appendChild(canvas);

const ctx=canvas.getContext('2d');

ctx.fillStyle="white";
ctx.fillRect(0,0,250,250);

let drawing=false;
let lastX=0;
let lastY=0;

function startDraw(x,y){
 drawing=true;
 lastX=x;
 lastY=y;
}

function drawLine(x,y){

 if(!drawing) return;

 ctx.strokeStyle="black";
 ctx.lineWidth=14;
 ctx.lineCap="round";

 ctx.beginPath();
 ctx.moveTo(lastX,lastY);
 ctx.lineTo(x,y);
 ctx.stroke();

 lastX=x;
 lastY=y;
}

function stopDraw(){
 drawing=false;
}

canvas.addEventListener("mousedown",(e)=>{
 const rect=canvas.getBoundingClientRect();
 startDraw(e.clientX-rect.left,e.clientY-rect.top);
});

canvas.addEventListener("mousemove",(e)=>{
 const rect=canvas.getBoundingClientRect();
 drawLine(e.clientX-rect.left,e.clientY-rect.top);
});

canvas.addEventListener("mouseup",stopDraw);
canvas.addEventListener("mouseleave",stopDraw);

canvas.addEventListener("touchstart",(e)=>{
 const rect=canvas.getBoundingClientRect();
 const touch=e.touches[0];
 startDraw(touch.clientX-rect.left,touch.clientY-rect.top);
 e.preventDefault();
});

canvas.addEventListener("touchmove",(e)=>{
 const rect=canvas.getBoundingClientRect();
 const touch=e.touches[0];
 drawLine(touch.clientX-rect.left,touch.clientY-rect.top);
 e.preventDefault();
});

canvas.addEventListener("touchend",stopDraw);

const predictBtn=document.createElement("button");
predictBtn.innerHTML="Predict";

const clearBtn=document.createElement("button");
clearBtn.innerHTML="Clear";

document.body.appendChild(document.createElement("br"));
document.body.appendChild(predictBtn);
document.body.appendChild(clearBtn);

predictBtn.onclick=()=>{

 const data=canvas.toDataURL("image/png");

 google.colab.kernel.invokeFunction(
   'notebook.predict_character',
   [data],
   {}
 );
}

clearBtn.onclick=()=>{
 ctx.fillStyle="white";
 ctx.fillRect(0,0,250,250);
}

"""))