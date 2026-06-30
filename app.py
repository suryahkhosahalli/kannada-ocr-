import os
import base64
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict_kannada

app = Flask(__name__)

# Configure uploads folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file limit

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample images folder
SAMPLE_IMAGES_FOLDER = os.path.join(BASE_DIR, 'sample images', 'Img_26_100')

# Mapping file
MAPPING_FILE = os.path.join(BASE_DIR, 'class_mapping.csv')

# Auto-create mapping file if it doesn't exist
if not os.path.exists(MAPPING_FILE):
    try:
        pd.DataFrame(columns=['class_id', 'kannada_char']).to_csv(MAPPING_FILE, index=False)
        print(f"Created class_mapping.csv at {MAPPING_FILE}")
    except Exception as e:
        print(f"Warning: Could not create class_mapping.csv: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_base64_of_file(filepath):
    """Converts a local file to a base64 data URI."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode('utf-8')
            ext = filepath.rsplit('.', 1)[1].lower()
            if ext == 'jpg':
                ext = 'jpeg'
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"Error base64 encoding file: {e}")
        return ""

@app.route('/')
def index():
    """Renders the main home page."""
    return render_template('index.html')

@app.route('/class-explorer')
def class_explorer():
    """Renders the class explorer page to browse and map classes."""
    try:
        # Get class_id from query parameter, default to 1
        class_id = request.args.get('class_id', 1, type=int)
        
        # Validate class_id
        if class_id is None or class_id < 1:
            class_id = 1
        elif class_id > 656:
            class_id = 656
        
        # Get sample images for this class (ensure it's always a list)
        sample_images = get_sample_images(class_id)
        if sample_images is None:
            sample_images = []
        
        # Get current mapping for this class (ensure it's always a string)
        current_mapping = get_mapping(class_id)
        if current_mapping is None:
            current_mapping = ''
        
        # Get total mapped count (ensure it's always an integer)
        mapped_count = get_mapped_count()
        if mapped_count is None:
            mapped_count = 0
        
        # Get message parameters (if any)
        message = request.args.get('message', '')
        message_type = request.args.get('message_type', '')
        
        return render_template('class_explorer.html',
                             class_id=class_id,
                             sample_images=sample_images,
                             current_mapping=current_mapping,
                             mapped_count=mapped_count,
                             message=message,
                             message_type=message_type)
    except Exception as e:
        print(f"Error in class_explorer route: {e}")
        # Return a safe fallback
        return render_template('class_explorer.html',
                             class_id=1,
                             sample_images=[],
                             current_mapping='',
                             mapped_count=0,
                             message='An error occurred loading the class explorer.',
                             message_type='error')

@app.route('/save-mapping', methods=['POST'])
def save_mapping():
    """Saves a class-to-Kannada-character mapping."""
    try:
        class_id = request.form.get('class_id', type=int)
        kannada_char = request.form.get('kannada_char', '').strip()
        
        if not class_id or class_id < 1 or class_id > 656:
            return redirect(url_for('class_explorer', class_id=1))
        
        # Load existing mappings
        mappings = load_mappings()
        if mappings is None:
            mappings = {}
        
        # Update or add mapping
        mappings[class_id] = kannada_char
        
        # Save mappings
        save_mappings(mappings)
        
        return redirect(url_for('class_explorer', 
                              class_id=class_id,
                              message='Mapping saved successfully!',
                              message_type='success'))
    except Exception as e:
        print(f"Error saving mapping: {e}")
        return redirect(url_for('class_explorer', 
                              class_id=class_id if 'class_id' in locals() else 1,
                              message='Error saving mapping.',
                              message_type='error'))

def get_sample_images(class_id):
    """Gets up to 20 sample images for a given class."""
    try:
        images = []
        class_prefix = f"img{class_id:03d}"
        
        if os.path.exists(SAMPLE_IMAGES_FOLDER):
            # Get all files matching the class pattern
            files = [f for f in os.listdir(SAMPLE_IMAGES_FOLDER) 
                    if f.startswith(class_prefix) and f.lower().endswith('.png')]
            
            # Sort and take first 20
            files.sort()[:20]
            
            for filename in files[:20]:
                filepath = os.path.join(SAMPLE_IMAGES_FOLDER, filename)
                base64_image = get_base64_of_file(filepath)
                if base64_image:
                    images.append(base64_image)
        
        return images
    except Exception as e:
        print(f"Error getting sample images for class {class_id}: {e}")
        return []

def load_mappings():
    """Loads class mappings from CSV file."""
    mappings = {}
    if os.path.exists(MAPPING_FILE):
        try:
            df = pd.read_csv(MAPPING_FILE)
            # Check if required columns exist
            if 'class_id' in df.columns and 'kannada_char' in df.columns:
                for _, row in df.iterrows():
                    # Handle potential None values
                    class_id = row.get('class_id')
                    kannada_char = row.get('kannada_char', '')
                    if class_id is not None and pd.notna(class_id):
                        mappings[int(class_id)] = str(kannada_char) if pd.notna(kannada_char) else ''
            else:
                print(f"Warning: CSV file missing required columns. Found: {df.columns.tolist()}")
        except pd.errors.EmptyDataError:
            print("Warning: CSV file is empty")
        except Exception as e:
            print(f"Error loading mappings: {e}")
    return mappings

def save_mappings(mappings):
    """Saves class mappings to CSV file."""
    try:
        data = [{'class_id': k, 'kannada_char': v} for k, v in mappings.items()]
        df = pd.DataFrame(data)
        # Ensure columns exist
        if df.empty:
            df = pd.DataFrame(columns=['class_id', 'kannada_char'])
        df.to_csv(MAPPING_FILE, index=False)
    except Exception as e:
        print(f"Error saving mappings: {e}")

def get_mapping(class_id):
    """Gets the Kannada character mapping for a class."""
    mappings = load_mappings()
    return mappings.get(class_id, '')

def get_mapped_count():
    """Gets the total number of mapped classes."""
    mappings = load_mappings()
    return len([v for v in mappings.values() if v.strip()])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles file upload, saves it to uploads folder, runs prediction,
    and returns results either as JSON or by rendering the template.
    """
    if 'file' not in request.files:
        return error_response("No file part in the request", 400)
        
    file = request.files['file']
    
    if file.filename == '':
        return error_response("No file selected", 400)
        
    if not file or not allowed_file(file.filename):
        return error_response("File type not allowed. Please upload an image (PNG, JPG, JPEG, BMP, WEBP, GIF)", 400)
        
    try:
        # Save file to uploads folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get base64 URI of original uploaded image
        original_base64 = get_base64_of_file(filepath)
        
        # Run inference
        results = predict_kannada(filepath)
        
        # Add original image base64 to results
        results['original_img'] = original_base64
        
        # Check if client prefers JSON (AJAX) or HTML
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({
                "success": True,
                "data": results
            })
            
        # Standard form submit fallback - render template with results
        return render_template('index.html', results=results)
        
    except Exception as e:
        print(f"Exception during prediction: {e}")
        return error_response(f"Prediction failed: {str(e)}", 500)

def error_response(message, status_code):
    if request.headers.get('Accept') == 'application/json' or request.is_json:
        return jsonify({
            "success": False,
            "error": message
        }), status_code
    return render_template('index.html', error=message), status_code

if __name__ == '__main__':
    # Start Flask development server
    print("Starting Flask Kannada OCR website on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
