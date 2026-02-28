"""
Handwritten Digit Recognizer - Flask Web Application
A web-based digit recognition application using a CNN model.
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from digit_recognizer import DigitRecognizer, load_model_check

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the digit recognizer
print("Initializing digit recognizer...")
recognizer = None

def get_recognizer():
    """Get or initialize the recognizer."""
    global recognizer
    if recognizer is None:
        # Check multiple possible model locations
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible locations
        possible_paths = [
            os.path.join(base_dir, 'models', 'digit_model.h5'),
            os.path.join(base_dir, 'Handwritten Digit Recognizer', 'models', 'digit_model.h5'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {path}")
                break
        
        if model_path is None:
            print(f"Model not found. Searched in: {possible_paths}")
            return None
        
        print(f"Loading model from: {model_path}")
        recognizer = load_model_check(model_path)
    return recognizer


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    recognizer = get_recognizer()
    
    if recognizer is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    try:
        # Check if image is uploaded or sent as canvas data
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = recognizer.predict(filepath)
            result['image_path'] = f'/static/uploads/{filename}'
            
            return jsonify(result)
        
        elif 'canvas_data' in request.json:
            # Canvas drawing data
            canvas_data = request.json['canvas_data']
            result = recognizer.predict(canvas_data)
            return jsonify(result)
        
        else:
            return jsonify({'error': 'No image data provided'}), 400
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_top_k', methods=['POST'])
def predict_top_k():
    """Handle top-k prediction requests."""
    recognizer = get_recognizer()
    
    if recognizer is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    try:
        k = request.json.get('k', 3)
        
        if 'image' in request.files:
            file = request.files['image']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            
            result = recognizer.predict_top_k(filepath, k)
            return jsonify({'top_predictions': result})
        
        elif 'canvas_data' in request.json:
            canvas_data = request.json['canvas_data']
            result = recognizer.predict_top_k(canvas_data, k)
            return jsonify({'top_predictions': result})
        
        else:
            return jsonify({'error': 'No image data provided'}), 400
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    recognizer = get_recognizer()
    return jsonify({
        'status': 'healthy' if recognizer else 'model_not_loaded',
        'model_loaded': recognizer is not None
    })


if __name__ == '__main__':
    print("Starting Handwritten Digit Recognizer...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
