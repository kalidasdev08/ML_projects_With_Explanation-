"""
Dog Breed Analyzer - Flask Web Application
Using the best ML algorithms for dog breed classification
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from dog_breed_classifier import DogBreedClassifier


# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dog-breed-analyzer-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the classifier
print("Initializing Dog Breed Classifier...")
classifier = DogBreedClassifier()
classifier.load_model()
print("Classifier ready!")


def allowed_file(filename):
    """
    Check if the file has an allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """
    Main page
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded dog image
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        
        try:
            # Make prediction
            predictions = classifier.predict(filepath, top_k=5)
            
            if predictions:
                # Get breed info for top prediction
                breed_info = classifier.get_breed_info(predictions[0]['breed'])
                
                return jsonify({
                    'success': True,
                    'image_url': filepath,
                    'predictions': predictions,
                    'breed_info': breed_info
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not classify the image'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """
    Predict breed from URL
    """
    data = request.get_json()
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Download and save image
        import requests
        from PIL import Image
        from io import BytesIO
        
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Could not download image'}), 400
        
        # Save image
        img = Image.open(BytesIO(response.content))
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        img.save(filepath)
        
        # Make prediction
        predictions = classifier.predict(filepath, top_k=5)
        
        if predictions:
            breed_info = classifier.get_breed_info(predictions[0]['breed'])
            
            return jsonify({
                'success': True,
                'image_url': filepath,
                'predictions': predictions,
                'breed_info': breed_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not classify the image'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500


@app.route('/about')
def about():
    """
    About page
    """
    return render_template('index.html', section='about')


@app.route('/api/breeds')
def get_breeds():
    """
    Get list of supported breeds
    """
    return jsonify({
        'breeds': list(classifier.breed_labels.values()),
        'count': len(classifier.breed_labels)
    })


@app.route('/api/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.is_loaded,
        'num_breeds': len(classifier.breed_labels)
    })


if __name__ == '__main__':
    print("="*60)
    print("Dog Breed Analyzer Web Application")
    print("="*60)
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
