"""
Flask Web Application for Fake News Detection
Provides:
1. Text input option
2. File/PDF upload option
"""

import os
import sys
import pickle
import PyPDF2
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fake_news_kaggle_model.pkl')

# Global model objects
vectorizer = None
classifier = None
label_map = None

def load_model():
    """Load the trained fake news detection model."""
    global vectorizer, classifier, label_map
    try:
        # Import the detector to use its cleaning methods
        from fake_news_detector import FakeNewsDetector
        
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check what format the model is saved in
        if isinstance(model_data, dict):
            vectorizer = model_data.get('vectorizer')
            classifier = model_data.get('classifier')
            label_map = model_data.get('label_map', {0: 'FAKE', 1: 'REAL'})
        else:
            # Try to get attributes from the object
            detector = model_data
            vectorizer = detector.vectorizer
            classifier = detector.classifier
            label_map = getattr(detector, 'label_map', {0: 'FAKE', 1: 'REAL'})
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def clean_text(text):
    """Clean and preprocess text."""
    import re
    from fake_news_detector import TextCleaner
    
    cleaner = TextCleaner()
    return cleaner.clean_text(text)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def extract_text_from_file(file_path):
    """Extract text from various file types."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        # Try to find text column
        for col in ['text', 'article', 'content', 'news']:
            if col in df.columns:
                return ' '.join(df[col].astype(str).tolist())
        return ' '.join(df.iloc[:, 0].astype(str).tolist())
    else:
        # Try to read as plain text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return None

# Load model at startup
model_loaded = load_model()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    global vectorizer, classifier, label_map
    
    if not model_loaded or vectorizer is None or classifier is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Check if text or file was provided
        text = request.form.get('text', '').strip()
        file = request.files.get('file')
        
        if not text and not file:
            return jsonify({'error': 'Please provide text or upload a file.'}), 400
        
        # Extract text from file if provided
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            extracted_text = extract_text_from_file(file_path)
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            if extracted_text:
                text = extracted_text
            elif not text:
                return jsonify({'error': 'Could not extract text from file.'}), 400
        
        if not text:
            return jsonify({'error': 'No text content to analyze.'}), 400
        
        # Preprocess the text
        cleaned_text = clean_text(text)
        
        # Vectorize and predict
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = classifier.predict(text_tfidf)[0]
        confidence = classifier.predict_proba(text_tfidf)[0]
        
        # Get the label
        label = label_map.get(prediction, 'UNKNOWN')
        confidence_score = max(confidence) * 100
        
        return jsonify({
            'prediction': label,
            'confidence': f"{confidence_score:.2f}%",
            'real_probability': f"{confidence[1]*100:.2f}%" if len(confidence) > 1 else "0%",
            'fake_probability': f"{confidence[0]*100:.2f}%" if len(confidence) > 1 else "0%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

if __name__ == '__main__':
    print("="*50)
    print("FAKE NEWS DETECTION WEB APP")
    print("="*50)
    print(f"Model loaded: {model_loaded}")
    print("Starting Flask server...")
    print("Go to: http://localhost:5000")
    print("="*50)
    app.run(debug=True, port=5000)
