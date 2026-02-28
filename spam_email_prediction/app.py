"""
Spam Email Detector - Web Application
=======================================
A Flask web application for detecting spam emails using
CountVectorizer and Naive Bayes classifier.

This application provides a user-friendly interface to classify
emails as spam or not spam (ham).
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'models')

# Load the trained model and vectorizer
def load_models():
    """Load the trained classifier and vectorizer"""
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    classifier_path = os.path.join(MODEL_DIR, 'spam_classifier.pkl')
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    
    return classifier, vectorizer

# Try to load models at startup
try:
    classifier, vectorizer = load_models()
    MODEL_LOADED = True
    print("Models loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    classifier = None
    vectorizer = None
    print(f"Warning: Could not load models - {e}")
    print("Please run train_model.py first to train the model.")

def predict_spam(email_text):
    """
    Predict whether an email is spam or ham
    
    Parameters:
    -----------
    email_text : str
        The email content to classify
        
    Returns:
    --------
    dict : Contains prediction, confidence scores, and status
    """
    if not MODEL_LOADED:
        return {
            'status': 'error',
            'message': 'Model not loaded. Please train the model first.',
            'prediction': None,
            'confidence': None
        }
    
    try:
        # Transform the email text using the vectorizer
        email_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = classifier.predict(email_vectorized)[0]
        
        # Get probability scores
        proba = classifier.predict_proba(email_vectorized)[0]
        ham_prob = proba[0]
        spam_prob = proba[1]
        
        # Determine confidence (max probability)
        confidence = max(ham_prob, spam_prob)
        
        # Calculate confidence bar width (for CSS)
        spam_width = round(spam_prob * 100, 1)
        
        # Return results
        return {
            'status': 'success',
            'prediction': prediction,
            'confidence': {
                'ham': round(ham_prob * 100, 2),
                'spam': round(spam_prob * 100, 2),
                'spam_width': spam_width
            },
            'is_spam': prediction == 'spam',
            'message': 'Email classified successfully'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Prediction error: {str(e)}',
            'prediction': None,
            'confidence': None
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for spam prediction
    Accepts JSON: {"email": "email text here"}
    Returns: {"prediction": "spam/ham", "confidence": {...}}
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'email' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please provide email text in JSON format: {"email": "text"}'
            }), 400
        
        email_text = data['email']
        
        if not email_text or not email_text.strip():
            return jsonify({
                'status': 'error',
                'message': 'Email text cannot be empty'
            }), 400
        
        # Make prediction
        result = predict_spam(email_text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """
    HTML form endpoint for spam prediction
    Accepts form data: email=text
    Returns: Rendered HTML page with results
    """
    email_text = request.form.get('email', '')
    
    if not email_text.strip():
        return render_template('index.html', 
                             model_loaded=MODEL_LOADED,
                             error='Please enter email text')
    
    result = predict_spam(email_text)
    
    return render_template('index.html',
                         model_loaded=MODEL_LOADED,
                         prediction=result['prediction'],
                         confidence=result.get('confidence', {}),
                         is_spam=result.get('is_spam', False),
                         email_text=email_text)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'model_type': 'MultinomialNB with CountVectorizer'
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("SPAM EMAIL DETECTOR - WEB APPLICATION")
    print("=" * 60)
    print(f"\nModel Status: {'Loaded' if MODEL_LOADED else 'NOT LOADED'}")
    print("\nStarting Flask server...")
    print("Go to: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
