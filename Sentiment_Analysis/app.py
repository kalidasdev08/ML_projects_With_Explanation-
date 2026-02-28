"""
Flask Web Application for Sentiment Analysis
=============================================
This application provides a web interface for sentiment analysis of product reviews.

Author: ML Projects
"""

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Add parent directory to path for imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing Flask
try:
    from flask import Flask, render_template, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    Flask = None

# Try to import sentiment analyzer, handle missing dependencies gracefully
try:
    from sentiment_analyzer import SentimentAnalyzer, load_analyzer
except ImportError as e:
    print(f"Warning: Could not import sentiment_analyzer: {e}")
    SentimentAnalyzer = None
    load_analyzer = None

if Flask is None:
    raise ImportError("Flask is not installed. Run: pip install flask")

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'sentiment-analysis-secret-key-2024'
app.config['JSON_SORT_KEYS'] = False

# Global analyzer instance
analyzer = None


def initialize_analyzer():
    """Initialize the sentiment analyzer."""
    global analyzer
    
    # Check if sentiment_analyzer is available
    if load_analyzer is None:
        print("Warning: sentiment_analyzer module not available. Running in demo mode.")
        analyzer = None
        return
    
    # Try multiple possible model paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'models'),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.listdir(path):
            # Check if ml_model.pkl exists
            ml_model_file = os.path.join(path, 'ml_model.pkl')
            if os.path.exists(ml_model_file):
                model_path = path
                break
    
    if model_path is None:
        print("Warning: Model not found. Please train the model first.")
        print("Run: python train_model.py")
        analyzer = None
        return
    
    try:
        print(f"Loading trained model from {model_path}...")
        analyzer = load_analyzer(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        analyzer = None


@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze sentiment of a review.
    
    Expects JSON: {'review': 'text to analyze'}
    Returns: {'sentiment': 'Positive/Negative/Neutral', 'confidence': 0.95, ...}
    """
    data = request.get_json()
    
    if not data or 'review' not in data:
        return jsonify({
            'error': 'Please provide a review text'
        }), 400
    
    review = data['review'].strip()
    
    if not review:
        return jsonify({
            'error': 'Review cannot be empty'
        }), 400
    
    if analyzer is None:
        # Return mock response if model not loaded
        # This allows the app to run for demonstration
        result = get_mock_prediction(review)
        return jsonify(result)
    
    try:
        result = analyzer.predict(review)
        
        # Format response
        response = {
            'success': True,
            'review': review,
            'sentiment': result['sentiment'],
            'sentiment_id': result['sentiment_id'],
            'confidence': round(result['confidence'] * 100, 2),
            'emoji': result['emoji'],
            'color': result['color'],
            'probabilities': {
                k: round(v * 100, 2) for k, v in result['probabilities'].items()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple reviews at once.
    
    Expects JSON: {'reviews': ['review1', 'review2', ...]}
    Returns: {'results': [...]}
    """
    data = request.get_json()
    
    if not data or 'reviews' not in data:
        return jsonify({
            'error': 'Please provide a list of reviews'
        }), 400
    
    reviews = data['reviews']
    
    if not isinstance(reviews, list):
        return jsonify({
            'error': 'Reviews must be a list'
        }), 400
    
    if len(reviews) > 50:
        return jsonify({
            'error': 'Maximum 50 reviews allowed at once'
        }), 400
    
    if analyzer is None:
        # Return mock predictions
        results = [get_mock_prediction(review) for review in reviews]
    else:
        try:
            results = analyzer.predict_batch(reviews)
            results = [{
                'success': True,
                'review': r['review'],
                'sentiment': r['sentiment'],
                'sentiment_id': r['sentiment_id'],
                'confidence': round(r['confidence'] * 100, 2),
                'emoji': r['emoji'],
                'color': r['color']
            } for r in results]
        except Exception as e:
            return jsonify({
                'error': f'Prediction error: {str(e)}'
            }), 500
    
    return jsonify({'results': results})


@app.route('/status', methods=['GET'])
def status():
    """Check if model is loaded."""
    return jsonify({
        'model_loaded': analyzer is not None,
        'model_type': 'LSTM with Word Embeddings' if analyzer else None
    })


def get_mock_prediction(review):
    """
    Generate a mock prediction when model is not available.
    This allows the demo to work without training.
    """
    review_lower = review.lower()
    
    # Simple keyword-based sentiment detection
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 
                     'perfect', 'wonderful', 'fantastic', 'happy', 'satisfied',
                     'recommend', 'awesome', 'superb', 'nice', 'beautiful']
    
    negative_words = ['bad', 'terrible', 'horrible', 'worst', 'hate', 'poor',
                     'disappointed', 'awful', 'waste', 'broken', 'cheap', 
                     'never', 'refund', 'return', 'regret', 'faulty']
    
    positive_count = sum(1 for word in positive_words if word in review_lower)
    negative_count = sum(1 for word in negative_words if word in review_lower)
    
    if positive_count > negative_count:
        sentiment = 'Positive'
        sentiment_id = 1
        emoji = '😊'
        color = '#27ae60'
        confidence = min(0.95, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = 'Negative'
        sentiment_id = 0
        emoji = '😞'
        color = '#e74c3c'
        confidence = min(0.95, 0.5 + (negative_count - positive_count) * 0.1)
    else:
        sentiment = 'Neutral'
        sentiment_id = 2
        emoji = '😐'
        color = '#f39c12'
        confidence = 0.5
    
    # Calculate probabilities
    base_prob = 1 - confidence
    if sentiment_id == 0:
        prob_neg = confidence
        prob_pos = base_prob / 2
        prob_neu = base_prob / 2
    elif sentiment_id == 1:
        prob_pos = confidence
        prob_neg = base_prob / 2
        prob_neu = base_prob / 2
    else:
        prob_neu = confidence
        prob_pos = base_prob / 2
        prob_neg = base_prob / 2
    
    return {
        'success': True,
        'review': review,
        'sentiment': sentiment,
        'sentiment_id': sentiment_id,
        'confidence': round(confidence * 100, 2),
        'emoji': emoji,
        'color': color,
        'probabilities': {
            'Positive': round(prob_pos * 100, 2),
            'Negative': round(prob_neg * 100, 2),
            'Neutral': round(prob_neu * 100, 2)
        },
        'demo_mode': True
    }


# Initialize on startup
initialize_analyzer()


if __name__ == '__main__':
    print("=" * 60)
    print("Sentiment Analysis Web Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Go to: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
