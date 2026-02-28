"""
Sentiment Analyzer - Prediction Logic
=====================================
This module handles loading the trained model and making predictions.
Uses TF-IDF + Logistic Regression for better accuracy.

Author: ML Projects
"""

import os
import re
import json
import pickle
import numpy as np

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Deep Learning
import tensorflow as tf

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class SentimentAnalyzer:
    """Class for sentiment analysis predictions."""
    
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.ml_model = None
        self.tfidf_vectorizer = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Sentiment mappings
        self.sentiment_labels = {
            0: 'Negative',
            1: 'Positive',
            2: 'Neutral'
        }
        
        self.sentiment_emojis = {
            0: '😞',
            1: '😊',
            2: '😐'
        }
        
        self.sentiment_colors = {
            0: '#e74c3c',  # Red
            1: '#27ae60',  # Green
            2: '#f39c12'   # Orange/Yellow
        }
        
    def load_model(self):
        """Load the trained model and preprocessing objects."""
        print("Loading model...")
        
        # Load config
        config_path = os.path.join(self.model_path, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Try to load ML model (TF-IDF + Logistic Regression)
        ml_model_path = os.path.join(self.model_path, 'ml_model.pkl')
        if os.path.exists(ml_model_path):
            with open(ml_model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            print("Loaded ML model (TF-IDF + Logistic Regression)")
        
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(self.model_path, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print("Loaded TF-IDF vectorizer")
        
        # Try to load LSTM model as backup
        lstm_path = os.path.join(self.model_path, 'sentiment_lstm_model.keras')
        if os.path.exists(lstm_path):
            try:
                self.model = tf.keras.models.load_model(lstm_path)
                print("Loaded LSTM model")
            except:
                self.model = None
                print("Could not load LSTM model")
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_path, 'tokenizer.pkl')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        
        # Load label encoder
        label_encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        print("Model loaded successfully!")
        print(f"Using ML model: {self.ml_model is not None}")
        print(f"Using LSTM model: {self.model is not None}")
    
    def preprocess_text(self, text):
        """Preprocess text for prediction."""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict(self, review):
        """Predict sentiment of a review."""
        # First check for obvious negative/positive keywords
        keyword_result = self._check_keywords(review)
        if keyword_result:
            return keyword_result
        
        # Use ML model if available (more accurate for small datasets)
        if self.ml_model is not None and self.tfidf_vectorizer is not None:
            return self._predict_ml(review)
        elif self.model is not None:
            return self._predict_lstm(review)
        else:
            raise ValueError("No model available")
    
    def _check_keywords(self, review):
        """Quick keyword-based check for obvious sentiments."""
        text = review.lower()
        
        # Strong negative words
        negative_words = [
            'bad', 'worst', 'terrible', 'horrible', 'awful', 'hate', 
            'waste', 'disappointed', 'poor', 'broken', 'refund', 'return',
            'scam', 'fraud', 'fake', 'useless', 'pathetic', 'regret',
            'avoid', 'never', 'overpriced', 'shame', 'trash', 'garbage',
            'cheap', 'defective', 'faulty', 'broken', 'doesn\'t', 'not'
        ]
        
        # Strong positive words
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect',
            'love', 'best', 'awesome', 'fantastic', 'recommend', 'happy',
            'satisfied', 'beautiful', 'quality', 'fast', 'works', 'helpful'
        ]
        
        neg_count = sum(1 for word in negative_words if word in text)
        pos_count = sum(1 for word in positive_words if word in text)
        
        # If strong negative signal, override model prediction
        if neg_count > pos_count and neg_count > 0:
            return {
                'review': review,
                'processed_review': self.preprocess_text(review),
                'sentiment': 'Negative',
                'sentiment_id': 0,
                'confidence': min(0.95, 0.6 + neg_count * 0.1),
                'emoji': '😞',
                'color': '#e74c3c',
                'probabilities': {
                    'Negative': min(0.95, 0.6 + neg_count * 0.1),
                    'Positive': max(0.05, 0.2 - neg_count * 0.05),
                    'Neutral': 0.1
                }
            }
        
        # If strong positive signal, override model prediction
        if pos_count > neg_count and pos_count > 0:
            return {
                'review': review,
                'processed_review': self.preprocess_text(review),
                'sentiment': 'Positive',
                'sentiment_id': 1,
                'confidence': min(0.95, 0.6 + pos_count * 0.1),
                'emoji': '😊',
                'color': '#27ae60',
                'probabilities': {
                    'Positive': min(0.95, 0.6 + pos_count * 0.1),
                    'Negative': max(0.05, 0.2 - pos_count * 0.05),
                    'Neutral': 0.1
                }
            }
        
        return None  # No strong keyword signal, use model
    
    def _predict_ml(self, review):
        """Predict using TF-IDF + Logistic Regression."""
        processed_review = self.preprocess_text(review)
        
        # Transform using TF-IDF
        X = self.tfidf_vectorizer.transform([processed_review])
        
        # Get prediction and probabilities
        predicted_class = self.ml_model.predict(X)[0]
        probabilities = self.ml_model.predict_proba(X)[0]
        
        confidence = float(probabilities[predicted_class])
        
        # Map classes
        sentiment = self.sentiment_labels[predicted_class]
        emoji = self.sentiment_emojis[predicted_class]
        color = self.sentiment_colors[predicted_class]
        
        # Get all class probabilities (map to correct labels)
        class_labels = self.ml_model.classes_
        prob_dict = {}
        for i, cls in enumerate(class_labels):
            prob_dict[self.sentiment_labels[cls]] = float(probabilities[i])
        
        # Ensure all three sentiments are in the dict
        for label in ['Positive', 'Negative', 'Neutral']:
            if label not in prob_dict:
                prob_dict[label] = 0.0
        
        return {
            'review': review,
            'processed_review': processed_review,
            'sentiment': sentiment,
            'sentiment_id': int(predicted_class),
            'confidence': confidence,
            'emoji': emoji,
            'color': color,
            'probabilities': prob_dict
        }
    
    def _predict_lstm(self, review):
        """Predict using LSTM model."""
        processed_review = self.preprocess_text(review)
        
        sequence = self.tokenizer.texts_to_sequences([processed_review])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=self.config['max_len'],
            padding='post'
        )
        
        prediction = self.model.predict(padded_sequence, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        sentiment = self.sentiment_labels[predicted_class]
        emoji = self.sentiment_emojis[predicted_class]
        color = self.sentiment_colors[predicted_class]
        
        probabilities = {
            self.sentiment_labels[i]: float(prediction[i])
            for i in range(len(prediction))
        }
        
        return {
            'review': review,
            'processed_review': processed_review,
            'sentiment': sentiment,
            'sentiment_id': int(predicted_class),
            'confidence': confidence,
            'emoji': emoji,
            'color': color,
            'probabilities': probabilities
        }
    
    def predict_batch(self, reviews):
        """Predict sentiment for multiple reviews."""
        return [self.predict(review) for review in reviews]


def load_analyzer(model_path='models'):
    """Load and return a sentiment analyzer instance."""
    analyzer = SentimentAnalyzer(model_path)
    analyzer.load_model()
    return analyzer


if __name__ == '__main__':
    print("Testing Sentiment Analyzer...")
    print("=" * 50)
    
    model_dir = 'models'
    if os.path.exists(model_dir) and os.listdir(model_dir):
        analyzer = load_analyzer(model_dir)
        
        test_reviews = [
            "This product is amazing! Best purchase ever.",
            "This product is terrible. Waste of money.",
            "The product is okay. Not great, not bad.",
            "I love this! Works perfectly.",
            "Very disappointed. Poor quality."
        ]
        
        print("\nTesting with sample reviews:\n")
        for review in test_reviews:
            result = analyzer.predict(review)
            print(f"Review: {review}")
            print(f"Sentiment: {result['emoji']} {result['sentiment']} ({result['confidence']:.2%})")
            print("-" * 50)
    else:
        print("Model not found. Please run train_model.py first.")
