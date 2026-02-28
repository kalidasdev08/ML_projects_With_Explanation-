"""
Sentiment Analysis Model Training Script
=========================================
This script trains sentiment analysis models using multiple approaches:
1. TF-IDF + Logistic Regression (better for small datasets)
2. LSTM with Word Embeddings (deep learning)

Author: ML Projects
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SentimentTrainer:
    """Class to handle sentiment analysis model training."""
    
    def __init__(self, max_words=10000, max_len=100, embedding_dim=128):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.ml_model = None
        
        self.model = None
        self.history = None
        
    def preprocess_text(self, text):
        """Preprocess text data."""
        if not isinstance(text, str):
            return ""
        
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
    
    def load_amazon_data(self, filepath):
        """Load Amazon reviews data."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Total reviews loaded: {len(df)}")
        
        if 'reviewText' not in df.columns:
            raise ValueError("CSV must have 'reviewText' column")
        if 'overall' not in df.columns:
            raise ValueError("CSV must have 'overall' column (ratings)")
        
        # Convert ratings to sentiment
        def rating_to_sentiment(rating):
            if rating >= 4:
                return 1  # Positive
            elif rating == 3:
                return 2  # Neutral
            else:
                return 0  # Negative
        
        df['sentiment'] = df['overall'].apply(rating_to_sentiment)
        df = df.rename(columns={'reviewText': 'review'})
        df = df.dropna(subset=['review'])
        
        print(f"After cleaning: {len(df)} reviews")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts().sort_index())
        
        return df[['review', 'sentiment']]
    
    def train_ml_model(self, X_texts, y):
        """Train TF-IDF + ML model (more reliable for small datasets)."""
        print("\n" + "="*50)
        print("Training ML Model (TF-IDF + Logistic Regression)")
        print("="*50)
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in X_texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Train Logistic Regression with class weights to handle imbalance
        self.ml_model = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            C=1.0,
            random_state=42
        )
        self.ml_model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test_tfidf)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Negative', 'Positive', 'Neutral']))
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return X_test, y_test
    
    def build_lstm_model(self):
        """Build LSTM model architecture."""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, df):
        """Prepare data for LSTM training."""
        print("Preprocessing reviews...")
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        self.tokenizer.fit_on_texts(df['processed_review'])
        
        X = self.tokenizer.texts_to_sequences(df['processed_review'])
        X = pad_sequences(X, maxlen=self.max_len, padding='post')
        
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        return X, y
    
    def train_lstm(self, X, y):
        """Train LSTM model."""
        print("\n" + "="*50)
        print("Training LSTM Model")
        print("="*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        self.model = self.build_lstm_model()
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=15,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                   target_names=['Negative', 'Positive', 'Neutral']))
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return X_test, y_test
    
    def save_model(self, path='models'):
        """Save all models and preprocessing objects."""
        os.makedirs(path, exist_ok=True)
        
        # Save LSTM model
        if self.model:
            self.model.save(os.path.join(path, 'sentiment_lstm_model.keras'))
        
        # Save ML model (TF-IDF + Logistic Regression)
        if self.ml_model:
            with open(os.path.join(path, 'ml_model.pkl'), 'wb') as f:
                pickle.dump(self.ml_model, f)
        
        # Save TF-IDF vectorizer
        with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save tokenizer
        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save configuration
        config = {
            'max_words': self.max_words,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'use_ml_model': True
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"Models saved to {path}")


def main():
    """Main training function."""
    print("=" * 60)
    print("Sentiment Analysis Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SentimentTrainer(max_words=10000, max_len=100, embedding_dim=128)
    
    # Check for Amazon reviews data
    data_path = 'amazon_reviews.csv'
    
    if os.path.exists(data_path):
        print("\nLoading Amazon reviews data...")
        df = trainer.load_amazon_data(data_path)
    else:
        print("No data file found!")
        return
    
    print(f"\nDataset size: {len(df)} reviews")
    
    # Train ML model (TF-IDF + Logistic Regression) - more reliable
    X_test, y_test = trainer.train_ml_model(df['review'].values, df['sentiment'].values)
    
    # Also prepare and train LSTM (optional)
    X, y = trainer.prepare_data(df)
    trainer.train_lstm(X, y)
    
    # Save all models
    trainer.save_model('models')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
