"""
Fake News Detection System
ML Type: NLP Classification
Dataset: Fake News Dataset
Skills: TF-IDF, Naive Bayes, Text Cleaning

This module provides a complete pipeline for detecting fake news articles
using TF-IDF vectorization and Naive Bayes classification.
"""

import os
import re
import string
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data if needed
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

warnings.filterwarnings('ignore')


class TextCleaner:
    """Class for cleaning and preprocessing text data."""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                                   'ourselves', 'you', 'your', 'yours', 'yourself', 
                                   'yourselves', 'he', 'him', 'his', 'himself', 
                                   'she', 'her', 'hers', 'herself', 'it', 'its', 
                                   'itself', 'they', 'them', 'their', 'theirs', 
                                   'themselves', 'what', 'which', 'who', 'whom', 
                                   'this', 'that', 'these', 'those', 'am', 'is', 
                                   'are', 'was', 'were', 'be', 'been', 'being', 
                                   'have', 'has', 'had', 'having', 'do', 'does', 
                                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
                                   'if', 'or', 'because', 'as', 'until', 'while', 
                                   'of', 'at', 'by', 'for', 'with', 'about', 
                                   'against', 'between', 'into', 'through', 'during', 
                                   'before', 'after', 'above', 'below', 'to', 
                                   'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                                   'over', 'under', 'again', 'further', 'then', 
                                   'once', 'here', 'there', 'when', 'where', 'why', 
                                   'how', 'all', 'any', 'both', 'each', 'few', 
                                   'more', 'most', 'other', 'some', 'such', 'no', 
                                   'nor', 'not', 'only', 'own', 'same', 'so', 
                                   'than', 'too', 'very', 's', 't', 'can', 'will', 
                                   'just', 'don', 'should', 'now'])
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove special characters and numbers
        5. Remove extra whitespace
        6. Remove stopwords
        7. Lemmatize words
        """
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
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords and lemmatize
        if self.lemmatizer:
            words = text.split()
            words = [self.lemmatizer.lemmatize(word) for word in words 
                    if word not in self.stop_words]
            text = ' '.join(words)
        else:
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def clean_dataframe(self, df, text_column='text'):
        """Clean all text in a DataFrame column."""
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        return df


class FakeNewsDetector:
    """
    Fake News Detection model using TF-IDF and Naive Bayes.
    
    Pipeline:
    1. Text cleaning and preprocessing
    2. TF-IDF vectorization
    3. Naive Bayes classification
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the Fake News Detector.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to use (1, 2) = unigrams and bigrams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.text_cleaner = TextCleaner()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.classifier = MultinomialNB(alpha=0.1)
        self.is_trained = False
        self.training_accuracy = None
        self.test_accuracy = None
    
    def preprocess_data(self, df, text_column='text', label_column='label'):
        """
        Preprocess the dataset.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Preprocessed DataFrame
        """
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna('')
        
        # Clean text
        df = self.text_cleaner.clean_dataframe(df, text_column)
        
        # Remove empty texts
        df = df[df[text_column].str.len() > 0]
        
        print(f"Preprocessing complete. {len(df)} samples remaining.")
        return df
    
    def load_data(self, file_path, text_column='text', label_column='label'):
        """
        Load dataset from CSV or Excel file.
        
        Args:
            file_path: Path to the dataset file
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        print(f"Loading data from {file_path}...")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Rename columns if needed
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'article' in col_lower or 'news' in col_lower:
                if col != text_column:
                    column_mapping[col] = text_column
            if 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
                if col != label_column:
                    column_mapping[col] = label_column
        
        df = df.rename(columns=column_mapping)
        
        # Ensure label column exists and is binary
        if label_column not in df.columns:
            # Try to find a suitable column
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].unique()
                    if len(unique_vals) == 2:
                        column_mapping[col] = label_column
                        df = df.rename(columns=column_mapping)
                        break
        
        print(f"Loaded {len(df)} samples.")
        return df
    
    def prepare_labels(self, df, label_column='label'):
        """
        Convert labels to binary (0/1).
        
        Args:
            df: DataFrame with label column
            label_column: Name of the label column
            
        Returns:
            DataFrame with binary labels
        """
        df = df.copy()
        
        # Convert labels to binary
        if df[label_column].dtype == 'object':
            # Map labels like 'REAL', 'FAKE', 'True', 'False', etc.
            unique_labels = df[label_column].unique()
            print(f"Found labels: {unique_labels}")
            
            # Create mapping
            label_map = {}
            for label in unique_labels:
                label_lower = str(label).lower()
                if 'real' in label_lower or 'true' in label_lower or label == '1':
                    label_map[label] = 1
                elif 'fake' in label_lower or 'false' in label_lower or label == '0':
                    label_map[label] = 0
            
            df[label_column] = df[label_column].map(label_map)
        
        return df
    
    def train(self, df, text_column='text', label_column='label', test_size=0.2):
        """
        Train the fake news detection model.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*50)
        print("Training Fake News Detection Model")
        print("="*50)
        
        # Preprocess data
        df = self.prepare_labels(df, label_column)
        df = self.preprocess_data(df, text_column, label_column)
        
        # Extract features and labels
        X = df[text_column].values
        y = df[label_column].values
        
        # Remove any remaining NaN
        valid_idx = ~(pd.isna(X) | pd.isna(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize text using TF-IDF
        print("\nVectorizing text with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train Naive Bayes classifier
        print("\nTraining Naive Bayes classifier...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        train_pred = self.classifier.predict(X_train_tfidf)
        test_pred = self.classifier.predict(X_test_tfidf)
        
        self.training_accuracy = accuracy_score(y_train, train_pred)
        self.test_accuracy = accuracy_score(y_test, test_pred)
        
        print("\n" + "="*50)
        print("Training Results")
        print("="*50)
        print(f"Training Accuracy: {self.training_accuracy:.4f} ({self.training_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {self.test_accuracy:.4f} ({self.test_accuracy*100:.2f}%)")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_pred, target_names=['FAKE', 'REAL']))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"                Predicted")
        print(f"              FAKE    REAL")
        print(f"Actual FAKE   {cm[0][0]:4d}   {cm[0][1]:4d}")
        print(f"Actual REAL   {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        self.is_trained = True
        self.label_map = {0: 'FAKE', 1: 'REAL'}
        
        # Feature importance (top words)
        self._print_top_features()
        
        results = {
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_features': X_train_tfidf.shape[1]
        }
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        return results
    
    def _print_top_features(self):
        """Print top features for each class."""
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get log probabilities for each class
        log_prob = self.classifier.feature_log_prob_
        
        # Top features for FAKE (class 0)
        top_fake_idx = log_prob[0].argsort()[-10:][::-1]
        print("\nTop 10 FAKE News Indicators:")
        for idx in top_fake_idx:
            print(f"  - {feature_names[idx]}")
        
        # Top features for REAL (class 1)
        top_real_idx = log_prob[1].argsort()[-10:][::-1]
        print("\nTop 10 REAL News Indicators:")
        for idx in top_real_idx:
            print(f"  - {feature_names[idx]}")
    
    def predict(self, text):
        """
        Predict whether a news article is REAL or FAKE.
        
        Args:
            text: News article text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Clean the text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.classifier.predict(text_tfidf)[0]
        
        # Get probabilities
        proba = self.classifier.predict_proba(text_tfidf)[0]
        
        result = {
            'prediction': self.label_map[prediction],
            'prediction_code': int(prediction),
            'confidence': float(max(proba)),
            'probability_fake': float(proba[0]),
            'probability_real': float(proba[1])
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict for multiple news articles.
        
        Args:
            texts: List of news article texts
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Clean all texts
        cleaned_texts = [self.text_cleaner.clean_text(text) for text in texts]
        
        # Vectorize
        texts_tfidf = self.vectorizer.transform(cleaned_texts)
        
        # Predict
        predictions = self.classifier.predict(texts_tfidf)
        probas = self.classifier.predict_proba(texts_tfidf)
        
        results = []
        for pred, proba in zip(predictions, probas):
            results.append({
                'prediction': self.label_map[pred],
                'prediction_code': int(pred),
                'confidence': float(max(proba)),
                'probability_fake': float(proba[0]),
                'probability_real': float(proba[1])
            })
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'label_map': self.label_map
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        self.training_accuracy = model_data['training_accuracy']
        self.test_accuracy = model_data['test_accuracy']
        self.label_map = model_data['label_map']
        
        print(f"Model loaded from {filepath}")
        print(f"Training Accuracy: {self.training_accuracy:.4f}")
        print(f"Test Accuracy: {self.test_accuracy:.4f}")


def create_sample_dataset():
    """
    Create a sample dataset for demonstration.
    This mimics the structure of a fake news dataset.
    """
    # Sample fake news articles
    fake_articles = [
        "BREAKING: Scientists discover secret cure for cancer that pharmaceutical companies don't want you to know about. Big Pharma is hiding this from you!",
        "SHOCKING: Politician caught accepting bribes from foreign governments. The mainstream media won't report this!",
        "ALERT: New world order plan exposed! Government officials secretly meeting to control population through vaccines.",
        "WARNING: Your food is being poisoned by the government! Watch this video to learn the truth they don't want you to know.",
        "EXCLUSIVE: Celebrity scandal cover-up! Hollywood elites involved in satanic rituals. The truth finally revealed!",
        "URGENT: Bank accounts about to be frozen! International bankers plotting to steal your money. Share before it's too late!",
        "SENSATIONAL: NASA faked the moon landing! Evidence proves astronauts never went to space. It's all a lie!",
        "SHOCKING: Vaccines cause autism! Doctors paid to hide the truth. Your children are in danger!",
        "BREAKING: Earth is actually flat! NASA and governments have been lying to us for centuries!",
        "ALERT: Massive conspiracy revealed! The Illuminati controls everything. Wake up sheeple!"
    ]
    
    # Sample real news articles
    real_articles = [
        "The Federal Reserve announced today that it would maintain current interest rates following the latest economic data analysis. Fed Chair emphasized the importance of careful monetary policy.",
        "Scientists at MIT have published a new study on renewable energy efficiency. The research shows significant improvements in solar panel technology.",
        "The Senate voted today to approve the new infrastructure bill. The legislation includes funding for roads, bridges, and broadband expansion.",
        "The World Health Organization released updated guidelines for COVID-19 prevention. The recommendations include vaccination and mask-wearing in crowded areas.",
        "Economic indicators show moderate growth in the manufacturing sector. Analysts predict continued expansion in the coming quarters.",
        "The Supreme Court heard arguments today on a case involving environmental regulations. A decision is expected within the next few months.",
        "Tech companies reported strong quarterly earnings, with revenue growth driven by cloud computing and artificial intelligence services.",
        "The United Nations climate summit concluded with agreements on emissions reduction targets. World leaders committed to renewable energy investments.",
        "Medical researchers announced promising results from clinical trials for a new treatment. The therapy shows effectiveness in early-stage patients.",
        "The central bank released its monthly economic report, noting stable employment figures and moderate inflation growth."
    ]
    
    # Create DataFrame
    data = []
    for article in fake_articles:
        data.append({'text': article, 'label': 'FAKE'})
    for article in real_articles:
        data.append({'text': article, 'label': 'REAL'})
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main function to demonstrate the Fake News Detection system."""
    
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("ML Type: NLP Classification")
    print("Method: TF-IDF + Naive Bayes")
    print("="*60)
    
    # Try to load available dataset, or use sample data
    dataset_paths = [
        'Fake_new_prediction/train.csv',
        'Fake_new_prediction/fake_news_dataset.csv',
        'data/fake_news.csv',
        'fake_news.csv'
    ]
    
    df = None
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                detector = FakeNewsDetector()
                df = detector.load_data(path)
                break
            except Exception as e:
                print(f"Could not load {path}: {e}")
    
    # If no dataset found, create sample data
    if df is None:
        print("\nNo dataset found. Creating sample dataset for demonstration...")
        df = create_sample_dataset()
        print(f"Created sample dataset with {len(df)} articles")
        print(f"Fake articles: {sum(df['label'] == 'FAKE')}")
        print(f"Real articles: {sum(df['label'] == 'REAL')}")
    
    # Initialize and train model
    detector = FakeNewsDetector(max_features=5000, ngram_range=(1, 2))
    results = detector.train(df, text_column='text', label_column='label')
    
    # Save model
    model_path = 'Fake_new_prediction/fake_news_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    detector.save_model(model_path)
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_articles = [
        "Scientists discover secret cure for cancer that pharmaceutical companies don't want you to know about!",
        "The Federal Reserve announced today that it would maintain current interest rates.",
        "BREAKING: Government is adding chemicals to your water supply! This secret will shock you!",
        "The Senate voted today to approve the new infrastructure bill with bipartisan support.",
        "ALERT: Massive conspiracy revealed! The Illuminati controls everything!"
    ]
    
    for article in test_articles:
        result = detector.predict(article)
        print(f"\nArticle: {article[:80]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"P(FAKE): {result['probability_fake']*100:.2f}% | P(REAL): {result['probability_real']*100:.2f}%")
    
    return detector, results


if __name__ == "__main__":
    detector, results = main()
