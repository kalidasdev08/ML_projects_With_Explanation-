"""
Rainfall Prediction Model Training Script
This script trains a Random Forest classifier to predict whether it will rain tomorrow
based on various weather features from the Australian weather dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Set paths
DATA_PATH = 'Project_Files/WeatherAUS.csv'
MODEL_PATH = 'Project_Files/rainfall_model.pkl'
SCALER_PATH = 'Project_Files/scaler.pkl'

def load_and_preprocess_data():
    """Load and preprocess the weather data"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select features that match the frontend form
    features_to_use = [
        'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Temp9am', 'Temp3pm', 'RainToday', 'WindGustDir', 
        'WindDir9am', 'WindDir3pm'
    ]
    
    target = 'RainTomorrow'
    
    # Check if target exists
    if target not in df.columns:
        print(f"Warning: '{target}' not found in dataset. Creating from Rainfall...")
        # If no RainTomorrow, we'll create a binary target based on Rainfall
        df['RainTomorrow'] = df['Rainfall'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # Select only the columns we need
    df = df[features_to_use + [target]]
    
    # Drop rows with missing target
    df = df.dropna(subset=[target])
    
    # Replace 'NA' with NaN
    df = df.replace('NA', np.nan)
    
    # Fill missing values
    numeric_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
                       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                       'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical variables
    categorical_columns = ['Location', 'RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target].astype(str))
    
    print(f"Target classes: {target_encoder.classes_}")
    print(f"Data after preprocessing: {df.shape}")
    
    return df, features_to_use, target, label_encoders, target_encoder

def train_model():
    """Train the Random Forest model"""
    # Load and preprocess data
    df, features, target, label_encoders, target_encoder = load_and_preprocess_data()
    
    # Prepare features and target
    X = df[features]
    y = df[target]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Save model and related objects
    print("\nSaving model and encoders...")
    
    # Create a dictionary to save all necessary objects
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_importance': feature_importance
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Also save scaler separately (for backward compatibility)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    
    return model_data

if __name__ == "__main__":
    print("=" * 60)
    print("Rainfall Prediction Model Training")
    print("=" * 60)
    
    model_data = train_model()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
