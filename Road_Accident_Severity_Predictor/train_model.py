"""
Road Accident Severity Predictor - Model Training Script
Predicts accident severity (fatality level) using road and weather features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic road accident data with road and weather features."""
    
    # Define feature ranges and categories
    weather_conditions = ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Stormy', 'Cloudy']
    road_types = ['Highway', 'Urban Road', 'Rural Road', 'Expressway', 'Residential']
    road_surface = ['Dry', 'Wet', 'Icy', 'Snowy', 'Flooded']
    light_conditions = ['Daylight', 'Night - Lit', 'Night - Unlit', 'Dawn', 'Dusk']
    vehicle_types = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Van', 'Bicycle']
    driver_conditions = ['Normal', 'Drowsy', 'Drunk', 'Distracted', 'Reckless']
    
    data = {
        'weather_condition': np.random.choice(weather_conditions, n_samples),
        'road_type': np.random.choice(road_types, n_samples),
        'road_surface': np.random.choice(road_surface, n_samples),
        'light_condition': np.random.choice(light_conditions, n_samples),
        'vehicle_type': np.random.choice(vehicle_types, n_samples),
        'driver_condition': np.random.choice(driver_conditions, n_samples),
        'speed_limit': np.random.randint(20, 130, n_samples),
        'vehicle_speed': np.random.randint(0, 150, n_samples),
        'number_of_vehicles': np.random.randint(1, 10, n_samples),
        'number_of_casualties': np.random.randint(0, 20, n_samples),
        'humidity': np.random.randint(20, 100, n_samples),
        'temperature': np.random.randint(-10, 45, n_samples),
        'wind_speed': np.random.randint(0, 50, n_samples),
        'visibility': np.random.randint(100, 10000, n_samples),
        'traffic_density': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate severity labels based on feature combinations (synthetic rules)
    def calculate_severity(row):
        score = 0
        
        # Road and environmental factors
        if row['road_surface'] in ['Icy', 'Snowy', 'Flooded']:
            score += 3
        elif row['road_surface'] == 'Wet':
            score += 1
            
        if row['weather_condition'] in ['Stormy', 'Foggy', 'Snowy']:
            score += 2
        elif row['weather_condition'] == 'Rainy':
            score += 1
            
        if row['light_condition'] in ['Night - Unlit', 'Dusk', 'Dawn']:
            score += 2
            
        if row['road_type'] in ['Rural Road', 'Highway']:
            score += 1
            
        # Speed factors
        if row['vehicle_speed'] > 100:
            score += 3
        elif row['vehicle_speed'] > 70:
            score += 2
        elif row['vehicle_speed'] > 50:
            score += 1
            
        if row['speed_limit'] > 100:
            score += 1
            
        # Vehicle and driver factors
        if row['driver_condition'] in ['Drunk', 'Drowsy', 'Reckless']:
            score += 4
        elif row['driver_condition'] == 'Distracted':
            score += 2
            
        if row['vehicle_type'] in ['Truck', 'Bus']:
            score += 1
            
        # Traffic and visibility factors
        if row['traffic_density'] == 'High':
            score += 1
            
        if row['visibility'] < 500:
            score += 2
        elif row['visibility'] < 1000:
            score += 1
            
        # Number of vehicles and casualties
        if row['number_of_vehicles'] > 3:
            score += 1
        score += min(row['number_of_casualties'], 3)
        
        # Convert score to severity level
        if score <= 3:
            return 'Slight'
        elif score <= 7:
            return 'Serious'
        else:
            return 'Fatal'
    
    df['severity'] = df.apply(calculate_severity, axis=1)
    
    return df

def train_model():
    """Train the accident severity prediction model."""
    
    print("Generating synthetic accident data...")
    df = generate_synthetic_data(n_samples=15000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nSeverity distribution:\n{df['severity'].value_counts()}")
    
    # Prepare features and target
    X = df.drop('severity', axis=1)
    y = df['severity']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
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
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=target_encoder.classes_
    ))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model and encoders
    model_dir = 'Road_Accident_Severity_Predictor'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'accident_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    joblib.dump(target_encoder, os.path.join(model_dir, 'target_encoder.pkl'))
    
    print(f"\nModel saved to {model_dir}/accident_model.pkl")
    
    # Return feature names for the app
    return X.columns.tolist(), label_encoders, target_encoder

if __name__ == '__main__':
    feature_names, encoders, target_encoder = train_model()
    print("\nTraining complete!")
