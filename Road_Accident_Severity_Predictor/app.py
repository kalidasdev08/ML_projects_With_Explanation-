"""
Road Accident Severity Predictor - Flask Web Application
Predicts accident fatality level using road and weather data.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and preprocessors
model_dir = 'Road_Accident_Severity_Predictor'

try:
    model = joblib.load(os.path.join(model_dir, 'accident_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    target_encoder = joblib.load(os.path.join(model_dir, 'target_encoder.pkl'))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found. Please run train_model.py first.")
    model = None

# Feature options for the form
FEATURE_OPTIONS = {
    'weather_condition': ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Stormy', 'Cloudy'],
    'road_type': ['Highway', 'Urban Road', 'Rural Road', 'Expressway', 'Residential'],
    'road_surface': ['Dry', 'Wet', 'Icy', 'Snowy', 'Flooded'],
    'light_condition': ['Daylight', 'Night - Lit', 'Night - Unlit', 'Dawn', 'Dusk'],
    'vehicle_type': ['Car', 'Motorcycle', 'Truck', 'Bus', 'Van', 'Bicycle'],
    'driver_condition': ['Normal', 'Drowsy', 'Drunk', 'Distracted', 'Reckless'],
    'traffic_density': ['Low', 'Medium', 'High']
}

# Column order must match training data
COLUMN_ORDER = [
    'weather_condition', 'road_type', 'road_surface', 'light_condition',
    'vehicle_type', 'driver_condition', 'speed_limit', 'vehicle_speed',
    'number_of_vehicles', 'number_of_casualties', 'humidity', 'temperature',
    'wind_speed', 'visibility', 'traffic_density'
]

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    if model is None:
        return "Error: Model not loaded. Please run train_model.py first."
    return render_template('index.html', feature_options=FEATURE_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    if model is None:
        return "Error: Model not loaded."
    
    try:
        # Get form data
        features = {
            'weather_condition': request.form.get('weather_condition'),
            'road_type': request.form.get('road_type'),
            'road_surface': request.form.get('road_surface'),
            'light_condition': request.form.get('light_condition'),
            'vehicle_type': request.form.get('vehicle_type'),
            'driver_condition': request.form.get('driver_condition'),
            'speed_limit': int(request.form.get('speed_limit', 60)),
            'vehicle_speed': int(request.form.get('vehicle_speed', 50)),
            'number_of_vehicles': int(request.form.get('number_of_vehicles', 1)),
            'number_of_casualties': int(request.form.get('number_of_casualties', 0)),
            'humidity': int(request.form.get('humidity', 50)),
            'temperature': int(request.form.get('temperature', 20)),
            'wind_speed': int(request.form.get('wind_speed', 10)),
            'visibility': int(request.form.get('visibility', 5000)),
            'traffic_density': request.form.get('traffic_density')
        }
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([features])[COLUMN_ORDER]
        
        # Encode categorical features
        for col in FEATURE_OPTIONS.keys():
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Scale numerical features
        numerical_cols = ['speed_limit', 'vehicle_speed', 'number_of_vehicles', 
                         'number_of_casualties', 'humidity', 'temperature', 
                         'wind_speed', 'visibility']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)
        severity = target_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        proba = model.predict_proba(input_df)[0]
        probabilities = {
            cls: round(prob * 100, 2) 
            for cls, prob in zip(target_encoder.classes_, proba)
        }
        
        return render_template('index.html', 
                             feature_options=FEATURE_OPTIONS,
                             prediction=severity,
                             probabilities=probabilities,
                             input_data=features)
    
    except Exception as e:
        return render_template('index.html',
                             feature_options=FEATURE_OPTIONS,
                             error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
