"""
Rainfall Prediction Flask Backend
This Flask application serves rainfall predictions using a trained ML model.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Set paths
MODEL_PATH = 'Project_Files/rainfall_model.pkl'
TEMPLATES_PATH = 'Project_Files/templates'

# Load the trained model
print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    target_encoder = model_data['target_encoder']
    
    print(f"Model loaded successfully!")
    print(f"Target classes: {target_encoder.classes_}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    features = None
    target_encoder = None

@app.route('/')
def home():
    """Render the main prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a rainfall prediction based on form data"""
    if model is None:
        return render_template('index.html', error="Model not loaded. Please train the model first.")
    
    try:
        # Extract form data
        input_data = {
            'Location': int(request.form.get('Location', 0)),
            'MinTemp': float(request.form.get('MinTemp', 0)),
            'MaxTemp': float(request.form.get('MaxTemp', 0)),
            'Rainfall': float(request.form.get('Rainfall', 0)),
            'WindGustSpeed': float(request.form.get('WindGustSpeed', 0)),
            'WindSpeed9am': float(request.form.get('WindSpeed9am', 0)),
            'WindSpeed3pm': float(request.form.get('WindSpeed3pm', 0)),
            'Humidity9am': float(request.form.get('Humidity9am', 0)),
            'Humidity3pm': float(request.form.get('Humidity3pm', 0)),
            'Pressure9am': float(request.form.get('Pressure9am', 0)),
            'Pressure3pm': float(request.form.get('Pressure3pm', 0)),
            'Temp9am': float(request.form.get('Temp9am', 0)),
            'Temp3pm': float(request.form.get('Temp3pm', 0)),
            'RainToday': int(request.form.get('RainToday', 0)),
            'WindGustDir': int(request.form.get('WindGustDir', 0)),
            'WindDir9am': int(request.form.get('WindDir9am', 0)),
            'WindDir3pm': int(request.form.get('WindDir3pm', 0))
        }
        
        # Create feature array in the correct order
        feature_array = np.array([[
            input_data['Location'],
            input_data['MinTemp'],
            input_data['MaxTemp'],
            input_data['Rainfall'],
            input_data['WindGustSpeed'],
            input_data['WindSpeed9am'],
            input_data['WindSpeed3pm'],
            input_data['Humidity9am'],
            input_data['Humidity3pm'],
            input_data['Pressure9am'],
            input_data['Pressure3pm'],
            input_data['Temp9am'],
            input_data['Temp3pm'],
            input_data['RainToday'],
            input_data['WindGustDir'],
            input_data['WindDir9am'],
            input_data['WindDir3pm']
        ]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # Get probability of rain (class 1 typically represents "Yes")
        rain_class_idx = list(target_encoder.classes_).index('Yes') if 'Yes' in target_encoder.classes_ else 1
        rain_probability = probability[rain_class_idx] * 100
        
        # Map prediction to label
        prediction_label = target_encoder.inverse_transform([prediction])[0]
        
        print(f"Prediction: {prediction_label}, Rain Probability: {rain_probability:.2f}%")
        
        # Render appropriate template based on prediction
        if prediction_label == 'Yes' or prediction == 1:
            return render_template('chance.html', probability=f"{rain_probability:.2f}%")
        else:
            return render_template('nochance.html', probability=f"{rain_probability:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Extract features from JSON
        feature_values = [
            data.get('Location', 0),
            data.get('MinTemp', 0),
            data.get('MaxTemp', 0),
            data.get('Rainfall', 0),
            data.get('WindGustSpeed', 0),
            data.get('WindSpeed9am', 0),
            data.get('WindSpeed3pm', 0),
            data.get('Humidity9am', 0),
            data.get('Humidity3pm', 0),
            data.get('Pressure9am', 0),
            data.get('Pressure3pm', 0),
            data.get('Temp9am', 0),
            data.get('Temp3pm', 0),
            data.get('RainToday', 0),
            data.get('WindGustDir', 0),
            data.get('WindDir9am', 0),
            data.get('WindDir3pm', 0)
        ]
        
        feature_array = np.array([feature_values])
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # Get rain probability
        rain_class_idx = list(target_encoder.classes_).index('Yes') if 'Yes' in target_encoder.classes_ else 1
        rain_probability = float(probability[rain_class_idx])
        
        prediction_label = target_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            'prediction': prediction_label,
            'rain_probability': f"{rain_probability * 100:.2f}%",
            'no_rain_probability': f"{(1 - rain_probability) * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Set template folder
    app.template_folder = TEMPLATES_PATH
    
    print("=" * 60)
    print("Starting Rainfall Prediction Backend")
    print("=" * 60)
    print(f"Template folder: {TEMPLATES_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print("\nServer running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
