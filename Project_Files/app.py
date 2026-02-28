import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
model = pickle.load(open('rainfall.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))
@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/predict', methods=["POST", "GET"])# route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    if request.method == "GET":
        return jsonify({"error": "Please submit form data via POST"}), 400
    
    try:
        # Define feature names in the exact order the model expects
        feature_names = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir', 'WindGustSpeed',
                        'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                        'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday']
        
        # Extract values in the correct order
        input_values = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None:
                return jsonify({"error": f"Missing required field: {feature}"}), 400
            input_values.append(float(value))
        
        # Convert to numpy array for scaling
        features_values = np.array(input_values).reshape(1, -1)
        
        # Scale the data
        scaled_data = scale.transform(features_values)
        
        # Make predictions using the loaded model file
        prediction = model.predict(scaled_data)
        pred_prob = model.predict_proba(scaled_data)
        
        # Log actual prediction results
        print(f"[PREDICTION] Input: {input_values}")
        print(f"[PREDICTION] Result: {prediction[0]}")
        print(f"[PREDICTION] Probability - Yes: {pred_prob[0][1]:.4f}, No: {pred_prob[0][0]:.4f}")
        
        # Return result based on prediction
        if prediction[0] == 1:
            return render_template("chance.html", probability=round(pred_prob[0][1], 2))
        else:
            return render_template("nochance.html", probability=round(pred_prob[0][0], 2))
            
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
# showing the prediction results in a UI
if __name__=="__main__":
    app.run(debug=True, port=5000)