import pickle
import numpy as np
import pandas as pd
import os

def test_model():
    print("Loading models...")
    try:
        model = pickle.load(open('rainfall.pkl', 'rb'))
        scale = pickle.load(open('scale.pkl', 'rb'))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Feature names validation - Corrected Order matching Scaler/Model inspection
    feature_names = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir', 'WindGustSpeed',
                     'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                     'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday']
    
    print(f"\nExpected Features: {len(feature_names)}")

    # Test Case 1: Likely No Rain (Sunny)
    # Reordered values to match new feature_names order
    # Old Order: Location, MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Temp9am, Temp3pm, RainToday, WindGustDir, WindDir9am, WindDir3pm
    # New Order: Location, MinTemp, MaxTemp, Rainfall, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Temp9am, Temp3pm, RainToday

    # Input Sunny Original: [2, 15.0, 30.0, 0.0, 30.0, 10.0, 15.0, 40.0, 25.0, 1020.0, 1018.0, 20.0, 28.0, 0, 13, 13, 13]
    # Input Sunny Reordered:
    input_sunny = [2, 15.0, 30.0, 0.0, 13, 30.0, 
                   13, 13, 10.0, 15.0, 40.0, 
                   25.0, 1020.0, 1018.0, 20.0, 28.0, 0]
    
    # Test Case 2: Likely Rain (Stormy)
    # Input Rainy Original: [2, 10.0, 15.0, 15.0, 60.0, 20.0, 25.0, 95.0, 90.0, 1005.0, 1002.0, 11.0, 13.0, 1, 0, 0, 0]
    # Input Rainy Reordered:
    input_rainy = [2, 10.0, 15.0, 15.0, 0, 60.0, 
                   0, 0, 20.0, 25.0, 95.0, 
                   90.0, 1005.0, 1002.0, 11.0, 13.0, 1]

    # Test Case 3: User's Failing Input
    # [6.0, 12.0, 32.4, 10.2, 1.0, 32.9, 3.0, 1.0, 43.9, 22.1, 56.9, 56.9, 7709.9, 1212.9, 32.0, 21.0, 1.0]
    input_user_fail = [6.0, 12.0, 32.4, 10.2, 1.0, 32.9, 3.0, 1.0, 43.9, 22.1, 56.9, 56.9, 7709.9, 1212.9, 32.0, 21.0, 1.0]

    tests = [("Sunny/Dry", input_sunny), ("Stormy/Wet", input_rainy), ("User Input (Fail)", input_user_fail)]

    for name, data in tests:
        print(f"\n--- Testing Scenario: {name} ---")
        try:
            # Scale
            features_values = np.array(data).reshape(1, -1)
            scaled_data = scale.transform(features_values)
            
            # Predict
            prediction = model.predict(scaled_data)
            pred_prob = model.predict_proba(scaled_data)
            
            print(f"Input Vector: {data}")
            print(f"Prediction Raw: {prediction[0]}")
            # Assuming class 1 is Rain based on typical binary classification
            prediction_label = "Yes" if prediction[0] == 1 else "No"
            print(f"Prediction Label: {prediction_label}")

            print(f"Probabilities: {pred_prob[0]}")
            if len(pred_prob[0]) > 1:
                print(f"Probability (No Rain - Class 0): {pred_prob[0][0]:.4f}")
                print(f"Probability (Rain - Class 1): {pred_prob[0][1]:.4f}")
        except Exception as e:
            print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_model()
