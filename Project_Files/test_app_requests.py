
import requests
import json

# Define the base URL
BASE_URL = "http://127.0.0.1:5000/predict"

# Test Case 1: Likely No Rain (Sunny)
# Based on verify_model.py and index.html structure
payload_sunny = {
    'Location': '2',          # Albury
    'MinTemp': '15.0',
    'MaxTemp': '30.0',
    'Rainfall': '0.0',
    'WindGustDir': '13',      # WNW
    'WindGustSpeed': '30.0',
    'WindDir9am': '13',       # WNW
    'WindDir3pm': '13',       # WNW
    'WindSpeed9am': '10.0',
    'WindSpeed3pm': '15.0',
    'Humidity9am': '40.0',
    'Humidity3pm': '25.0',
    'Pressure9am': '1020.0',
    'Pressure3pm': '1018.0',
    'Temp9am': '20.0',
    'Temp3pm': '28.0',
    'RainToday': '0'          # No
}

# Test Case 2: Likely Rain (Stormy)
payload_rainy = {
    'Location': '2',          # Albury
    'MinTemp': '10.0',
    'MaxTemp': '15.0',
    'Rainfall': '15.0',
    'WindGustDir': '0',       # N
    'WindGustSpeed': '60.0',
    'WindDir9am': '0',        # N
    'WindDir3pm': '0',        # N
    'WindSpeed9am': '20.0',
    'WindSpeed3pm': '25.0',
    'Humidity9am': '95.0',
    'Humidity3pm': '90.0',
    'Pressure9am': '1005.0',
    'Pressure3pm': '1002.0',
    'Temp9am': '11.0',
    'Temp3pm': '13.0',
    'RainToday': '1'          # Yes
}

def test_prediction(payload, case_name):
    print(f"\n--- Testing Scenario: {case_name} ---")
    try:
        response = requests.post(BASE_URL, data=payload)
        
        if response.status_code == 200:
            print("Request Successful (200 OK)")
            # Check if response content contains "Chance of Rain" or "No Chance"
            # Since app.py returns render_template, we look for keywords in HTML
            content = response.text
            if "Chance of Rain" in content or "Yes" in content:
                 # The 'chance.html' likely contains "Chance of Rain"
                 print("Result: Predicted Rain (Chance)")
            elif "No Chance" in content or "No" in content:
                 # The 'nochance.html' likely contains "No Chance"
                 print("Result: Predicted No Rain (No Chance)")
            else:
                 print("Result: Unknown (Could not parse HTML response)")
            
            # Print a snippet of the response for verification
            # print(content[:200]) # Optional
            
        else:
            print(f"Request Failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure app.py is running on port 5000.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction(payload_sunny, "Sunny/Dry (Expect No Rain)")
    test_prediction(payload_rainy, "Stormy/Wet (Expect Rain)")
