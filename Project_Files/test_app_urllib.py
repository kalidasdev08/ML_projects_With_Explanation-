
import urllib.request
import urllib.parse
import json
import re

# Define the base URL
BASE_URL = "http://127.0.0.1:5000/predict"

# Test Case 1: Likely No Rain (Sunny)
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
        data = urllib.parse.urlencode(payload).encode()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(BASE_URL, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("Request Successful (200 OK)")
                content = response.read().decode('utf-8')
                
                if "Rain Expected!" in content:
                     print("Result: Predicted Rain (Chance)")
                elif "No chances of rain" in content:
                     print("Result: Predicted No Rain (No Chance)")
                else:
                     print("Result: Unknown (Could not parse HTML response)")
                
                # Extract Probability
                prob_match = re.search(r"Probability.*: (0\.\d+)", content)
                if prob_match:
                    print(f"Verified Probability in HTML: {prob_match.group(1)}")
                else:
                    print("Warning: Probability not found in HTML response")
                
            else:
                print(f"Request Failed: {response.status}")
            
    except urllib.error.URLError as e:
        print(f"Error: Could not connect to the server. Make sure app.py is running on port 5000. Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction(payload_sunny, "Sunny/Dry (Expect No Rain)")
    test_prediction(payload_rainy, "Stormy/Wet (Expect Rain)")
