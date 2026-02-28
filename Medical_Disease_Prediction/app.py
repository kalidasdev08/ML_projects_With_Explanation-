"""
Medical Disease Prediction Web Application
==========================================
Flask web application with chatbot interface for disease prediction from symptoms.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from disease_predictor import DiseasePredictor

app = Flask(__name__)

# Initialize the predictor
predictor = None

def init_predictor():
    """
    Initialize the disease predictor.
    """
    global predictor
    try:
        predictor = DiseasePredictor(
            model_path='models/disease_model.pkl',
            symptom_list_path='models/symptom_list.pkl',
            symptom_desc_path='models/symptom_descriptions.pkl'
        )
        print("Disease predictor initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print("Please run train_model.py first to create the model.")
        return False

@app.route('/')
def index():
    """
    Main page - renders the web interface.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle disease prediction requests.
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500
    
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided. Please enter your symptoms.'
            }), 400
        
        # Make prediction
        prediction = predictor.predict_disease(symptoms)
        
        # Format response
        response_text = predictor.format_prediction_response(prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'response': response_text
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chatbot messages.
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'error': 'No message provided.'
            }), 400
        
        # Check if it's a symptom description
        chatbot_response = predictor.get_chatbot_response(user_message)
        
        # If chatbot can't handle it, try to interpret as symptoms
        if chatbot_response is None:
            # Extract symptoms from message
            symptoms = extract_symptoms_from_message(user_message)
            
            if symptoms:
                prediction = predictor.predict_disease(symptoms)
                chatbot_response = predictor.format_prediction_response(prediction)
            else:
                chatbot_response = ("I couldn't identify specific symptoms in your message. "
                                  "Please describe your symptoms more clearly, for example: "
                                  "'I have fever and cough' or 'I'm experiencing headache and fatigue'. "
                                  "You can also type 'list symptoms' to see all symptoms I can analyze.")
        
        return jsonify({
            'success': True,
            'response': chatbot_response
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Chat error: {str(e)}'
        }), 500

def extract_symptoms_from_message(message):
    """
    Extract symptoms from a natural language message.
    
    Args:
        message: User's message
        
    Returns:
        list: List of identified symptoms
    """
    message_lower = message.lower()
    available_symptoms = predictor.get_available_symptoms()
    
    identified_symptoms = []
    
    for symptom in available_symptoms:
        # Check for symptom keywords
        symptom_keywords = symptom.replace('_', ' ').split()
        for keyword in symptom_keywords:
            if keyword in message_lower:
                identified_symptoms.append(symptom)
                break
    
    return identified_symptoms

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """
    Get list of all available symptoms.
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded.'
        }), 500
    
    try:
        symptoms = predictor.get_available_symptoms()
        descriptions = {}
        
        for symptom in symptoms:
            descriptions[symptom] = predictor.get_symptom_description(symptom)
        
        return jsonify({
            'success': True,
            'symptoms': symptoms,
            'descriptions': descriptions
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

if __name__ == '__main__':
    # Initialize predictor
    if init_predictor():
        # Run the app
        print("\n" + "="*60)
        print("Starting Medical Disease Prediction App")
        print("="*60)
        print("\nOpen your browser and navigate to: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nFailed to initialize the predictor.")
        print("Please ensure you have run train_model.py first.")
        sys.exit(1)
