"""
Disease Predictor Module
========================
This module handles the disease prediction logic using the trained model.
"""

import pickle
import os
import numpy as np

class DiseasePredictor:
    """
    A class to handle disease prediction from symptoms.
    """
    
    def __init__(self, model_path='models/disease_model.pkl', 
                 symptom_list_path='models/symptom_list.pkl',
                 symptom_desc_path='models/symptom_descriptions.pkl'):
        """
        Initialize the disease predictor with trained model.
        
        Args:
            model_path: Path to the trained model
            symptom_list_path: Path to the symptom list
            symptom_desc_path: Path to symptom descriptions
        """
        self.model = None
        self.symptom_list = []
        self.symptom_descriptions = {}
        
        # Load model and related files
        self._load_model(model_path, symptom_list_path, symptom_desc_path)
        
    def _load_model(self, model_path, symptom_list_path, symptom_desc_path):
        """
        Load the trained model and related data.
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            
            # Load symptom list
            with open(symptom_list_path, 'rb') as f:
                self.symptom_list = pickle.load(f)
            print(f"Symptom list loaded: {len(self.symptom_list)} symptoms")
            
            # Load symptom descriptions
            with open(symptom_desc_path, 'rb') as f:
                self.symptom_descriptions = pickle.load(f)
            print("Symptom descriptions loaded")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found. {e}")
            print("Please run train_model.py first to create the model.")
            raise
            
    def predict_disease(self, symptoms):
        """
        Predict disease from given symptoms.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            dict: Prediction result with disease and confidence
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Convert symptoms to feature vector
        features = self._symptoms_to_features(symptoms)
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities) * 100
        except:
            confidence = None
            
        # Get top 3 predictions
        top_predictions = self._get_top_predictions(features)
        
        return {
            "primary_disease": prediction,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "input_symptoms": symptoms
        }
    
    def _symptoms_to_features(self, symptoms):
        """
        Convert symptom strings to binary feature vector.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            list: Binary feature vector
        """
        # Normalize symptoms
        symptoms_lower = [s.lower().strip() for s in symptoms]
        
        # Create binary features
        features = []
        for symptom in self.symptom_list:
            # Check if symptom is in input (partial match)
            symptom_found = False
            for input_symptom in symptoms_lower:
                if symptom in input_symptom or input_symptom in symptom:
                    symptom_found = True
                    break
            features.append(1 if symptom_found else 0)
        
        return features
    
    def _get_top_predictions(self, features, top_n=3):
        """
        Get top N disease predictions.
        
        Args:
            features: Feature vector
            top_n: Number of top predictions to return
            
        Returns:
            list: List of top predictions with probabilities
        """
        try:
            probabilities = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            
            # Get indices of top predictions
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            
            top_predictions = []
            for idx in top_indices:
                if probabilities[idx] > 0.01:  # Only include if > 1% probability
                    top_predictions.append({
                        "disease": classes[idx],
                        "probability": round(probabilities[idx] * 100, 2)
                    })
            
            return top_predictions
        except:
            return []
    
    def get_available_symptoms(self):
        """
        Get list of all available symptoms.
        
        Returns:
            list: List of symptom names
        """
        return self.symptom_list
    
    def get_symptom_description(self, symptom):
        """
        Get description for a symptom.
        
        Args:
            symptom: Symptom name
            
        Returns:
            str: Symptom description
        """
        return self.symptom_descriptions.get(symptom, "Description not available")
    
    def get_chatbot_response(self, user_message):
        """
        Generate chatbot response for user queries.
        
        Args:
            user_message: User's message
            
        Returns:
            str: Chatbot response
        """
        user_message = user_message.lower().strip()
        
        # Greeting patterns
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if any(greet in user_message for greet in greetings):
            return ("Hello! I'm your Medical Disease Prediction Assistant. "
                    "I can help you identify potential diseases based on your symptoms. "
                    "Please list the symptoms you're experiencing, and I'll provide "
                    "a prediction of possible diseases.\n\n"
                    "For example, you can say: 'I have fever and cough' or "
                    "'I'm experiencing headache and fatigue'.")
        
        # Help pattern
        if 'help' in user_message or 'how' in user_message:
            return ("I can help you predict potential diseases based on your symptoms. "
                    "Here's how to use me:\n\n"
                    "1. Describe your symptoms (e.g., 'fever, cough, headache')\n"
                    "2. I'll analyze your symptoms and suggest possible conditions\n"
                    "3. You can ask about specific symptoms for more details\n\n"
                    "Remember: This is for preliminary guidance only. "
                    "Please consult a healthcare professional for proper diagnosis.")
        
        # List symptoms pattern
        if 'symptom' in user_message and ('list' in user_message or 'available' in user_message or 'what' in user_message):
            symptoms_text = "Here are the symptoms I can analyze:\n\n"
            for i, symptom in enumerate(self.symptom_list, 1):
                desc = self.symptom_descriptions.get(symptom, '')
                symptoms_text += f"{i}. {symptom.replace('_', ' ').title()}"
                if desc:
                    symptoms_text += f" - {desc}"
                symptoms_text += "\n"
            return symptoms_text
        
        # Ask about specific symptom
        if 'what is' in user_message or 'tell me about' in user_message:
            for symptom in self.symptom_list:
                if symptom.replace('_', ' ') in user_message:
                    desc = self.symptom_descriptions.get(symptom, 'Description not available')
                    return f"{symptom.replace('_', ' ').title()}: {desc}"
        
        # Thank you pattern
        if 'thank' in user_message:
            return "You're welcome! If you have any more symptoms to report, feel free to share them. Take care!"
        
        # Goodbye pattern
        if 'bye' in user_message or 'goodbye' in user_message:
            return "Goodbye! Remember to consult a healthcare professional for proper medical advice. Stay healthy!"
        
        # Try to interpret as symptoms and make prediction
        return None  # Return None to indicate should make prediction
    
    def format_prediction_response(self, prediction):
        """
        Format prediction result as a friendly response.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            str: Formatted response
        """
        if "error" in prediction:
            return f"Error: {prediction['error']}"
        
        response = "📋 **Disease Prediction Result**\n\n"
        
        # Primary prediction
        response += f"🏥 **Primary Prediction:** {prediction['primary_disease']}\n"
        
        if prediction['confidence']:
            response += f"📊 **Confidence:** {prediction['confidence']:.1f}%\n"
        
        # Top predictions
        if prediction['top_predictions']:
            response += "\n**Possible Conditions:**\n"
            for i, pred in enumerate(prediction['top_predictions'], 1):
                response += f"  {i}. {pred['disease']} ({pred['probability']}%)\n"
        
        # Input symptoms
        response += f"\n**Symptoms Analyzed:** {', '.join(prediction['input_symptoms'])}\n"
        
        # Disclaimer
        response += "\n⚠️ **Disclaimer:** This prediction is based on machine learning "
        response += "and should be used for preliminary guidance only. "
        response += "Please consult a healthcare professional for proper diagnosis and treatment."
        
        return response


def load_predictor():
    """
    Load and return the disease predictor.
    
    Returns:
        DiseasePredictor: Loaded predictor instance
    """
    try:
        predictor = DiseasePredictor()
        return predictor
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return None


if __name__ == "__main__":
    # Test the predictor
    predictor = load_predictor()
    if predictor:
        print("\nAvailable symptoms:")
        print(predictor.get_available_symptoms())
        
        # Test prediction
        test_symptoms = ['fever', 'cough', 'fatigue']
        result = predictor.predict_disease(test_symptoms)
        print(f"\nPrediction for {test_symptoms}:")
        print(predictor.format_prediction_response(result))
