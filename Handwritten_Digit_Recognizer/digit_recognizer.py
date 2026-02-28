"""
Handwritten Digit Recognizer - Prediction image preprocessing and digit Module
Handles prediction using the trained CNN model.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw

class DigitRecognizer:
    """Class for recognizing handwritten digits from images."""
    
    def __init__(self, model_path=None):
        """Initialize the digit recognizer with a trained model."""
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                'models', 
                'digit_model.h5'
            )
        
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        print("Model loaded successfully!")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction.
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # Convert to numpy array if needed
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert('L')
            image = np.array(image)
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image.convert('L'))
        elif isinstance(image, np.ndarray):
            # Ensure grayscale - handle different array shapes
            if len(image.shape) == 3:
                # Color image - convert to grayscale
                pil_image = Image.fromarray(image).convert('L')
                image = np.array(pil_image)
        
        # Resize to 28x28 if needed
        if image.shape != (28, 28):
            pil_image = Image.fromarray(image.astype('uint8'))
            pil_image = pil_image.resize((28, 28), Image.LANCZOS)
            image = np.array(pil_image)
        
        # Invert colors if needed (white text on black background)
        # MNIST has white digits on black background
        if np.mean(image) > 127:
            image = 255 - image
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Reshape to include channel dimension
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def preprocess_canvas_data(self, canvas_data_url):
        """
        Preprocess base64 encoded canvas data.
        
        Args:
            canvas_data_url: Base64 encoded image data URL
            
        Returns:
            Preprocessed image ready for model prediction
        """
        import base64
        from io import BytesIO
        
        # Remove data URL prefix
        if ',' in canvas_data_url:
            canvas_data_url = canvas_data_url.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(canvas_data_url)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data)).convert('L')
        
        # Get image statistics to determine if we need to invert
        img_array = np.array(image)
        mean_val = np.mean(img_array)
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array
        image = np.array(image)
        
        # Invert colors if needed (white text on black background or vice versa)
        # MNIST has white digits on black background
        if mean_val > 127:
            # Canvas has white background, invert to get white digits on black
            image = 255 - image
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Reshape to include channel dimension
        image = image.reshape(1, 28, 28, 1)
        
        return image
        
        # Reshape to include channel dimension
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def predict(self, image):
        """
        Predict the digit in an image.
        
        Args:
            image: PIL Image, numpy array, file path, or base64 canvas data
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        if isinstance(image, str) and image.startswith('data:image'):
            # Base64 encoded canvas data
            processed_image = self.preprocess_canvas_data(image)
        else:
            processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get the predicted digit and confidence
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get probabilities for all digits
        digit_probs = {i: float(prob) for i, prob in enumerate(predictions[0])}
        
        return {
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': digit_probs
        }
    
    def predict_top_k(self, image, k=3):
        """
        Predict the top k digits with highest probabilities.
        
        Args:
            image: Input image
            k: Number of top predictions to return
            
        Returns:
            List of top k predictions with their probabilities
        """
        result = self.predict(image)
        
        # Sort probabilities in descending order
        sorted_probs = sorted(
            result['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top k predictions
        return [
            {'digit': digit, 'probability': prob} 
            for digit, prob in sorted_probs[:k]
        ]


def load_model_check(model_path=None):
    """Load and verify the model is working."""
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(base_dir, 'models', 'digit_model.h5'),
            os.path.join(base_dir, '..', 'Handwritten Digit Recognizer', 'models', 'digit_model.h5'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please run train_model.py first to train the model.")
        return None
    
    try:
        recognizer = DigitRecognizer(model_path)
        print("Model loaded and verified successfully!")
        return recognizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    # Test the recognizer
    recognizer = load_model_check()
    if recognizer:
        print("\nModel is ready for predictions!")
