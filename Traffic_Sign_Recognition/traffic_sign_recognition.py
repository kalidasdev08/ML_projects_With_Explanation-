"""
Traffic Sign Recognition - Model Inference
Deep Learning CNN for Self-Driving Cars

This script provides functionality to:
- Load a trained CNN model
- Preprocess input images
- Predict traffic sign classes
- Display results with confidence scores
"""

import os
import numpy as np

# Try to import cv2, but make it optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Demo mode only.")

# Try to import TensorFlow, but make it optional
try:
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    pd = None
    keras = None
    print("Warning: TensorFlow not installed. Demo mode only.")

try:
    import pandas as pd
except:
    pd = None
    print("Warning: Pandas not installed.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model and data paths
    MODEL_PATH = 'models/traffic_sign_model.keras'
    CLASS_NAMES_PATH = 'models/class_names.csv'
    
    # Image parameters
    IMG_SIZE = 48
    NUM_CHANNELS = 3
    NUM_CLASSES = 43


# ============================================================================
# TRAFFIC SIGN CLASS NAMES
# ============================================================================

# GTSRB traffic sign class names
TRAFFIC_SIGN_CLASSES = {
    0: 'Speed Limit 20',
    1: 'Speed Limit 30',
    2: 'Speed Limit 50',
    3: 'Speed Limit 60',
    4: 'Speed Limit 70',
    5: 'Speed Limit 80',
    6: 'End of Speed Limit 80',
    7: 'Speed Limit 100',
    8: 'Speed Limit 120',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 tons prohibited',
    17: 'No entry',
    18: 'General danger',
    19: 'Curve left',
    20: 'Curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals',
    32: 'End of all restrictions',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout',
    41: 'End of no passing',
    42: 'End of no passing for vehicles over 3.5 tons'
}


# ============================================================================
# TRAFFIC SIGN RECOGNIZER CLASS
# ============================================================================

class TrafficSignRecognizer:
    """
    Traffic Sign Recognition model wrapper.
    
    Provides methods to:
    - Load a trained model
    - Preprocess images
    - Make predictions
    - Display results
    """
    
    def __init__(self, model_path=None, class_names_path=None):
        """
        Initialize the traffic sign recognizer.
        
        Args:
            model_path: Path to the trained Keras model
            class_names_path: Path to class names CSV (optional)
        """
        self.model = None
        self.class_names = TRAFFIC_SIGN_CLASSES.copy()
        self.img_size = Config.IMG_SIZE
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        
        # Load custom class names if provided
        if class_names_path and os.path.exists(class_names_path):
            self.load_class_names(class_names_path)
    
    def load_model(self, model_path):
        """
        Load a trained Keras model.
        
        Args:
            model_path: Path to the model file
        """
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Please train the model first using train_model.py")
            return False
        
        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_class_names(self, class_names_path):
        """
        Load class names from a CSV file.
        
        Args:
            class_names_path: Path to class names CSV
        """
        try:
            df = pd.read_csv(class_names_path)
            for _, row in df.iterrows():
                self.class_names[row['class_id']] = row['class_name']
            print(f"Loaded {len(self.class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction.
        
        Args:
            image: Input image (numpy array or file path)
            
        Returns:
            Preprocessed image array
        """
        # Load image if path provided
        if isinstance(image, str):
            if CV2_AVAILABLE:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Use PIL as fallback
                from PIL import Image
                image = Image.open(image).convert('RGB')
                image = np.array(image)
        
        # Resize to model input size
        if CV2_AVAILABLE:
            image = cv2.resize(image, (self.img_size, self.img_size))
        else:
            from PIL import Image
            image = Image.fromarray(image).resize((self.img_size, self.img_size))
            image = np.array(image)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image, return_top_k=5):
        """
        Predict traffic sign class for an image.
        
        Args:
            image: Input image (numpy array or file path)
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            print("Error: Model not loaded. Please load a model first.")
            return None
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        # Get top k predictions
        top_k_indices = np.argsort(predictions)[::-1][:return_top_k]
        
        results = {
            'predictions': [],
            'success': True
        }
        
        for i, idx in enumerate(top_k_indices):
            results['predictions'].append({
                'class_id': int(idx),
                'class_name': self.class_names.get(idx, f'Unknown ({idx})'),
                'confidence': float(predictions[idx]),
                'rank': i + 1
            })
        
        return results
    
    def predict_single(self, image_path):
        """
        Predict traffic sign class and return top prediction.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with top prediction
        """
        results = self.predict(image_path, return_top_k=1)
        
        if results and results['predictions']:
            return results['predictions'][0]
        
        return None
    
    def predict_batch(self, image_paths):
        """
        Predict traffic sign classes for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            prediction = self.predict_single(img_path)
            results.append({
                'image_path': img_path,
                'prediction': prediction
            })
        
        return results
    
    def visualize_prediction(self, image, predictions=None):
        """
        Visualize the prediction on the image.
        
        Args:
            image: Input image
            predictions: Prediction results (if None, will predict)
            
        Returns:
            Image with prediction visualization
    """
        # Make prediction if not provided
        if predictions is None:
            predictions = self.predict(image)
        
        if not predictions:
            return image
        
        # Convert to RGB if needed
        if isinstance(image, str):
            if CV2_AVAILABLE:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                from PIL import Image
                image = Image.open(image).convert('RGB')
                image = np.array(image)
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw prediction info on image
        height, width = vis_image.shape[:2]
        
        if CV2_AVAILABLE:
            # Add a bar at the bottom showing predictions
            bar_height = 80
            vis_image = np.vstack([
                vis_image, 
                np.ones((bar_height, width, 3), dtype=np.uint8) * 255
            ])
            
            # Draw top 3 predictions
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = height + 25
            
            for i, pred in enumerate(predictions['predictions'][:3]):
                text = f"{pred['class_name']}: {pred['confidence']*100:.1f}%"
                cv2.putText(vis_image, text, (10, y_offset + i * 25), 
                           font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # Use PIL for text drawing
            from PIL import Image, ImageDraw, ImageFont
            
            # Add a bar at the bottom
            bar_height = 80
            vis_image_pil = Image.fromarray(vis_image)
            white_bar = Image.new('RGB', (width, bar_height), (255, 255, 255))
            vis_image_pil = Image.new('RGB', (width, height + bar_height))
            vis_image_pil.paste(Image.fromarray(vis_image), (0, 0))
            vis_image_pil.paste(white_bar, (0, height))
            
            draw = ImageDraw.Draw(vis_image_pil)
            y_offset = height + 25
            
            for i, pred in enumerate(predictions['predictions'][:3]):
                text = f"{pred['class_name']}: {pred['confidence']*100:.1f}%"
                draw.text((10, y_offset + i * 25), text, fill=(0, 0, 0))
            
            vis_image = np.array(vis_image_pil)
        
        return vis_image
    
    def get_model_summary(self):
        """
        Get model architecture summary.
        """
        if self.model is None:
            print("No model loaded")
            return
        
        return self.model.summary()


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_recognition():
    """
    Demo function to demonstrate traffic sign recognition.
    """
    print("=" * 60)
    print("Traffic Sign Recognition - Demo")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = TrafficSignRecognizer()
    
    # Try to load model
    model_loaded = recognizer.load_model(Config.MODEL_PATH)
    
    if not model_loaded:
        print("\nGenerating a demo prediction with random data...")
        # Create a synthetic image for demo
        demo_image = np.random.randint(0, 255, (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        
        # For demo, simulate prediction
        print("\nSimulated prediction (model not trained yet):")
        for i in range(5):
            class_id = np.random.randint(0, Config.NUM_CLASSES)
            confidence = np.random.uniform(0.5, 0.99)
            print(f"  {i+1}. {TRAFFIC_SIGN_CLASSES[class_id]}: {confidence*100:.1f}%")
        
        return
    
    # Test with synthetic image
    print("\nTesting with sample image...")
    test_image = np.random.randint(0, 255, (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
    
    results = recognizer.predict(test_image)
    
    print("\nPrediction Results:")
    print("-" * 40)
    for pred in results['predictions']:
        print(f"  {pred['rank']}. {pred['class_name']}")
        print(f"     Confidence: {pred['confidence']*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def recognize_from_camera(camera_index=0):
    """
    Real-time traffic sign recognition from camera.
    
    Args:
        camera_index: Camera device index
    """
    print("=" * 60)
    print("Traffic Sign Recognition - Camera Mode")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = TrafficSignRecognizer()
    recognizer.load_model(Config.MODEL_PATH)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Make prediction on frame
        results = recognizer.predict(frame)
        
        # Draw prediction on frame
        if results and results['predictions']:
            top_pred = results['predictions'][0]
            
            # Draw text
            text = f"{top_pred['class_name']}: {top_pred['confidence']*100:.1f}%"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display frame
        cv2.imshow('Traffic Sign Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--camera':
            recognize_from_camera()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python traffic_sign_recognition.py           - Run demo")
            print("  python traffic_sign_recognition.py --camera   - Run camera mode")
        else:
            # Assume it's an image path
            image_path = sys.argv[1]
            recognizer = TrafficSignRecognizer()
            recognizer.load_model(Config.MODEL_PATH)
            
            result = recognizer.predict_single(image_path)
            if result:
                print(f"\nPredicted: {result['class_name']}")
                print(f"Confidence: {result['confidence']*100:.2f}%")
    else:
        demo_recognition()
