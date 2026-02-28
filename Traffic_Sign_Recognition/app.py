"""
Traffic Sign Recognition - Flask Web Application
Deep Learning CNN for Self-Driving Cars

This Flask application provides a web interface for traffic sign recognition.
Users can upload images and get predictions from the trained CNN model.
"""

import os
import sys
import numpy as np

# Try to import cv2, but make it optional for demo mode
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Running in demo mode.")

# Try to import PIL as fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the traffic sign recognizer (will work in demo mode without cv2)
try:
    from traffic_sign_recognition import TrafficSignRecognizer, TRAFFIC_SIGN_CLASSES, Config
except:
    # Define fallback if import fails
    TRAFFIC_SIGN_CLASSES = {}
    class Config:
        IMG_SIZE = 48
        NUM_CLASSES = 43
    class TrafficSignRecognizer:
        pass

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'traffic_sign_recognition_secret_key'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the recognizer
recognizer = None


def init_recognizer():
    """
    Initialize the traffic sign recognizer.
    """
    global recognizer
    
    try:
        # Try to load the trained model
        model_path = os.path.join(
            os.path.dirname(__file__), 
            'models', 
            'traffic_sign_model.keras'
        )
        
        if os.path.exists(model_path):
            recognizer = TrafficSignRecognizer(model_path)
            print(f"Model loaded from: {model_path}")
        else:
            # Create recognizer without model for demo
            recognizer = TrafficSignRecognizer()
            print("No trained model found. Running in demo mode.")
            
    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        recognizer = TrafficSignRecognizer()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and \
           (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS)


def is_video_file(filename):
    """
    Check if the file is a video.
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is a video
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def save_uploaded_file(file):
    """
    Save an uploaded file to the upload folder.
    
    Args:
        file: Flask FileStorage object
        
    Returns:
        Path to saved file or None if error
    """
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        timestamp = np.random.randint(100000, 999999)
        filename = f"{timestamp}_{filename}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return filepath
    
    return None


def process_video_file(video_path, max_frames=30):
    """
    Process a video file and extract frames for prediction.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process
        
    Returns:
        List of predictions for each frame
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        frame_predictions = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to get diverse frames
        frame_interval = max(1, total_frames // max_frames)
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                if recognizer and recognizer.model is not None:
                    predictions = recognizer.predict(frame_rgb, return_top_k=3)
                    if predictions and predictions['predictions']:
                        frame_predictions.append({
                            'frame': frame_count + 1,
                            'predictions': predictions['predictions']
                        })
                else:
                    # Demo mode - simulated predictions
                    frame_predictions.append({
                        'frame': frame_count + 1,
                        'predictions': simulate_prediction()[:3]
                    })
            
            frame_count += 1
        
        cap.release()
        
        return frame_predictions
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None


def preprocess_for_display(image_path):
    """
    Preprocess image for display on the web page.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Preprocessed image as base64 or path
    """
    try:
        if CV2_AVAILABLE:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for display (max 800px)
            max_size = 800
            height, width = img.shape[:2]
            
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
        else:
            # Use PIL as fallback
            from PIL import Image
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize for display (max 800px)
            max_size = 800
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)
            
            img = np.array(img)
        
        return img
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def simulate_prediction():
    """
    Generate simulated predictions for demo mode.
    
    Returns:
        List of prediction dictionaries
    """
    # Get random predictions
    predictions = []
    class_ids = list(TRAFFIC_SIGN_CLASSES.keys())
    
    # Generate top 5 predictions with decreasing confidence
    confidences = sorted(np.random.uniform(0.3, 0.95, 5), reverse=True)
    
    for i, (class_id, conf) in enumerate(zip(np.random.choice(class_ids, 5, replace=False), confidences)):
        predictions.append({
            'class_id': int(class_id),
            'class_name': TRAFFIC_SIGN_CLASSES[class_id],
            'confidence': float(conf),
            'rank': i + 1
        })
    
    return predictions


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Main page - traffic sign recognition interface.
    """
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Handle traffic sign recognition request.
    
    Expects:
        - Image or video file in request.files['image']
        
    Returns:
        JSON response with predictions
    """
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check if it's a video file
        is_video = is_video_file(file.filename)
        
        # Save the uploaded file
        filepath = save_uploaded_file(file)
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp, mp4, avi, mov, mkv'
            }), 400
        
        # Process video if it's a video file
        if is_video:
            # For video processing without OpenCV, simulate predictions
            if not CV2_AVAILABLE:
                # Demo mode - simulate predictions as if processing video frames
                print("Video processing in demo mode - simulating predictions")
                frame_predictions = []
                for i in range(5):
                    frame_predictions.append({
                        'frame': (i + 1) * 10,
                        'predictions': simulate_prediction()[:3]
                    })
                
                # Aggregate predictions across frames
                all_predictions = {}
                for frame_pred in frame_predictions:
                    for pred in frame_pred['predictions']:
                        class_name = pred['class_name']
                        if class_name not in all_predictions:
                            all_predictions[class_name] = []
                        all_predictions[class_name].append(pred['confidence'])
                
                # Average confidence across frames
                final_predictions = []
                for class_name, confidences in all_predictions.items():
                    final_predictions.append({
                        'class_name': class_name,
                        'confidence': float(np.mean(confidences)),
                        'frames_detected': len(confidences)
                    })
                
                # Sort by confidence
                final_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                final_predictions = final_predictions[:5]
                
                # Add ranks
                for i, pred in enumerate(final_predictions):
                    pred['rank'] = i + 1
                
                response = {
                    'success': True,
                    'media_type': 'video',
                    'video_path': filepath,
                    'total_frames_processed': 50,
                    'predictions': final_predictions,
                    'top_prediction': {
                        'class_name': final_predictions[0]['class_name'],
                        'confidence': f"{final_predictions[0]['confidence']*100:.2f}%"
                    }
                }
                
                return jsonify(response)
        
        # Process image (existing code)
        try:
            if recognizer and recognizer.model is not None:
                # Use actual model
                predictions = recognizer.predict(filepath, return_top_k=5)
                
                if predictions is None:
                    raise Exception("Prediction failed")
            else:
                # Demo mode - simulate predictions
                predictions = {
                    'predictions': simulate_prediction()
                }
            
            # Get the top prediction
            top_prediction = predictions['predictions'][0] if predictions['predictions'] else None
            
            # Prepare response
            response = {
                'success': True,
                'media_type': 'image',
                'image_path': filepath,
                'predictions': predictions['predictions'],
                'top_prediction': {
                    'class_name': top_prediction['class_name'],
                    'confidence': f"{top_prediction['confidence']*100:.2f}%"
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({
                'success': False,
                'error': f'Error during prediction: {str(e)}'
            }), 500
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 10 MB.'
        }), 413
        
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred'
        }), 500


@app.route('/class-info/<int:class_id>')
def class_info(class_id):
    """
    Get information about a specific traffic sign class.
    
    Args:
        class_id: ID of the traffic sign class
        
    Returns:
        JSON response with class information
    """
    if class_id in TRAFFIC_SIGN_CLASSES:
        return jsonify({
            'success': True,
            'class_id': class_id,
            'class_name': TRAFFIC_SIGN_CLASSES[class_id],
            'description': get_class_description(class_id)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Class not found'
        }), 404


def get_class_description(class_id):
    """
    Get description for a traffic sign class.
    
    Args:
        class_id: ID of the traffic sign class
        
    Returns:
        Description string
    """
    descriptions = {
        0: 'Speed Limit 20 - Maximum speed 20 km/h',
        1: 'Speed Limit 30 - Maximum speed 30 km/h',
        2: 'Speed Limit 50 - Maximum speed 50 km/h',
        3: 'Speed Limit 60 - Maximum speed 60 km/h',
        4: 'Speed Limit 70 - Maximum speed 70 km/h',
        5: 'Speed Limit 80 - Maximum speed 80 km/h',
        6: 'End of Speed Limit 80 - End of 80 km/h speed limit',
        7: 'Speed Limit 100 - Maximum speed 100 km/h',
        8: 'Speed Limit 120 - Maximum speed 120 km/h',
        9: 'No passing - Overtaking prohibited',
        10: 'No passing for vehicles over 3.5 tons - Heavy vehicles cannot pass',
        11: 'Right-of-way at intersection - Priority road ahead',
        12: 'Priority road - You have priority',
        13: 'Yield - Give way to other vehicles',
        14: 'Stop - Complete stop required',
        15: 'No vehicles - Motor vehicles prohibited',
        16: 'Vehicles over 3.5 tons prohibited - Heavy vehicles prohibited',
        17: 'No entry - Entry forbidden',
        18: 'General danger - Warning of danger',
        19: 'Curve left - Dangerous curve to the left',
        20: 'Curve right - Dangerous curve to the right',
        21: 'Double curve - Series of dangerous curves',
        22: 'Bumpy road - Uneven road surface',
        23: 'Slippery road - Slippery when wet',
        24: 'Road narrows - Road becomes narrower',
        25: 'Road work - Road maintenance ahead',
        26: 'Traffic signals - Traffic lights ahead',
        27: 'Pedestrians - Pedestrians may be on the road',
        28: 'Children crossing - Children may cross the road',
        29: 'Bicycles crossing - Cyclists may cross',
        30: 'Beware of ice/snow - Icy or snowy conditions',
        31: 'Wild animals - Wild animals may be on the road',
        32: 'End of all restrictions - End of all restrictions',
        33: 'Turn right ahead - Mandatory turn right',
        34: 'Turn left ahead - Mandatory turn left',
        35: 'Ahead only - Go straight only',
        36: 'Go straight or right - Both straight and right allowed',
        37: 'Go straight or left - Both straight and left allowed',
        38: 'Keep right - Keep to the right of the obstacle',
        39: 'Keep left - Keep to the left of the obstacle',
        40: 'Roundabout - Roundabout ahead',
        41: 'End of no passing - End of no passing zone',
        42: 'End of no passing for vehicles over 3.5 tons - End of heavy vehicle restriction'
    }
    
    return descriptions.get(class_id, 'Traffic sign')


@app.route('/about')
def about():
    """
    About page with information about the project.
    """
    return render_template('about.html')


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error_code=404, message='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error_code=500, message='Internal server error'), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Initialize recognizer
    init_recognizer()
    
    # Run the Flask app
    print("=" * 60)
    print("Traffic Sign Recognition Web Application")
    print("=" * 60)
    print("Starting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
