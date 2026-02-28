"""
House Price Prediction - Flask Web Application
A web interface for predicting house prices using Linear Regression.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

app = Flask(__name__, 
            template_folder=os.path.join(current_dir, 'templates'))

# Global predictor instance
predictor = None


def load_model():
    """Load the trained model and preprocessing objects."""
    global predictor
    
    # Get the directory where app.py is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(app_dir, 'models')
    
    print(f"App directory: {app_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {os.path.exists(models_dir)}")
    
    try:
        # Import here to avoid circular import
        from house_price_predictor import HousePricePredictor
        
        model_path = os.path.join(models_dir, 'house_price_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
        
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        predictor = HousePricePredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
            feature_names_path=feature_names_path
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have run train_model.py first!")


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request.
    
    Expected JSON payload:
    {
        "location": "Urban|Suburban|Rural",
        "size_sqft": <number>,
        "bedrooms": <number>,
        "age_years": <number>,
        "amenities": <number>
    }
    """
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        location = data.get('location')
        size_sqft = float(data.get('size_sqft'))
        bedrooms = int(data.get('bedrooms'))
        age_years = int(data.get('age_years'))
        amenities = int(data.get('amenities'))
        
        # Validate inputs
        if not location:
            return jsonify({
                'success': False,
                'error': 'Location is required'
            }), 400
        
        valid_locations = ['Urban', 'Suburban', 'Rural']
        if location not in valid_locations:
            return jsonify({
                'success': False,
                'error': f'Invalid location. Must be one of: {valid_locations}'
            }), 400
        
        # Make prediction
        predicted_price = predictor.predict(
            location=location,
            size_sqft=size_sqft,
            bedrooms=bedrooms,
            age_years=age_years,
            amenities=amenities
        )
        
        # Convert to INR (1 USD ≈ 83 INR)
        usd_to_inr = 83.0
        predicted_price_inr = predicted_price * usd_to_inr
        
        # Get detailed prediction
        details = predictor.predict_with_details(
            location=location,
            size_sqft=size_sqft,
            bedrooms=bedrooms,
            age_years=age_years,
            amenities=amenities
        )
        
        return jsonify({
            'success': True,
            'prediction': {
                'price': round(predicted_price, 2),
                'formatted_price': f"${predicted_price:,.2f}",
                'price_inr': round(predicted_price_inr, 2),
                'formatted_price_inr': f"₹{predicted_price_inr:,.2f}"
            },
            'input': {
                'location': location,
                'size_sqft': size_sqft,
                'bedrooms': bedrooms,
                'age_years': age_years,
                'amenities': amenities
            },
            'breakdown': details['feature_contributions']
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model."""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/sample-predictions', methods=['GET'])
def sample_predictions():
    """Get sample predictions for demonstration."""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        samples = [
            {'location': 'Urban', 'size_sqft': 1500, 'bedrooms': 3, 'age_years': 10, 'amenities': 2},
            {'location': 'Suburban', 'size_sqft': 1800, 'bedrooms': 4, 'age_years': 5, 'amenities': 3},
            {'location': 'Rural', 'size_sqft': 1200, 'bedrooms': 2, 'age_years': 20, 'amenities': 1},
            {'location': 'Urban', 'size_sqft': 2500, 'bedrooms': 5, 'age_years': 2, 'amenities': 5},
            {'location': 'Suburban', 'size_sqft': 2000, 'bedrooms': 4, 'age_years': 8, 'amenities': 4},
        ]
        
        usd_to_inr = 83.0
        
        predictions = []
        for sample in samples:
            price = predictor.predict(
                location=sample['location'],
                size_sqft=sample['size_sqft'],
                bedrooms=sample['bedrooms'],
                age_years=sample['age_years'],
                amenities=sample['amenities']
            )
            price_inr = price * usd_to_inr
            predictions.append({
                'input': sample,
                'price': round(price, 2),
                'formatted_price': f"${price:,.2f}",
                'price_inr': round(price_inr, 2),
                'formatted_price_inr': f"₹{price_inr:,.2f}"
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    print("\n" + "="*60)
    print("Starting House Price Prediction Web Application")
    print("="*60)
    print("\nOpen your browser and navigate to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
