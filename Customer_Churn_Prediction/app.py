"""
Customer Churn Prediction Web Application
Flask app for predicting customer churn
"""

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'churn_model.pkl')

def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None

model_data = load_model()

def predict_churn(customer_data):
    """Predict churn for a single customer"""
    if model_data is None:
        return None, "Model not loaded"
    
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    categorical_cols = model_data['categorical_cols']
    numerical_cols = model_data['numerical_cols']
    
    # Create DataFrame from input data
    df = pd.DataFrame([customer_data])
    
    # Get column order from feature_names
    col_order = feature_names
    df = df[col_order]
    
    # Get the columns the scaler was actually fit on
    scaler_cols = list(model_data['scaler'].feature_names_in_)
    
    # Encode categorical variables (keep track of indices)
    cat_indices = []
    for i, col in enumerate(col_order):
        if col in categorical_cols:
            le = label_encoders[col]
            df[col] = le.transform([str(df[col].iloc[0])])[0]
            cat_indices.append(i)
    
    # Get numerical indices (only columns that scaler was fit on)
    num_indices = [col_order.index(col) for col in scaler_cols if col in col_order]
    
    # Convert to numpy array
    X = df.values.astype(float)
    
    # Scale numerical features only (using only the columns scaler was fit on)
    X[:, num_indices] = scaler.transform(X[:, num_indices])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return prediction, probability

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model_data is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    try:
        # Get customer data from request
        data = request.get_json()
        
        # Prepare customer data for prediction
        customer_data = {
            'gender': data.get('gender'),
            'city_tier': data.get('city_tier'),
            'subscription_type': data.get('subscription_type'),
            'contract_length': data.get('contract_length'),
            'age': float(data.get('age', 0)),
            'tenure_months': float(data.get('tenure_months', 0)),
            'monthly_charge': float(data.get('monthly_charge', 0)),
            'total_charges': float(data.get('total_charges', 0)),
            'payment_method': data.get('payment_method'),
            'num_support_tickets': float(data.get('num_support_tickets', 0)),
            'num_complaints': float(data.get('num_complaints', 0)),
            'product_usage_score': float(data.get('product_usage_score', 0)),
            'login_frequency': float(data.get('login_frequency', 0)),
            'last_purchase_days_ago': float(data.get('last_purchase_days_ago', 0)),
            'avg_session_duration': float(data.get('avg_session_duration', 0)),
            'is_premium_member': int(data.get('is_premium_member', 0)),
            'has_Referral': int(data.get('has_referral', 0)),
            'engagement_score': float(data.get('engagement_score', 0)),
            'satisfaction_rating': float(data.get('satisfaction_rating', 0))
        }
        
        # Make prediction
        prediction, probability = predict_churn(customer_data)
        
        if prediction is None:
            return jsonify({'error': probability}), 500
        
        # Prepare response
        churn_prediction = "Yes" if prediction == 1 else "No"
        churn_prob = float(probability[1]) * 100
        not_churn_prob = float(probability[0]) * 100
        
        return jsonify({
            'prediction': churn_prediction,
            'churn_probability': round(churn_prob, 2),
            'not_churn_probability': round(not_churn_prob, 2),
            'risk_level': 'High' if churn_prob > 70 else ('Medium' if churn_prob > 40 else 'Low')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction request"""
    if model_data is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        customers = data.get('customers', [])
        
        results = []
        for customer in customers:
            customer_data = {
                'gender': customer.get('gender'),
                'city_tier': customer.get('city_tier'),
                'subscription_type': customer.get('subscription_type'),
                'contract_length': customer.get('contract_length'),
                'age': float(customer.get('age', 0)),
                'tenure_months': float(customer.get('tenure_months', 0)),
                'monthly_charge': float(customer.get('monthly_charge', 0)),
                'total_charges': float(customer.get('total_charges', 0)),
                'payment_method': customer.get('payment_method'),
                'num_support_tickets': float(customer.get('num_support_tickets', 0)),
                'num_complaints': float(customer.get('num_complaints', 0)),
                'product_usage_score': float(customer.get('product_usage_score', 0)),
                'login_frequency': float(customer.get('login_frequency', 0)),
                'last_purchase_days_ago': float(customer.get('last_purchase_days_ago', 0)),
                'avg_session_duration': float(customer.get('avg_session_duration', 0)),
                'is_premium_member': int(customer.get('is_premium_member', 0)),
                'has_Referral': int(customer.get('has_referral', 0)),
                'engagement_score': float(customer.get('engagement_score', 0)),
                'satisfaction_rating': float(customer.get('satisfaction_rating', 0))
            }
            
            prediction, probability = predict_churn(customer_data)
            
            results.append({
                'customer_id': customer.get('customer_id', 'Unknown'),
                'prediction': "Yes" if prediction == 1 else "No",
                'churn_probability': round(float(probability[1]) * 100, 2)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature_importance')
def feature_importance():
    """Return feature importance data"""
    if model_data is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    try:
        importance_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models', 'feature_importance.csv'))
        return jsonify(importance_df.to_dict(orient='records'))
    except FileNotFoundError:
        # Calculate from model if file not found
        model = model_data['model']
        feature_names = model_data['feature_names']
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return jsonify(importance_df.to_dict(orient='records'))

@app.route('/model_info')
def model_info():
    """Return model information"""
    if model_data is None:
        return jsonify({
            'error': 'Model not found. Please train the model first.'
        }), 500
    
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'n_estimators': 200,
        'max_depth': 15,
        'features': model_data['feature_names'],
        'categorical_features': model_data['categorical_cols'],
        'numerical_features': model_data['numerical_cols'],
        'smote_applied': True,
        'class_balance': 'Balanced using SMOTE'
    })

if __name__ == '__main__':
    if model_data is None:
        print("Warning: Model not found. Please run train_model.py first.")
    else:
        print("Model loaded successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
