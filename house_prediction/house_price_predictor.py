"""
House Price Predictor - Reusable Prediction Class
Loads the trained model and makes predictions on new house data.
"""

import pickle
import pandas as pd
import numpy as np


class HousePricePredictor:
    """A class to predict house prices using trained Linear Regression model."""
    
    def __init__(self, model_path='models/house_price_model.pkl',
                 scaler_path='models/scaler.pkl',
                 encoder_path='models/label_encoder.pkl',
                 feature_names_path='models/feature_names.pkl'):
        """
        Initialize the predictor by loading the model and preprocessing objects.
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the scaler file
            encoder_path: Path to the label encoder file
            feature_names_path: Path to the feature names file
        """
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Load all required objects
        self._load_model(model_path)
        self._load_scaler(scaler_path)
        self._load_encoder(encoder_path)
        self._load_feature_names(feature_names_path)
        
        print("HousePricePredictor initialized successfully!")
        print(f"Location classes: {self.label_encoder.classes_.tolist()}")
    
    def _load_model(self, path):
        """Load the trained model from pickle file."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
    
    def _load_scaler(self, path):
        """Load the scaler from pickle file."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {path}")
    
    def _load_encoder(self, path):
        """Load the label encoder from pickle file."""
        with open(path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {path}")
    
    def _load_feature_names(self, path):
        """Load feature names from pickle file."""
        with open(path, 'rb') as f:
            self.feature_names = pickle.load(f)
        print(f"Feature names loaded from {path}")
    
    def predict(self, location, size_sqft, bedrooms, age_years, amenities):
        """
        Predict house price for a single house.
        
        Args:
            location: Location of the house ('Urban', 'Suburban', or 'Rural')
            size_sqft: Size of the house in square feet
            bedrooms: Number of bedrooms
            age_years: Age of the house in years
            amenities: Amenity level (0-5)
        
        Returns:
            float: Predicted house price
        """
        # Validate location
        if location not in self.label_encoder.classes_:
            raise ValueError(f"Invalid location. Must be one of: {self.label_encoder.classes_}")
        
        # Validate input ranges
        if size_sqft < 500:
            raise ValueError("Size must be at least 500 sqft for a valid prediction")
        if bedrooms < 1:
            raise ValueError("At least 1 bedroom is required")
        if age_years < 0:
            raise ValueError("Age cannot be negative")
        if amenities < 0 or amenities > 5:
            raise ValueError("Amenities must be between 0 and 5")
        
        # Encode location
        location_encoded = self.label_encoder.transform([location])[0]
        
        # Create feature array
        features = np.array([[
            location_encoded,
            size_sqft,
            bedrooms,
            age_years,
            amenities
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predicted_price = self.model.predict(features_scaled)[0]
        
        # Ensure non-negative price
        predicted_price = max(0, predicted_price)
        
        return predicted_price
    
    def predict_batch(self, data):
        """
        Predict house prices for multiple houses.
        
        Args:
            data: DataFrame or list of dictionaries containing house data
        
        Returns:
            numpy.ndarray: Array of predicted prices
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Encode locations
        data = data.copy()
        data['location_encoded'] = self.label_encoder.transform(data['location'])
        
        # Select features
        X = data[['location_encoded', 'size_sqft', 'bedrooms', 'age_years', 'amenities']]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_model_info(self):
        """Get information about the trained model."""
        info = {
            'model_type': 'LinearRegression',
            'features': self.feature_names,
            'location_classes': self.label_encoder.classes_.tolist(),
            'coefficients': dict(zip(self.feature_names, self.model.coef_.tolist())),
            'intercept': float(self.model.intercept_)
        }
        return info
    
    def predict_with_details(self, location, size_sqft, bedrooms, age_years, amenities):
        """
        Predict house price with detailed breakdown.
        
        Args:
            location: Location of the house
            size_sqft: Size of the house in square feet
            bedrooms: Number of bedrooms
            age_years: Age of the house in years
            amenities: Amenity level
        
        Returns:
            dict: Dictionary containing prediction and feature values
        """
        predicted_price = self.predict(location, size_sqft, bedrooms, age_years, amenities)
        
        # Calculate feature contributions
        location_encoded = self.label_encoder.transform([location])[0]
        
        # Get scaled feature values
        features = np.array([[
            location_encoded,
            size_sqft,
            bedrooms,
            age_years,
            amenities
        ]])
        features_scaled = self.scaler.transform(features)
        
        # Calculate contribution of each feature
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            contributions[feature] = float(features_scaled[0][i] * self.model.coef_[i])
        
        result = {
            'predicted_price': round(predicted_price, 2),
            'input_features': {
                'location': location,
                'size_sqft': size_sqft,
                'bedrooms': bedrooms,
                'age_years': age_years,
                'amenities': amenities
            },
            'feature_contributions': contributions,
            'base_price': round(self.model.intercept_, 2)
        }
        
        return result


def main():
    """Demo function to test the predictor."""
    print("="*60)
    print("House Price Prediction Demo")
    print("="*60)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    print("\n" + "-"*60)
    print("Single Prediction Demo")
    print("-"*60)
    
    # Test single prediction
    location = "Urban"
    size_sqft = 1800
    bedrooms = 3
    age_years = 5
    amenities = 3
    
    price = predictor.predict(location, size_sqft, bedrooms, age_years, amenities)
    
    print(f"\nInput:")
    print(f"  Location: {location}")
    print(f"  Size: {size_sqft} sqft")
    print(f"  Bedrooms: {bedrooms}")
    print(f"  Age: {age_years} years")
    print(f"  Amenities: {amenities}")
    print(f"\nPredicted Price: ${price:,.2f}")
    
    # Test with detailed prediction
    print("\n" + "-"*60)
    print("Detailed Prediction Demo")
    print("-"*60)
    
    result = predictor.predict_with_details("Suburban", 2000, 4, 5, 4)
    print(f"\nPredicted Price: ${result['predicted_price']:,.2f}")
    print(f"Base Price: ${result['base_price']:,.2f}")
    print(f"\nFeature Contributions:")
    for feature, contribution in result['feature_contributions'].items():
        sign = "+" if contribution >= 0 else ""
        print(f"  {feature}: {sign}{contribution:,.2f}")
    
    # Test batch prediction
    print("\n" + "-"*60)
    print("Batch Prediction Demo")
    print("-"*60)
    
    batch_data = [
        {'location': 'Urban', 'size_sqft': 1500, 'bedrooms': 3, 'age_years': 10, 'amenities': 2},
        {'location': 'Suburban', 'size_sqft': 1800, 'bedrooms': 4, 'age_years': 5, 'amenities': 3},
        {'location': 'Rural', 'size_sqft': 1200, 'bedrooms': 2, 'age_years': 20, 'amenities': 1},
    ]
    
    predictions = predictor.predict_batch(batch_data)
    
    print("\nBatch Predictions:")
    for i, (data, pred) in enumerate(zip(batch_data, predictions), 1):
        print(f"  {i}. {data['location']}, {data['size_sqft']} sqft, {data['bedrooms']} bed -> ${pred:,.2f}")
    
    # Print model info
    print("\n" + "-"*60)
    print("Model Information")
    print("-"*60)
    info = predictor.get_model_info()
    print(f"\nModel Type: {info['model_type']}")
    print(f"Features: {info['features']}")
    print(f"Location Classes: {info['location_classes']}")
    print(f"\nCoefficients:")
    for feature, coef in info['coefficients'].items():
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {info['intercept']:.4f}")


if __name__ == "__main__":
    main()
