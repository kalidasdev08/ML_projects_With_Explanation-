# House Price Prediction Project

A Machine Learning project that predicts house prices using Linear Regression with feature scaling.

## Features

- **Location**: Urban, Suburban, or Rural
- **Size**: House size in square feet
- **Bedrooms**: Number of bedrooms
- **Age**: Age of the house in years
- **Amenities**: Amenity level (0-5)

## Project Structure

```
house_prediction/
├── house_data.csv           # Sample dataset
├── train_model.py           # Training script
├── house_price_predictor.py # Prediction class
├── app.py                   # Flask web application
├── requirements.txt         # Dependencies
├── README.md                # This file
├── templates/
│   └── index.html          # Web interface
└── models/                  # Saved model files (generated after training)
    ├── house_price_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_names.pkl
```

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Run the training script to train the model and generate model files:

```bash
python train_model.py
```

This will:
- Load the dataset from `house_data.csv`
- Preprocess the data (encode location, scale features)
- Train a Linear Regression model
- Evaluate the model using various metrics (MSE, RMSE, MAE, R²)
- Save the model and preprocessing objects to the `models/` directory

### Step 2: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

### Step 3: Make Predictions

Use the web interface to:
- Enter house details (location, size, bedrooms, age, amenities)
- Get instant price predictions
- View sample predictions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | POST | Make a prediction |
| `/model-info` | GET | Get model information |
| `/sample-predictions` | GET | Get sample predictions |

### Prediction Request Example

```json
POST /predict
{
    "location": "Urban",
    "size_sqft": 1800,
    "bedrooms": 3,
    "age_years": 5,
    "amenities": 3
}
```

### Prediction Response

```json
{
    "success": true,
    "prediction": {
        "price": 385000.00,
        "formatted_price": "$385,000.00"
    },
    "input": {
        "location": "Urban",
        "size_sqft": 1800,
        "bedrooms": 3,
        "age_years": 5,
        "amenities": 3
    }
}
```

## Model Evaluation

The model is evaluated using:

- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **R² Score**: Coefficient of determination (0-1, higher is better)

## Using the Predictor Class

You can also use the `HousePricePredictor` class directly in your code:

```python
from house_price_predictor import HousePricePredictor

# Initialize the predictor
predictor = HousePricePredictor()

# Make a single prediction
price = predictor.predict(
    location="Urban",
    size_sqft=1800,
    bedrooms=3,
    age_years=5,
    amenities=3
)

print(f"Predicted Price: ${price:,.2f}")

# Get detailed prediction
details = predictor.predict_with_details(
    location="Suburban",
    size_sqft=2000,
    bedrooms=4,
    age_years=5,
    amenities=4
)

print(f"Predicted Price: ${details['predicted_price']:,.2f}")
print(f"Feature Contributions: {details['feature_contributions']}")
```

## Technologies Used

- **Python**: Programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **Flask**: Web framework

## License

This project is for educational purposes.
