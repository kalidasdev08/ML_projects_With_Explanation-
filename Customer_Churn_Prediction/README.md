# Customer Churn Prediction

A machine learning application to predict customer churn using classification techniques. This project handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and provides feature importance analysis to understand which factors contribute most to customer churn.

## Problem Statement

Companies lose customers over time and need early warning systems to identify customers at risk of leaving their subscription service. This project predicts whether a customer will churn based on various features such as demographics, subscription details, usage patterns, and satisfaction metrics.

## Features

- **Classification Model**: Random Forest Classifier for predicting customer churn
- **SMOTE Integration**: Handles class imbalance in the dataset (typically ~26% churn rate)
- **Feature Importance Analysis**: Identifies the most important factors contributing to churn
- **REST API**: Flask-based web application for real-time predictions
- **Batch Prediction**: Support for predicting churn for multiple customers at once

## Dataset Features

The model uses the following features to predict churn:

### Demographic Features
- Age
- Gender
- City Tier

### Subscription Features
- Subscription Type (Basic, Premium, Enterprise)
- Contract Length (Monthly, Quarterly, Annual)
- Monthly Charge
- Total Charges
- Payment Method

### Usage & Engagement Features
- Tenure (months)
- Number of Support Tickets
- Number of Complaints
- Product Usage Score
- Login Frequency
- Last Purchase Days Ago
- Average Session Duration
- Engagement Score
- Satisfaction Rating

### Membership Features
- Is Premium Member
- Has Referral

## Installation

1. Clone the repository or navigate to the project directory:
```bash
cd Customer_Churn_Prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script to generate synthetic data and train the model:

```bash
python train_model.py
```

This will:
1. Generate synthetic customer data (5000 records)
2. Preprocess the data (encode categorical variables, scale numerical features)
3. Apply SMOTE to handle class imbalance
4. Train a Random Forest classifier
5. Evaluate model performance
6. Save the model to `models/churn_model.pkl`
7. Save feature importance to `models/feature_importance.csv`

## Running the Web Application

Start the Flask application:

```bash
python app.py
```

The application will run on `http://localhost:5000`. Open this URL in your web browser to access the prediction interface.

## API Endpoints

### POST /predict
Predict churn for a single customer.

**Request Body:**
```json
{
    "gender": "Male",
    "city_tier": "Tier1",
    "subscription_type": "Premium",
    "contract_length": "Annual",
    "age": 35,
    "tenure_months": 24,
    "monthly_charge": 99.99,
    "total_charges": 2400.00,
    "payment_method": "Credit Card",
    "num_support_tickets": 2,
    "num_complaints": 0,
    "product_usage_score": 75,
    "login_frequency": 15,
    "last_purchase_days_ago": 10,
    "avg_session_duration": 25,
    "is_premium_member": 1,
    "has_referral": 1,
    "engagement_score": 80,
    "satisfaction_rating": 8
}
```

**Response:**
```json
{
    "prediction": "No",
    "churn_probability": 12.50,
    "not_churn_probability": 87.50,
    "risk_level": "Low"
}
```

### POST /batch_predict
Predict churn for multiple customers.

**Request Body:**
```json
{
    "customers": [
        {
            "customer_id": "CUST_001",
            "gender": "Male",
            ...
        },
        {
            "customer_id": "CUST_002",
            "gender": "Female",
            ...
        }
    ]
}
```

### GET /feature_importance
Get feature importance analysis.

### GET /model_info
Get model configuration and information.

## Model Performance

The model achieves:
- **Accuracy**: ~90%+
- **ROC-AUC Score**: ~95%+

After applying SMOTE, the training data is balanced, which helps the model learn patterns from both churned and non-churned customers effectively.

## Feature Importance

The top features that influence churn prediction typically include:
1. Tenure Months
2. Satisfaction Rating
3. Product Usage Score
4. Engagement Score
5. Number of Support Tickets
6. Monthly Charge
7. Contract Length
8. Age
9. Number of Complaints
10. Last Purchase Days Ago

## Tech Stack

- **Python 3.8+**
- **Flask**: Web framework
- **scikit-learn**: Machine learning algorithms
- **imbalanced-learn**: SMOTE implementation
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## Project Structure

```
Customer_Churn_Prediction/
├── app.py                  # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/
│   ├── churn_model.pkl   # Trained model and preprocessing objects
│   └── feature_importance.csv  # Feature importance analysis
└── templates/
    └── index.html        # Web interface template
```

## Usage Tips

1. **High Risk Customers**: Focus retention efforts on customers with >70% churn probability
2. **At-Risk Customers**: Monitor customers with 40-70% churn probability
3. **Feature Engineering**: Use feature importance to guide business decisions
4. **Regular Retraining**: Retrain the model periodically with updated data

## License

This project is for educational and demonstration purposes.
