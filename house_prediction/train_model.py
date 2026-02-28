"""
House Price Prediction - Training Script
Uses Linear Regression with feature scaling for house price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# Load the dataset
print("Loading house price dataset...")
df = pd.read_csv('house_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFeatures: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Encode categorical variable (location)
print("\nEncoding 'location' feature...")
label_encoder = LabelEncoder()
df['location_encoded'] = label_encoder.fit_transform(df['location'])
print(f"Location classes: {label_encoder.classes_}")

# Prepare features and target
feature_columns = ['location_encoded', 'size_sqft', 'bedrooms', 'age_years', 'amenities']
X = df[feature_columns]
y = df['price']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Feature Scaling using StandardScaler
print("\nApplying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\n--- Training Set Metrics ---")
print(f"Mean Squared Error (MSE): {train_mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {train_mae:,.2f}")
print(f"R² Score: {train_r2:.4f}")

# Test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- Test Set Metrics ---")
print(f"Mean Squared Error (MSE): {test_mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {test_mae:,.2f}")
print(f"R² Score: {test_r2:.4f}")

# Model coefficients
print("\n--- Model Coefficients ---")
print(f"Intercept: {model.intercept_:,.2f}")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature}: {coef:,.2f}")

# Feature importance (absolute coefficient values)
print("\n--- Feature Importance (by coefficient magnitude) ---")
importance = sorted(
    zip(feature_columns, np.abs(model.coef_)),
    key=lambda x: x[1],
    reverse=True
)
for feature, imp in importance:
    print(f"{feature}: {imp:,.2f}")

# Save the model and related objects
print("\nSaving model and preprocessing objects...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
with open('models/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("Model and preprocessing objects saved successfully!")
print("\nFiles saved:")
print("  - models/house_price_model.pkl")
print("  - models/scaler.pkl")
print("  - models/label_encoder.pkl")
print("  - models/feature_names.pkl")

# Sample predictions
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

# Test with a sample house
sample_house = pd.DataFrame({
    'location_encoded': [label_encoder.transform(['Urban'])[0]],
    'size_sqft': [1800],
    'bedrooms': [3],
    'age_years': [5],
    'amenities': [3]
})

sample_scaled = scaler.transform(sample_house)
predicted_price = model.predict(sample_scaled)[0]

print(f"\nSample House:")
print(f"  Location: Urban")
print(f"  Size: 1800 sqft")
print(f"  Bedrooms: 3")
print(f"  Age: 5 years")
print(f"  Amenities: 3")
print(f"  Predicted Price: ${predicted_price:,.2f}")

# Additional sample predictions
print("\n--- More Sample Predictions ---")
samples = [
    {'location': 'Suburban', 'size_sqft': 2000, 'bedrooms': 4, 'age_years': 5, 'amenities': 4},
    {'location': 'Rural', 'size_sqft': 1200, 'bedrooms': 2, 'age_years': 15, 'amenities': 1},
    {'location': 'Urban', 'size_sqft': 2500, 'bedrooms': 5, 'age_years': 2, 'amenities': 5},
]

for i, sample in enumerate(samples, 1):
    sample_df = pd.DataFrame({
        'location_encoded': [label_encoder.transform([sample['location']])[0]],
        'size_sqft': [sample['size_sqft']],
        'bedrooms': [sample['bedrooms']],
        'age_years': [sample['age_years']],
        'amenities': [sample['amenities']]
    })
    sample_scaled = scaler.transform(sample_df)
    pred = model.predict(sample_scaled)[0]
    
    print(f"\nSample {i}:")
    print(f"  {sample['location']}, {sample['size_sqft']} sqft, {sample['bedrooms']} bed, {sample['age_years']} years old, {sample['amenities']} amenities")
    print(f"  Predicted Price: ${pred:,.2f}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
