"""
Customer Churn Prediction Model Training Script
Uses SMOTE for handling class imbalance and Random Forest for feature importance
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic customer churn data"""
    print("Generating synthetic customer data...")
    
    # Customer demographics and account info
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.10]),
        'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], n_samples, p=[0.30, 0.45, 0.25]),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_samples, p=[0.40, 0.40, 0.20]),
        'contract_length': np.random.choice(['Monthly', 'Quarterly', 'Annual'], n_samples, p=[0.35, 0.40, 0.25]),
        'tenure_months': np.random.randint(1, 61, n_samples),
        'monthly_charge': np.round(np.random.uniform(10, 200, n_samples), 2),
        'total_charges': np.round(np.random.uniform(50, 10000, n_samples), 2),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Net Banking'], n_samples, p=[0.30, 0.25, 0.30, 0.15]),
        'num_support_tickets': np.random.randint(0, 20, n_samples),
        'num_complaints': np.random.randint(0, 10, n_samples),
        'product_usage_score': np.random.randint(1, 101, n_samples),
        'login_frequency': np.random.randint(1, 31, n_samples),
        'last_purchase_days_ago': np.random.randint(1, 180, n_samples),
        'avg_session_duration': np.random.randint(1, 61, n_samples),
        'is_premium_member': np.random.choice([0, 1], n_samples, p=[0.60, 0.40]),
        'has_Referral': np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        'engagement_score': np.random.randint(1, 101, n_samples),
        'satisfaction_rating': np.random.randint(1, 11, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target with realistic patterns (imbalanced - ~26% churn)
    # Churn is influenced by certain factors
    churn_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.1  # Base probability
        
        # Factors that increase churn
        if df.loc[i, 'tenure_months'] < 12:
            prob += 0.15
        if df.loc[i, 'monthly_charge'] > 150:
            prob += 0.10
        if df.loc[i, 'num_support_tickets'] > 5:
            prob += 0.12
        if df.loc[i, 'num_complaints'] > 3:
            prob += 0.10
        if df.loc[i, 'product_usage_score'] < 30:
            prob += 0.15
        if df.loc[i, 'satisfaction_rating'] <= 3:
            prob += 0.20
        if df.loc[i, 'engagement_score'] < 25:
            prob += 0.12
        if df.loc[i, 'last_purchase_days_ago'] > 90:
            prob += 0.08
        if df.loc[i, 'contract_length'] == 'Monthly':
            prob += 0.08
        if df.loc[i, 'subscription_type'] == 'Basic':
            prob += 0.05
            
        # Factors that decrease churn
        if df.loc[i, 'tenure_months'] > 36:
            prob -= 0.10
        if df.loc[i, 'is_premium_member'] == 1:
            prob -= 0.08
        if df.loc[i, 'has_Referral'] == 1:
            prob -= 0.05
        if df.loc[i, 'satisfaction_rating'] >= 8:
            prob -= 0.10
            
        churn_prob[i] = prob
    
    # Convert probabilities to churn (with threshold to get ~26% churn rate)
    threshold = np.percentile(churn_prob, 74)
    df['churn'] = (churn_prob > threshold).astype(int)
    
    print(f"Generated {n_samples} customer records")
    print(f"Churn distribution: {df['churn'].value_counts().to_dict()}")
    print(f"Churn rate: {df['churn'].mean()*100:.2f}%")
    
    return df

def preprocess_data(df):
    """Preprocess the data for model training"""
    print("\nPreprocessing data...")
    
    # Store customer_id for later reference
    customer_ids = df['customer_id'].copy()
    
    # Drop customer_id as it's not a feature
    df_model = df.drop('customer_id', axis=1)
    
    # Separate features and target
    X = df_model.drop('churn', axis=1)
    y = df_model['churn']
    
    # Encode categorical variables
    categorical_cols = ['gender', 'city_tier', 'subscription_type', 'contract_length', 'payment_method']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numerical features
    numerical_cols = ['age', 'tenure_months', 'monthly_charge', 'total_charges', 
                      'num_support_tickets', 'num_complaints', 'product_usage_score',
                      'login_frequency', 'last_purchase_days_ago', 'avg_session_duration',
                      'engagement_score', 'satisfaction_rating']
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"Features: {list(X.columns)}")
    print(f"Shape: {X.shape}")
    
    return X, y, label_encoders, scaler, customer_ids

def train_model_with_smote(X_train, y_train):
    """Train model with SMOTE for handling class imbalance"""
    print("\nApplying SMOTE for class imbalance...")
    
    # Check class distribution before SMOTE
    print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
    print(f"Training samples: {len(X_train_balanced)}")
    
    # Train Random Forest model
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    return model, smote

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy, roc_auc, y_pred

def get_feature_importance(model, feature_names):
    """Extract and display feature importance"""
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def save_model(model, label_encoders, scaler, smote, feature_names, output_dir):
    """Save the trained model and preprocessing objects"""
    print(f"\nSaving model to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'smote': smote,
        'feature_names': list(feature_names),
        'categorical_cols': ['gender', 'city_tier', 'subscription_type', 'contract_length', 'payment_method'],
        'numerical_cols': ['age', 'tenure_months', 'monthly_charge', 'total_charges', 
                          'num_support_tickets', 'num_complaints', 'product_usage_score',
                          'login_frequency', 'last_purchase_days_ago', 'avg_session_duration',
                          'engagement_score', 'satisfaction_rating']
    }
    
    with open(os.path.join(output_dir, 'churn_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")

def main():
    """Main training pipeline"""
    print("="*50)
    print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=5000)
    
    # Preprocess data
    X, y, label_encoders, scaler, customer_ids = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train with SMOTE
    model, smote = train_model_with_smote(X_train, y_train)
    
    # Evaluate
    accuracy, roc_auc, y_pred = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    importance_df = get_feature_importance(model, X.columns)
    
    # Save model
    save_model(model, label_encoders, scaler, smote, X.columns, 'models')
    
    # Save feature importance to CSV
    importance_df.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Model saved to: models/churn_model.pkl")
    print(f"Feature importance saved to: models/feature_importance.csv")
    
    return model, accuracy, roc_auc

if __name__ == "__main__":
    main()
