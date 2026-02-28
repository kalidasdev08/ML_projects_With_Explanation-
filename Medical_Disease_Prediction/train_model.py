"""
Medical Disease Prediction Model Training
==========================================
This script trains a machine learning model to predict diseases from symptoms.
Uses a Decision Tree classifier for interpretability in medical diagnosis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create sample medical dataset with symptoms and diseases
def create_medical_dataset():
    """
    Create a comprehensive medical dataset with symptoms and corresponding diseases.
    This dataset is designed to simulate real-world medical diagnosis scenarios.
    """
    # Define symptoms (features)
    symptoms = [
        'fever', 'cough', 'headache', 'fatigue', 'body_ache',
        'nausea', 'vomiting', 'diarrhea', 'rash', 'sore_throat',
        'shortness_of_breath', 'chest_pain', 'abdominal_pain',
        'dizziness', 'loss_of_taste_smell', 'joint_pain', 'skin_lesion',
        'eye_discharge', 'ear_pain', 'runny_nose'
    ]
    
    # Define diseases with their symptom patterns
    disease_data = [
        # Common Cold
        {'disease': 'Common Cold', 'symptoms': ['cough', 'runny_nose', 'sore_throat', 'headache', 'fatigue']},
        
        # Flu (Influenza)
        {'disease': 'Flu (Influenza)', 'symptoms': ['fever', 'cough', 'body_ache', 'headache', 'fatigue']},
        
        # Fever (General)
        {'disease': 'Viral Fever', 'symptoms': ['fever', 'headache', 'body_ache', 'fatigue', 'nausea']},
        
        # Dengue
        {'disease': 'Dengue', 'symptoms': ['fever', 'headache', 'body_ache', 'rash', 'nausea']},
        
        # Malaria
        {'disease': 'Malaria', 'symptoms': ['fever', 'headache', 'body_ache', 'vomiting', 'fatigue']},
        
        # Typhoid
        {'disease': 'Typhoid', 'symptoms': ['fever', 'abdominal_pain', 'vomiting', 'diarrhea', 'fatigue']},
        
        # COVID-19
        {'disease': 'COVID-19', 'symptoms': ['fever', 'cough', 'shortness_of_breath', 'loss_of_taste_smell', 'fatigue']},
        
        # Pneumonia
        {'disease': 'Pneumonia', 'symptoms': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue']},
        
        # Bronchitis
        {'disease': 'Bronchitis', 'symptoms': ['cough', 'shortness_of_breath', 'fatigue', 'chest_pain', 'body_ache']},
        
        # Gastritis
        {'disease': 'Gastritis', 'symptoms': ['abdominal_pain', 'nausea', 'vomiting', 'diarrhea', 'fatigue']},
        
        # Food Poisoning
        {'disease': 'Food Poisoning', 'symptoms': ['vomiting', 'diarrhea', 'abdominal_pain', 'nausea', 'body_ache']},
        
        # Migraine
        {'disease': 'Migraine', 'symptoms': ['headache', 'nausea', 'dizziness', 'vomiting', 'fatigue']},
        
        # Allergy
        {'disease': 'Allergy', 'symptoms': ['runny_nose', 'sore_throat', 'eye_discharge', 'rash', 'sneezing']},
        
        # Skin Infection
        {'disease': 'Skin Infection', 'symptoms': ['rash', 'skin_lesion', 'body_ache', 'fever', 'fatigue']},
        
        # Ear Infection
        {'disease': 'Ear Infection', 'symptoms': ['ear_pain', 'fever', 'headache', 'dizziness', 'nausea']},
        
        # Eye Infection
        {'disease': 'Eye Infection', 'symptoms': ['eye_discharge', 'rash', 'headache', 'sore_throat', 'fatigue']},
        
        # Arthritis
        {'disease': 'Arthritis', 'symptoms': ['joint_pain', 'body_ache', 'fatigue', 'fever', 'dizziness']},
        
        # Anemia
        {'disease': 'Anemia', 'symptoms': ['fatigue', 'dizziness', 'body_ache', 'shortness_of_breath', 'headache']},
        
        # Depression/Anxiety
        {'disease': 'Stress/Anxiety', 'symptoms': ['fatigue', 'headache', 'dizziness', 'body_ache', 'abdominal_pain']},
        
        # Dehydration
        {'disease': 'Dehydration', 'symptoms': ['dizziness', 'fatigue', 'headache', 'vomiting', 'diarrhea']}
    ]
    
    return disease_data, symptoms

def prepare_dataset():
    """
    Prepare the dataset for training by encoding symptoms as binary features.
    """
    disease_data, symptom_list = create_medical_dataset()
    
    # Create feature matrix (X) and target vector (y)
    X = []
    y = []
    
    for disease_info in disease_data:
        disease = disease_info['disease']
        disease_symptoms = set(disease_info['symptoms'])
        
        # Create binary features for each symptom
        # Add multiple samples with slight variations to increase dataset size
        for _ in range(10):  # 10 samples per disease
            features = []
            for symptom in symptom_list:
                # Add symptom with probability, making it slightly noisy
                if symptom in disease_symptoms:
                    features.append(1 if np.random.random() > 0.1 else 0)  # 90% chance of having symptom
                else:
                    features.append(1 if np.random.random() < 0.1 else 0)  # 10% chance of false positive
            
            # Ensure at least 2 symptoms are present
            if sum(features) < 2:
                # Add most characteristic symptoms
                for i, symptom in enumerate(symptom_list):
                    if symptom in disease_symptoms:
                        features[i] = 1
                        if sum(features) >= 2:
                            break
            
            X.append(features)
            y.append(disease)
    
    # Also add some random samples for robustness
    for _ in range(50):
        features = [np.random.randint(0, 2) for _ in symptom_list]
        # Ensure at least one symptom
        if sum(features) == 0:
            features[np.random.randint(0, len(symptom_list))] = 1
        X.append(features)
        y.append("Unknown")  # Add some unknown cases
    
    return np.array(X), np.array(y), symptom_list

def train_model():
    """
    Train the disease prediction model.
    """
    print("=" * 60)
    print("Medical Disease Prediction - Model Training")
    print("=" * 60)
    
    # Prepare dataset
    print("\n[1] Preparing dataset...")
    X, y, symptom_list = prepare_dataset()
    
    print(f"    - Total samples: {len(X)}")
    print(f"    - Number of symptoms (features): {len(symptom_list)}")
    print(f"    - Unique diseases: {len(set(y)) - 1}")  # -1 for 'Unknown'
    
    # Split data
    print("\n[2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Test samples: {len(X_test)}")
    
    # Train model - Using Decision Tree for interpretability
    print("\n[3] Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[4] Evaluating model...")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"    - Training Accuracy: {train_accuracy:.4f}")
    print(f"    - Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Filter out 'Unknown' for detailed report
    mask = y_test != 'Unknown'
    if mask.any():
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        if len(set(y_test_filtered)) > 1:
            print("\n[5] Classification Report:")
            print(classification_report(y_test_filtered, y_pred_filtered))
    
    # Save model and related objects
    print("\n[6] Saving model and related files...")
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "disease_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"    - Model saved to: {model_path}")
    
    # Save symptom list
    symptom_list_path = os.path.join(models_dir, "symptom_list.pkl")
    with open(symptom_list_path, 'wb') as f:
        pickle.dump(symptom_list, f)
    print(f"    - Symptom list saved to: {symptom_list_path}")
    
    # Save symptom descriptions for chatbot
    symptom_descriptions = {
        'fever': 'Elevated body temperature',
        'cough': 'Persistent coughing',
        'headache': 'Pain in the head region',
        'fatigue': 'Extreme tiredness',
        'body_ache': 'General muscle pain',
        'nausea': 'Feeling of sickness with urge to vomit',
        'vomiting': 'Expelling contents from stomach',
        'diarrhea': 'Loose, watery stools',
        'rash': 'Skin eruption or discoloration',
        'sore_throat': 'Pain or irritation in throat',
        'shortness_of_breath': 'Difficulty breathing',
        'chest_pain': 'Pain in chest region',
        'abdominal_pain': 'Pain in stomach area',
        'dizziness': 'Feeling of unsteadiness',
        'loss_of_taste_smell': 'Cannot taste or smell',
        'joint_pain': 'Pain in joints',
        'skin_lesion': 'Abnormal skin patch',
        'eye_discharge': 'Fluid from eyes',
        'ear_pain': 'Pain in ear',
        'runny_nose': 'Mucus from nose',
        'sneezing': 'Expelling air from nose'
    }
    
    symptom_desc_path = os.path.join(models_dir, "symptom_descriptions.pkl")
    with open(symptom_desc_path, 'wb') as f:
        pickle.dump(symptom_descriptions, f)
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)
    
    return model, symptom_list

if __name__ == "__main__":
    train_model()
