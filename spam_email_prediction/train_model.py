"""
Spam Email Detector Training Script
=====================================
This script trains a spam email classifier using:
- CountVectorizer for text feature extraction
- Naive Bayes (MultinomialNB) for classification

Dataset: UCI Spam Email Dataset (or generated sample data)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_or_generate_data():
    """
    Load spam email dataset or generate sample data
    Using the classic UCI Spam dataset format
    """
    # Sample spam and ham emails for training
    # In production, you would load from a CSV file
    
    spam_emails = [
        "Congratulations! You've won a free lottery ticket! Claim your prize now!",
        "Click here to claim your million dollar reward! Limited time offer!",
        "Your account has been compromised. Verify your password immediately!",
        "You have been selected for a cash prize of $1,000,000. Reply now!",
        "Buy cheap medications online without prescription! Best prices guaranteed!",
        "Make money from home! Work just 2 hours a day and earn $5000 weekly!",
        "Your PayPal account needs verification. Click the link below!",
        "Congratulations! Your mobile number has won $500,000!",
        "Get free credit checks instantly! No credit card required!",
        "Hot stocks! Invest now and double your money in one week!",
        "You have a pending package delivery. Update your shipping address!",
        "Your Netflix account will be suspended. Update payment info now!",
        "FREE iPhone 12! Complete this survey to claim your prize!",
        "Earn money fast! No experience needed! Starting salary $8000/month!",
        "Your bank account has been locked. Verify your identity immediately!",
        "Congratulations Amazon user! You've won a $1000 gift card!",
        "Click here to see who's viewed your profile! Hot singles in your area!",
        "Your IRS tax refund is waiting. Claim it now!",
        "Double your investment in 7 days! This offer expires soon!",
        "Your computer is infected! Download our free antivirus now!",
        "Win a free cruise vacation! Complete the survey to claim!",
        "Your electric bill is overdue. Pay now to avoid service interruption!",
        "Get rich quick! This secret investment will make you wealthy!",
        "Exclusive deal! Buy one get one free on all products!",
        "Your Amazon order cannot be delivered. Update your address!",
        "Congratulations! You've been pre-approved for a credit card!",
        "Make $10,000 per month working from home! Real job!",
        "Your Walmart gift card balance is $500. Claim it now!",
        "Urgent: Your social security number has been compromised!",
        "Free gift cards! Complete these simple offers!",
    ]
    
    ham_emails = [
        "Hi John, can we schedule a meeting for tomorrow afternoon?",
        "Thank you for your order. Your items will be shipped within 3-5 business days.",
        "Please find attached the report you requested.",
        "Hey, are you available for lunch next Tuesday?",
        "The project deadline has been extended to next Friday.",
        "I've attached the presentation slides from today's meeting.",
        "Can you please review the attached document and provide feedback?",
        "Thanks for referring me to your colleague. I really appreciate it!",
        "The team meeting has been rescheduled to Thursday at 2 PM.",
        "I've confirmed your reservation for next weekend.",
        "Please let me know if you need any additional information.",
        "The invoice for your recent purchase is attached.",
        "It was great meeting you at the conference last week!",
        "I've updated the shared document with the latest changes.",
        "Could you please send me the contact information for the vendor?",
        "The quarterly report shows promising results for our department.",
        "I've booked the conference room for Monday morning.",
        "Thanks for your help with the presentation. It went really well!",
        "The client has approved the proposal. We can proceed with the project.",
        "Please remember to submit your timesheet by end of day.",
        "I've attached the relevant files for your review.",
        "The new office policies have been posted on the company website.",
        "Can we reschedule our meeting to a different time?",
        "I've completed the training module as requested.",
        "The maintenance team will be in the office this Saturday.",
        "Thanks for your email. I'll get back to you on Monday.",
        "The budget for the upcoming project has been approved.",
        "I've scheduled a call with the client for next week.",
        "Please confirm your attendance for the company event.",
        "The new software update includes several bug fixes.",
    ]
    
    # Create DataFrame
    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
    
    df = pd.DataFrame({
        'text': emails,
        'label': labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_spam_classifier():
    """
    Train the spam email classifier using CountVectorizer and Naive Bayes
    """
    print("=" * 60)
    print("SPAM EMAIL DETECTOR - TRAINING")
    print("=" * 60)
    
    # Load or generate data
    print("\n[1/5] Loading dataset...")
    df = load_or_generate_data()
    print(f"      Total samples: {len(df)}")
    print(f"      Spam emails: {len(df[df['label'] == 'spam'])}")
    print(f"      Ham emails: {len(df[df['label'] == 'ham'])}")
    
    # Prepare features and labels
    X = df['text']
    y = df['label']
    
    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples: {len(X_test)}")
    
    # Create CountVectorizer
    print("\n[3/5] Creating CountVectorizer...")
    vectorizer = CountVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=1000,
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Fit and transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"      Feature matrix shape: {X_train_vectorized.shape}")
    
    # Train Naive Bayes classifier
    print("\n[4/5] Training Naive Bayes classifier...")
    classifier = MultinomialNB(alpha=1.0)  # Alpha is Laplace smoothing parameter
    classifier.fit(X_train_vectorized, y_train)
    print("      Training complete!")
    
    # Evaluate model
    print("\n[5/5] Evaluating model performance...")
    y_pred = classifier.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n      ACCURACY: {accuracy * 100:.2f}%")
    
    print("\n      CLASSIFICATION REPORT:")
    print("      " + "-" * 50)
    report = classification_report(y_test, y_pred)
    for line in report.split('\n'):
        print(f"      {line}")
    
    print("\n      CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"      Predicted:    ham  spam")
    print(f"      Actual ham:  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      Actual spam: {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Save the model and vectorizer
    print("\n[SAVING MODEL]")
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("      Saved: models/vectorizer.pkl")
    
    # Save classifier
    with open('models/spam_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("      Saved: models/spam_classifier.pkl")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return classifier, vectorizer

def test_predictions():
    """
    Test the trained model with sample predictions
    """
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    # Load saved model and vectorizer
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/spam_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    # Test emails
    test_emails = [
        "Congratulations! You've won a free lottery!",
        "Hi, can we meet for lunch tomorrow?",
        "Click here to claim your prize now!!!",
        "Please find attached the meeting notes.",
        "URGENT: Your account needs verification!",
        "Thanks for the lunch invitation, I'll be there.",
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    for email in test_emails:
        # Vectorize the email
        email_vectorized = vectorizer.transform([email])
        
        # Predict
        prediction = classifier.predict(email_vectorized)[0]
        
        # Get probability
        proba = classifier.predict_proba(email_vectorized)[0]
        ham_prob = proba[0]
        spam_prob = proba[1]
        
        print(f"\nEmail: {email[:50]}...")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence - Ham: {ham_prob*100:.1f}%, Spam: {spam_prob*100:.1f}%")

if __name__ == "__main__":
    # Train the model
    train_spam_classifier()
    
    # Test with sample predictions
    test_predictions()
