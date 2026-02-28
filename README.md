# Machine Learning Projects Collection 🤖📊

A comprehensive collection of 18 machine learning projects covering various domains including Natural Language Processing (NLP), Computer Vision, Predictive Analytics, and more. Each project is a complete, runnable application with its own README documentation.

---

## 📁 Project Overview

| # | Project Name | Type | Description |
|---|-------------|------|-------------|
| 1 | [AI_Interview_Bot](#1-ai_interview_bot) | NLP/AI | AI-powered interview practice application |
| 2 | [AI_Voice_Assistant_mini_alexa](#2-ai_voice_assistant_mini_alexa) | Speech/AI | Voice assistant like Alexa |
| 3 | [Customer_Churn_Prediction](#3-customer_churn_prediction) | Classification | Predict customer churn probability |
| 4 | [dog_bread_analyser](#4-dog_bread_analyser) | Computer Vision | Dog breed classification |
| 5 | [Fake_new_prediction](#5-fake_new_prediction) | NLP | Fake news detection |
| 6 | [Handwritten_Digit_Recognizer](#6-handwritten_digit_recognizer) | Computer Vision | CNN-based digit recognition |
| 7 | [house_prediction](#7-house_prediction) | Regression | House price prediction |
| 8 | [India_Rainfall_Analysis](#8-india_rainfall_analysis) | Analytics | Rainfall analysis for agriculture |
| 9 | [Medical_Disease_Prediction](#9-medical_disease_prediction) | Classification | Disease prediction from symptoms |
| 10 | [Movie_Recommendation_System](#10-movie_recommendation_system) | Recommendation | Movie recommendations |
| 11 | [Personal_Finance_Advisor](#11-personal_finance_advisor) | Finance | Financial advisory system |
| 12 | [resume_screening](#12-resume_screening) | NLP | Resume parsing and screening |
| 13 | [Road_Accident_Severity_Predictor](#13-road_accident_severity_predictor) | Classification | Predict accident severity |
| 14 | [Sentiment_Analysis](#14-sentiment_analysis) | NLP | Sentiment analysis on reviews |
| 15 | [spam_email_prediction](#15-spam_email_prediction) | NLP/Classification | Spam email detection |
| 16 | [student_result_predictions](#16-student_result_predictions) | Regression | Student performance prediction |
| 17 | [Traffic_Sign_Recognition](#17-traffic_sign_recognition) | Computer Vision | Traffic sign classification |
| 18 | [dog_bread_analyser](#18-dog_bread_classifier) | Computer Vision | Dog breed identification |

---

## 📋 Detailed Project Descriptions

---

### 1. AI_Interview_Bot

**Type:** NLP / AI  
**Description:** An AI-powered interview practice application that generates unique interview questions using Google Gemini API and evaluates your answers in real-time.

**Features:**
- AI-Generated Questions using Gemini AI
- Job-Role Customization for personalized questions
- Multiple Categories: Technical, Behavioral, Situational, General
- AI-Powered Evaluation with detailed feedback

**Tech Stack:** Flask, Google Gemini API, HTML/CSS/JavaScript

**Run:**
```bash
cd AI_Interview_Bot
pip install -r requirements.txt
python app.py
```

---

### 2. AI_Voice_Assistant_mini_alexa

**Type:** Speech Recognition / AI  
**Description:** A voice assistant similar to Alexa that can perform various tasks using voice commands.

**Features:**
- Voice command recognition
- Web search capabilities
- Task automation
- Interactive responses

**Tech Stack:** Python, Speech Recognition, pyttsx3, Wolfram Alpha API

**Run:**
```bash
cd AI_Voice_Assistant_mini_alexa
pip install -r requirements.txt
python voice_assistant.py
```

---

### 3. Customer_Churn_Prediction

**Type:** Classification  
**Description:** A machine learning application that predicts whether a customer will leave (churn) based on their behavior and demographic data.

**Features:**
- Customer data analysis
- Churn probability prediction
- Interactive web interface
- Model performance metrics

**Tech Stack:** Python, scikit-learn, Flask, pandas

**Run:**
```bash
cd Customer_Churn_Prediction
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 4. dog_bread_analyser

**Type:** Computer Vision  
**Description:** A deep learning application that classifies dog breeds from images using CNN architecture.

**Features:**
- Image upload for breed identification
- Multiple breed classification
- Confidence scores
- Interactive UI

**Tech Stack:** TensorFlow/Keras, Flask, OpenCV

**Run:**
```bash
cd dog_bread_analyser
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 5. Fake_new_prediction

**Type:** NLP  
**Description:** A machine learning project that detects whether a news article is REAL or FAKE using NLP classification techniques.

**Features:**
- Text input for news articles
- File upload (PDF, TXT, CSV)
- Real-time predictions
- Confidence scores

**ML Concepts:**
- NLP Classification (Binary)
- TF-IDF Vectorization
- Naive Bayes Classifier

**Performance:**
- Training Accuracy: 91.35%
- Test Accuracy: 87.45%

**Tech Stack:** Flask, scikit-learn, NLTK, pandas

**Run:**
```bash
cd Fake_new_prediction
pip install -r requirements.txt
python app.py
```

---

### 6. Handwritten_Digit_Recognizer

**Type:** Computer Vision  
**Description:** A web-based application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

**Features:**
- Interactive Drawing Canvas
- Image Upload
- Real-time Prediction
- Probability Distribution for all digits

**Model Architecture:**
- 3 Conv2D layers with MaxPooling
- Dense layers with Dropout
- Softmax output (10 classes)

**Performance:** ~99% accuracy on MNIST

**Tech Stack:** TensorFlow/Keras, Flask, OpenCV

**Run:**
```bash
cd "Handwritten Digit Recognizer"
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 7. house_prediction

**Type:** Regression  
**Description:** A Machine Learning project that predicts house prices using Linear Regression with feature scaling.

**Features:**
- Location-based prediction (Urban, Suburban, Rural)
- Size, bedrooms, age, amenities inputs
- Instant price predictions
- API endpoints

**Tech Stack:** Python, scikit-learn, Flask, pandas

**Run:**
```bash
cd house_prediction
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 8. India_Rainfall_Analysis

**Type:** Analytics / Prediction  
**Description:** Exploratory data analysis and prediction of rainfall patterns in India for agricultural purposes.

**Features:**
- Historical rainfall data analysis
- Predictive modeling
- Data visualization
- Agricultural insights

**Tech Stack:** Python, pandas, scikit-learn, Flask

**Run:**
```bash
cd India_Rainfall_Analysis
pip install -r requirements.txt
python app.py
```

---

### 9. Medical_Disease_Prediction

**Type:** Classification  
**Description:** A machine learning application that predicts diseases based on symptoms and patient data.

**Features:**
- Symptom-based prediction
- Multiple disease categories
- Probability scores
- Medical data analysis

**Tech Stack:** Python, scikit-learn, Flask, pandas

**Run:**
```bash
cd Medical_Disease_Prediction
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 10. Movie_Recommendation_System

**Type:** Recommendation System  
**Description:** A content-based and collaborative filtering movie recommendation system.

**Features:**
- Movie suggestions based on preferences
- Rating predictions
- Similar movie recommendations
- Interactive UI

**Tech Stack:** Python, scikit-learn, Flask, pandas

**Run:**
```bash
cd Movie_Recommendation_System
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 11. Personal_Finance_Advisor

**Type:** Finance / Analytics  
**Description:** An intelligent system that provides financial advice and manages personal finances.

**Features:**
- Financial data analysis
- Investment suggestions
- Budget planning
- Expense tracking

**Tech Stack:** Python, Flask, pandas

**Run:**
```bash
cd Personal_Finance_Advisor
pip install -r requirements.txt
python app.py
```

---

### 12. resume_screening

**Type:** NLP  
**Description:** An automated resume screening system that parses resumes and matches them to job requirements.

**Features:**
- Resume parsing (PDF, DOCX)
- Skill extraction
- Job matching
- Ranking candidates

**Tech Stack:** Python, Flask, NLTK, scikit-learn

**Run:**
```bash
cd resume_screening
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 13. Road_Accident_Severity_Predictor

**Type:** Classification  
**Description:** A machine learning model that predicts the severity of road accidents based on various factors.

**Features:**
- Severity prediction
- Factor analysis
- Real-time predictions
- Data visualization

**Tech Stack:** Python, scikit-learn, Flask, pandas

**Run:**
```bash
cd Road_Accident_Severity_Predictor
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 14. Sentiment_Analysis

**Type:** NLP  
**Description:** Sentiment analysis on product reviews using machine learning to classify reviews as positive, negative, or neutral.

**Features:**
- Review analysis
- Sentiment classification
- Confidence scores
- Batch processing

**ML Concepts:**
- Text preprocessing
- TF-IDF Vectorization
- Machine learning classifiers

**Tech Stack:** Python, Flask, NLTK, scikit-learn

**Run:**
```bash
cd Sentiment_Analysis
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 15. spam_email_prediction

**Type:** NLP / Classification  
**Description:** A machine learning application that detects spam emails using text classification techniques.

**Features:**
- Email classification
- Spam detection
- Confidence scores
- Real-time predictions

**Tech Stack:** Python, Flask, scikit-learn, NLTK

**Run:**
```bash
cd spam_email_prediction
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 16. student_result_predictions

**Type:** Regression  
**Description:** A machine learning model that predicts student results based on various factors like attendance, study hours, and previous performance.

**Features:**
- Student performance prediction
- Factor analysis
- Excel integration
- Batch predictions

**Tech Stack:** Python, pandas, scikit-learn

**Run:**
```bash
cd student_result_predictions
python student_predictor_excel.py
```

---

### 17. Traffic_Sign_Recognition

**Type:** Computer Vision  
**Description:** A deep learning application that recognizes and classifies traffic signs using Convolutional Neural Networks.

**Features:**
- Real-time traffic sign detection
- Multiple sign categories
- High accuracy classification
- Web interface

**Model Architecture:**
- CNN with multiple convolutional layers
- MaxPooling layers
- Fully connected layers
- Softmax output

**Tech Stack:** TensorFlow/Keras, Flask, OpenCV

**Run:**
```bash
cd Traffic_Sign_Recognition
pip install -r requirements.txt
python train_model.py
python app.py
```

---

### 18. Dog Breed Classifier

*(Same as #4 - dog_bread_analyser)*

**Type:** Computer Vision  
**Description:** Dog breed identification using deep learning with transfer learning capabilities.

**Features:**
- Breed identification from images
- Multiple breed support
- Confidence scores
- Interactive UI

---

## 🛠️ Common Technologies Used

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap for responsive design

### Backend
- Flask (Python web framework)

### Machine Learning
- **TensorFlow/Keras** - Deep Learning
- **scikit-learn** - Traditional ML
- **NLTK** - Natural Language Processing
- **OpenCV** - Computer Vision
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### Other
- **pickle** - Model serialization
- **pyttsx3** - Text-to-speech
- **speech_recognition** - Voice recognition

---

## 📊 Project Categories Summary

| Category | Projects |
|----------|----------|
| **NLP** | AI_Interview_Bot, Fake_new_prediction, Sentiment_Analysis, resume_screening, spam_email_prediction |
| **Computer Vision** | dog_bread_analyser, Handwritten_Digit_Recognizer, Traffic_Sign_Recognition |
| **Classification** | Customer_Churn_Prediction, Medical_Disease_Prediction, Road_Accident_Severity_Predictor |
| **Regression** | house_prediction, student_result_predictions |
| **Recommendation** | Movie_Recommendation_System |
| **Analytics** | India_Rainfall_Analysis, Personal_Finance_Advisor |
| **Speech/AI** | AI_Voice_Assistant_mini_alexa |

---

## 🚀 Quick Start

1. **Navigate to any project:**
   ```bash
   cd project_name
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if needed):**
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   ```
   http://localhost:5000
   ```

---

## 📝 Notes

- Each project has its own detailed README.md file with specific instructions
- Some projects require additional setup (API keys, model training)
- Model files (*.pkl, *.h5) may need to be generated by running train_model.py
- Check individual project requirements.txt for specific dependencies

---

## 🎯 Learning Outcomes

This collection covers:
- ✅ Natural Language Processing (NLP)
- ✅ Computer Vision
- ✅ Classification & Regression
- ✅ Recommendation Systems
- ✅ Deep Learning with CNNs
- ✅ Flask Web Development
- ✅ Data Preprocessing
- ✅ Model Evaluation

---

## 📄 License

This project collection is for educational purposes.

---

**Created:** 2024  
**Author:** ML Projects Collection
