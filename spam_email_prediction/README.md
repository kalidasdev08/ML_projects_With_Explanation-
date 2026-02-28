# 📧 Spam Email Detector

A machine learning project for classifying emails as **Spam** or **Not Spam (Ham)** using NLP techniques.

## 🧠 Project Overview

This project demonstrates an NLP classification task using:
- **CountVectorizer**: For converting text emails into numerical feature vectors
- **Naive Bayes (MultinomialNB)**: For probabilistic classification

### Problem Statement
Email providers need to automatically filter spam emails. This project builds a classifier that can determine whether an email is spam or legitimate (ham).

### ML Type
- **Classification**: Binary classification (Spam vs Ham)
- **NLP**: Natural Language Processing for text classification

---

## 📁 Project Structure

```
spam_email_prediction/
├── app.py                    # Flask web application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── models/
│   ├── vectorizer.pkl        # Trained CountVectorizer
│   └── spam_classifier.pkl   # Trained Naive Bayes classifier
└── templates/
    └── index.html            # Web interface
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load/generate the spam email dataset
- Train the CountVectorizer and Naive Bayes classifier
- Evaluate model performance
- Save the trained model to `models/` directory

### 3. Run the Web App

```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## 🔧 How It Works

### Text Preprocessing
1. Convert text to lowercase
2. Remove English stop words
3. Extract unigrams and bigrams (ngram_range=(1,2))

### Feature Extraction
- **CountVectorizer** converts email text into a sparse matrix of token counts
- Maximum 1000 features are extracted

### Classification
- **MultinomialNB** (Naive Bayes) is used for classification
- Works well with text classification due to its probabilistic approach

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web page |
| `/predict` | POST | JSON API for predictions |
| `/predict_form` | POST | Form-based prediction |
| `/health` | GET | Health check |

### Example API Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"email": "Congratulations! You won a free lottery!"}'
```

Response:
```json
{
    "confidence": {
        "ham": 15.32,
        "spam": 84.68
    },
    "is_spam": true,
    "prediction": "spam",
    "status": "success"
}
```

---

## 📊 Model Performance

The model is trained on sample spam and ham emails with:
- **80%** training data
- **20%** test data

Key metrics include accuracy, precision, recall, and F1-score for both spam and ham classes.

---

## 🎯 Sample Test Cases

### Spam Examples
- "Congratulations! You've won a free lottery ticket!"
- "URGENT: Your account needs verification!"
- "Make $10,000 per month working from home!"

### Ham (Not Spam) Examples
- "Hi John, can we schedule a meeting?"
- "Thank you for your order confirmation."
- "Please find attached the report."

---

## 🛠️ Technologies Used

- **Python 3.x**: Programming language
- **Scikit-learn**: Machine learning library
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **CountVectorizer**: Text feature extraction
- **MultinomialNB**: Naive Bayes classifier

---

## 📝 License

This project is for educational purposes and can be used freely.

---

## 🔬 Model Details

### CountVectorizer Parameters
```python
CountVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=1000,
    ngram_range=(1, 2)  # Unigrams and bigrams
)
```

### Naive Bayes Parameters
```python
MultinomialNB(alpha=1.0)  # Laplace smoothing
```

---

## 💡 Learning Outcomes

This project demonstrates:
1. **Text preprocessing** for NLP
2. **Feature extraction** using CountVectorizer
3. **Naive Bayes classification** for spam detection
4. **Flask web app** deployment
5. **API development** for ML models

---

## 🚦 Status

| Component | Status |
|-----------|--------|
| Training Script | ✅ Complete |
| Web Application | ✅ Complete |
| Model | ✅ Ready to train |
| Documentation | ✅ Complete |

---

*Built as an intermediate-level portfolio project demonstrating NLP and ML skills.*
