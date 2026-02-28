# Sentiment Analysis on Product Reviews

A machine learning application that analyzes product reviews and classifies them as **Positive**, **Negative**, or **Neutral** using LSTM neural networks with word embeddings.

## 🎯 Problem Statement

Companies want to automatically analyze product feedback to understand customer satisfaction. This project provides an NLP-based solution to classify reviews into three categories:

- 😊 **Positive** - Customer is satisfied
- 😞 **Negative** - Customer is dissatisfied  
- 😐 **Neutral** - Customer has mixed or neutral feelings

## 🧠 ML Approach

### Architecture

The model uses **Bidirectional LSTM** neural networks with:

- **Word Embeddings**: 128-dimensional word vectors
- **Bidirectional LSTM**: Captures context from both directions
- **Dropout Layers**: Prevent overfitting
- **Dense Classification Layers**: Final sentiment prediction

### Features

1. **Text Preprocessing**
   - Lowercase conversion
   - URL and HTML removal
   - Special character cleaning
   - Stopword removal
   - Lemmatization

2. **Deep Learning Model**
   - Embedding layer for word representations
   - Two Bidirectional LSTM layers
   - Dense layers with dropout
   - Softmax output for 3-class classification

## 📁 Project Structure

```
Sentiment_Analysis on Product Reviews/
├── app.py                      # Flask web application
├── train_model.py             # Model training script
├── sentiment_analyzer.py      # Prediction logic
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Trained model files
│   ├── sentiment_lstm_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── config.json
│   └── training_history.json
└── templates/
    └── index.html             # Web interface
```

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
- Create sample training data
- Train the LSTM model
- Save model files to `models/` directory

### 3. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

## 📊 Usage

### Web Interface

1. Enter a product review in the text box
2. Click "Analyze Sentiment" or press Ctrl+Enter
3. View the predicted sentiment and confidence scores

### Programmatic Usage

```python
from sentiment_analyzer import load_analyzer

# Load the trained model
analyzer = load_analyzer('models')

# Analyze a single review
result = analyzer.predict("This product is amazing!")
print(result['sentiment'])  # Output: Positive

# Analyze multiple reviews
results = analyzer.predict_batch([
    "Great product!",
    "Terrible quality",
    "It's okay"
])
```

## 🔧 API Endpoints

### POST /analyze

Analyze a single review.

```json
// Request
{
  "review": "This product is amazing!"
}

// Response
{
  "success": true,
  "sentiment": "Positive",
  "confidence": 95.5,
  "emoji": "😊",
  "color": "#27ae60",
  "probabilities": {
    "Positive": 95.5,
    "Negative": 2.2,
    "Neutral": 2.3
  }
}
```

### POST /analyze_batch

Analyze multiple reviews.

```json
// Request
{
  "reviews": ["Great!", "Bad", "Okay"]
}

// Response
{
  "results": [...]
}
```

### GET /status

Check if model is loaded.

## 📈 Model Performance

The model is trained on a sample dataset and achieves:
- **Accuracy**: ~85-95% on test set
- **Good generalization** with Bidirectional LSTM

For production use, train on larger labeled datasets.

## 🎨 Demo Mode

If no trained model is available, the application runs in **demo mode** using keyword-based sentiment analysis. This demonstrates the UI while showing a notice that the LSTM model needs to be trained.

## 🔨 Customization

### Training on Your Data

Modify `train_model.py` to load your own dataset:

```python
# Load custom data
df = pd.read_csv('your_data.csv')
# Columns: 'review', 'sentiment' (0=Negative, 1=Positive, 2=Neutral)
```

### Model Hyperparameters

Adjust in `SentimentTrainer.__init__`:

```python
trainer = SentimentTrainer(
    max_words=10000,      # Vocabulary size
    max_len=100,          # Max sequence length
    embedding_dim=128     # Word embedding dimension
)
```

## 📝 Requirements

- Python 3.8+
- TensorFlow 2.10+
- Keras
- NLTK
- Flask
- Scikit-learn

## 🌟 Features

- ✅ LSTM-based deep learning
- ✅ Word embeddings
- ✅ Real-time web interface
- ✅ REST API
- ✅ Batch prediction
- ✅ Confidence scores
- ✅ Demo mode fallback

## 📄 License

MIT License

## 👤 Author

ML Projects

---

*For accurate predictions, please train the model with sufficient labeled data.*
