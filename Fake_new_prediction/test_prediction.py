import pickle
import sys
from fake_news_detector import TextCleaner

# Load the model
MODEL_PATH = 'fake_news_kaggle_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

vectorizer = model_data['vectorizer']
classifier = model_data['classifier']
label_map = model_data['label_map']

# Test news
news = 'Apple announces it will discontinue all iPhones starting next month.'

# Clean
cleaner = TextCleaner()
cleaned = cleaner.clean_text(news)

# Predict
text_tfidf = vectorizer.transform([cleaned])
prediction = classifier.predict(text_tfidf)[0]
confidence = classifier.predict_proba(text_tfidf)[0]

label = label_map.get(prediction, 'UNKNOWN')
confidence_score = max(confidence) * 100

print('='*50)
print('FAKE NEWS DETECTION TEST')
print('='*50)
print('News:', news)
print('Prediction:', label)
print('Confidence:', str(confidence_score) + '%')
print('FAKE probability:', str(confidence[0]*100) + '%')
print('REAL probability:', str(confidence[1]*100) + '%')
print('='*50)
