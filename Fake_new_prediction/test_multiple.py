import pickle
from fake_news_detector import TextCleaner

MODEL_PATH = 'fake_news_kaggle_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

vectorizer = model_data['vectorizer']
classifier = model_data['classifier']
label_map = model_data['label_map']

# Test with different samples
samples = [
    ('BREAKING: Hillary Clinton caught in massive email scandal involving foreign governments and illegal activities', 'FAKE'),
    ('Scientists discover new species of deep sea fish in the Pacific Ocean during research expedition', 'REAL'),
    ('President announces new economic policy to boost job growth and help small businesses', 'REAL'),
    ('Alien spaceship spotted hovering over New York City claims eyewitness video', 'FAKE')
]

cleaner = TextCleaner()

print('='*60)
print('FAKE NEWS DETECTION - MULTIPLE TESTS')
print('='*60)

for news, expected in samples:
    cleaned = cleaner.clean_text(news)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = classifier.predict(text_tfidf)[0]
    confidence = classifier.predict_proba(text_tfidf)[0]
    label = label_map.get(prediction, 'UNKNOWN')
    
    print(f'\nNews: {news[:60]}...')
    print(f'Expected: {expected}')
    print(f'Prediction: {label} ({max(confidence)*100:.1f}%)')
    print(f'  FAKE: {confidence[0]*100:.1f}% | REAL: {confidence[1]*100:.1f}%')

print('\n' + '='*60)
