# Medical Disease Prediction from Symptoms

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Decision%20Tree-orange.svg)

## Problem Statement

**Early diagnosis in rural areas** - This project aims to provide accessible medical disease prediction to help people in rural and underserved areas get preliminary health assessments without immediate access to healthcare professionals.

## Overview

Medical Disease Prediction from Symptoms is a machine learning-powered web application that helps users identify potential diseases based on their symptoms. The system uses a Decision Tree classifier for accurate and interpretable predictions, with an integrated chatbot interface for natural language interaction.

## Features

- **Symptom-based Disease Prediction**: Enter your symptoms and get predicted diseases
- **Chatbot Interface**: Natural language conversation with the AI health assistant
- **Multiple Symptom Categories**: Filter symptoms by category (Common, Respiratory, Digestive, Pain, Skin)
- **Probability Scores**: Shows confidence levels for each prediction
- **User-friendly UI**: Clean, modern interface with responsive design
- **Healthcare Disclaimer**: Clear warnings that this is for preliminary guidance only

## Supported Diseases

The model can predict the following conditions:
- Common Cold
- Flu (Influenza)
- Viral Fever
- Dengue
- Malaria
- Typhoid
- COVID-19
- Pneumonia
- Bronchitis
- Gastritis
- Food Poisoning
- Migraine
- Allergy
- Skin Infection
- Ear Infection
- Eye Infection
- Arthritis
- Anemia
- Stress/Anxiety
- Dehydration

*Expanded to 30 diseases with enhanced Kaggle training:*
- Plus: Asthma, Sinusitis, Eczema, Hypothyroidism, Diabetes Type 2, Hypertension, Gastroenteritis, UTI, Kidney Stone, Depression, Anxiety

## Supported Symptoms

The system analyzes 60+ symptoms including:
- General: Fever, Cough, Headache, Fatigue, Body Ache, Chills, Sweating
- Digestive: Nausea, Vomiting, Diarrhea, Constipation, Abdominal Pain, Indigestion
- Respiratory: Sore Throat, Runny Nose, Sneezing, Congestion, Sputum, Wheezing
- Pain: Chest Pain, Joint Pain, Ear Pain, Severe Pain
- Skin: Rash, Skin Lesion, Itching, Redness, Swelling, Dry Skin
- Others: Dizziness, Shortness of Breath, Eye Discharge, and more...

## Project Structure

```
Medical_Disease_Prediction/
├── app.py                     # Flask web application
├── train_model.py             # Basic model training script
├── train_with_kaggle.py       # Enhanced model with Kaggle dataset
├── disease_predictor.py       # Prediction logic
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/
│   ├── disease_model.pkl      # Trained ML model
│   ├── symptom_list.pkl       # List of symptoms
│   ├── symptom_descriptions.pkl # Symptom details
│   └── training_metadata.json # Training information
└── templates/
    └── index.html             # Web interface
```

## Installation

1. **Clone the repository or navigate to the project directory:**
   ```bash
   cd Medical_Disease_Prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Basic Model (20 Diseases)

Train the basic model:
```bash
python train_model.py
```

### Option 2: Enhanced Model with Kaggle Dataset (30+ Diseases)

For improved accuracy with more diseases:
```bash
python train_with_kaggle.py
```

This version:
- Uses enhanced synthetic dataset (1500+ samples)
- Trains 30+ diseases
- Compares Decision Tree, Random Forest, and Gradient Boosting
- Selects the best model automatically
- Requires Kaggle API setup (optional)

#### To use real Kaggle data:
1. Get a Kaggle API token:
   - Go to Kaggle.com → Account → Create API Token
   - Download kaggle.json
   - Place in `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Or `~/.kaggle/kaggle.json` (Linux/Mac)

2. Update the dataset name in `train_with_kaggle.py`:
   ```python
   dataset_name = "your-chosen-dataset"
   ```

### Running the Web Application

After training the model:
```bash
python app.py
```

### Step 3: Use the Application

1. Open your browser and navigate to `http://localhost:5000`
2. You can:
   - **Chat with the AI**: Type symptoms in natural language (e.g., "I have fever and cough")
   - **Select symptoms**: Check the symptoms from the grid and click "Analyze Symptoms"
   - **Use quick actions**: Click common symptom sets or clear selections

## How It Works

### Machine Learning Model

- **Algorithms**: Decision Tree, Random Forest, Gradient Boosting (with auto-selection)
- **Features**: 20-60+ binary features representing different symptoms
- **Training Data**: 400-1500+ samples with 20-30+ disease categories
- **Preprocessing**: Symptom text normalized and converted to binary vectors

### Prediction Process

1. User inputs symptoms (via chat or checkboxes)
2. Symptoms converted to feature vector
3. Model predicts disease based on symptom patterns
4. Results displayed with confidence scores

### Chatbot Logic

- Handles greetings, help requests, and general queries
- Extracts symptoms from natural language
- Provides symptom information on request
- Formats predictions in friendly responses

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Get disease prediction |
| `/chat` | POST | Chat with AI assistant |
| `/symptoms` | GET | Get list of symptoms |
| `/health` | GET | Health check |

## Example Usage

### Via Chat:
```
User: I have fever and cough
Bot: Primary Prediction: Flu (Influenza)
     Confidence: 85%
     Possible Conditions:
     1. Flu (Influenza) (85%)
     2. COVID-19 (10%)
     3. Pneumonia (5%)
```

### Via Symptom Selection:
1. Select: Fever, Cough, Body Ache, Headache
2. Click "Analyze Symptoms"
3. View predicted diseases with probabilities

## Important Notes

⚠️ **Medical Disclaimer**: This application is for educational and preliminary assessment purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for proper medical care.

## Future Enhancements

- [ ] Integration with more advanced ML models (Random Forest, XGBoost)
- [ ] Support for more diseases and symptoms
- [ ] Multi-language support for rural areas
- [ ] Offline mobile app version
- [ ] Integration with telemedicine platforms
- [ ] Symptom severity analysis

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Decision Tree Classifier

## License

This project is for educational purposes. Use at your own risk for preliminary health assessment.

## Author

Created for improving healthcare accessibility in rural areas through AI-powered symptom analysis.

---

*Remember: This tool provides preliminary guidance only. Always seek professional medical advice for proper diagnosis and treatment.*
