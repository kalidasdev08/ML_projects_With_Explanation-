# Project Run Guide

## Prerequisites
Make sure you have Python installed with the following packages:
```bash
pip install pandas numpy matplotlib flask scikit-learn joblib
```

## Running the Project

### Option 1: Run Exploratory Data Analysis (EDA)
This generates analysis and visualizations:
```bash
cd "Exploratory-Analysis-of-Rain-Fall-Data-in-India-for-Agriculture-main\Project Files"
python eda_india.py
```
**Output:** Creates `plots/` folder with PNG charts and `analysis_summary.txt`

### Option 2: Run Web Application (Flask)
This starts the rainfall prediction web app:
```bash
cd "Exploratory-Analysis-of-Rain-Fall-Data-in-India-for-Agriculture-main\Project Files"
python app.py
```
Then open browser: `http://127.0.0.1:5000`

### Option 3: Test ML Model
Verify the prediction model works:
```bash
cd "Exploratory-Analysis-of-Rain-Fall-Data-in-India-for-Agriculture-main\Project Files"
python verify_model.py
```

### Option 4: Test Web API
Test the web predictions (requires app.py running):
```bash
cd "Exploratory-Analysis-of-Rain-Fall-Data-in-India-for-Agriculture-main\Project Files"
python test_app_requests.py
```

---

## Quick Start Commands

| Task | Command |
|------|---------|
| Run EDA | `python eda_india.py` |
| Run Web App | `python app.py` |
| Test Model | `python verify_model.py` |
| API Test | `python test_app_requests.py` |

## Files Overview
- `eda_india.py` - Data analysis script
- `app.py` - Flask web application  
- `rainfall.pkl` - Trained ML model
- `scale.pkl` - Data scaler
- `rainfall_india_1901_2015.csv` - India rainfall dataset
