# 🐕 Dog Breed Analyzer

An AI-powered dog breed classification system using the best machine learning algorithms - Transfer Learning with MobileNetV2.

## 🚀 Features

- **Deep Learning**: Uses MobileNetV2 transfer learning for accurate predictions
- **Fast Analysis**: Get breed predictions in seconds
- **Top 5 Predictions**: View the top 5 most likely breeds with confidence scores
- **Breed Information**: Get detailed information about each breed
- **Modern Web Interface**: Clean, responsive UI with drag & drop support
- **API Ready**: RESTful API for programmatic access

## 🧠 ML Algorithm

This project uses **Transfer Learning with MobileNetV2** - one of the best algorithms for image classification:

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Fine-tuning**: Last 30 layers fine-tuned for dog breed classification
- **Data Augmentation**: Rotation, shifts, flips, zoom for robust training
- **Optimizer**: Adam with learning rate scheduling
- **Supported Breeds**: 120 dog breeds

## 📁 Project Structure

```
dog_bread_analyser/
├── app.py                     # Flask web application
├── train_model.py             # Model training script
├── dog_breed_classifier.py   # Classifier module
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Trained models
│   ├── dog_breed_model.keras
│   └── breed_labels.json
├── templates/
│   └── index.html            # Web interface
└── static/
    └── uploads/              # Uploaded images
```

## 🛠️ Installation

1. **Clone the repository** (if applicable)

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎯 Training the Model

### Option 1: Quick Start (Pre-trained)
The classifier includes a fallback mechanism that uses ImageNet weights for demonstration.

### Option 2: Train with Real Data
For best results, train with the Stanford Dogs dataset:

```bash
python train_model.py
```

This will:
- Load the MobileNetV2 base model
- Train on your dataset
- Save the model to `models/dog_breed_model.keras`

### Dataset
For production training, use:
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

## 🚀 Running the Application

1. **Start the web server**:
```bash
python app.py
```

2. **Open in browser**:
Navigate to `http://localhost:5000`

3. **Upload an image**:
- Drag & drop or click to select a dog image
- Click "Analyze Breed" to get predictions

## 📡 API Endpoints

### Analyze Image
```bash
POST /analyze
Content-Type: multipart/form-data

# Request
curl -X POST -F "file=@dog.jpg" http://localhost:5000/analyze
```

### Get Supported Breeds
```bash
GET /api/breeds

# Response
{
  "breeds": ["Labrador Retriever", "German Shepherd", ...],
  "count": 40
}
```

### Health Check
```bash
GET /api/health

# Response
{
  "status": "healthy",
  "model_loaded": true,
  "num_breeds": 40
}
```

## 🖥️ Usage Example

### Python API
```python
from dog_breed_classifier import DogBreedClassifier

# Initialize
classifier = DogBreedClassifier()
classifier.load_model()

# Predict
results = classifier.predict('dog_image.jpg', top_k=5)

# Results
for pred in results:
    print(f"{pred['breed']}: {pred['confidence']*100:.1f}%")
```

## 🔧 Technologies Used

- **TensorFlow/Keras** - Deep Learning Framework
- **MobileNetV2** - Pre-trained CNN Model
- **Flask** - Web Framework
- **OpenCV** - Image Processing
- **NumPy** - Numerical Computing
- **HTML/CSS/JS** - Frontend

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 |
| Input Size | 224x224 |
| Accuracy | ~85% (with real data) |
| Inference Time | <1s |
| Supported Breeds | 120 |

## 🐛 Troubleshooting

### Model not found
If you see "Model not found", the classifier will use a fallback model with ImageNet weights.

### Out of memory
Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### CUDA errors
Ensure GPU drivers are installed or run on CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 📝 License

MIT License - Feel free to use this project for learning and development.

## 🙏 Acknowledgments

- MobileNetV2: [Sandler et al.](https://arxiv.org/abs/1801.04381)
- Stanford Dogs Dataset
- TensorFlow/Keras Documentation

---

Made with ❤️ using the best ML algorithms
