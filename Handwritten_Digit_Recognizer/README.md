# Handwritten Digit Recognizer

A web-based application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with TensorFlow/Keras and Flask.

## 🎯 Problem Statement

This application solves the problem of digit recognition for forms and banks, where handwritten digits need to be automatically identified and processed. The system uses a deep learning model to accurately recognize digits drawn by users.

## 🏗️ Project Structure

```
Handwritten Digit Recognizer/
├── app.py                    # Flask web application
├── train_model.py           # CNN model training script
├── digit_recognizer.py       # Prediction logic
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── templates/
│   └── index.html           # Web UI with drawing canvas
└── models/
    └── digit_model.h5        # Trained CNN model (after training)
```

## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits directly in the browser
- **Image Upload**: Upload digit images for prediction
- **Real-time Prediction**: Get instant results with confidence scores
- **Probability Distribution**: View prediction probabilities for all digits (0-9)
- **CNN Model**: Deep learning model achieving ~99% accuracy on MNIST
- **Responsive UI**: Works on desktop and mobile devices

## 🛠️ Installation

1. **Clone or navigate to the project folder**:
   ```bash
   cd "Handwritten Digit Recognizer"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
Flask>=2.0.0
Werkzeug>=2.0.0
```

## 🔧 Usage

### Step 1: Train the Model

Before running the web application, you need to train the CNN model:

```bash
python train_model.py
```

This will:
- Download the MNIST dataset
- Train a CNN model with 3 convolutional layers
- Save the trained model to `models/digit_model.h5`
- Display training progress and final test accuracy

Expected training time: ~5-10 minutes on CPU

### Step 2: Run the Web Application

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000`

### Step 3: Use the Application

1. Open your browser and go to `http://127.0.0.1:5000`
2. Draw a digit (0-9) on the canvas using your mouse or touch
3. Click "Recognize" to get the prediction
4. View the predicted digit and confidence score
5. Use "Clear" to draw a new digit

You can also upload an image file instead of drawing.

## 🧠 Model Architecture

The CNN model consists of:

```
┌─────────────────────────────────────┐
│ Input: (28, 28, 1) grayscale image │
├─────────────────────────────────────┤
│ Conv2D(32 filters, 3x3) + ReLU     │
│ MaxPooling2D(2x2)                  │
├─────────────────────────────────────┤
│ Conv2D(64 filters, 3x3) + ReLU     │
│ MaxPooling2D(2x2)                  │
├─────────────────────────────────────┤
│ Conv2D(64 filters, 3x3) + ReLU     │
├─────────────────────────────────────┤
│ Flatten                             │
│ Dense(128) + ReLU + Dropout(0.3)   │
├─────────────────────────────────────┤
│ Dense(10) + Softmax                 │
│ Output: 10 classes (0-9)            │
└─────────────────────────────────────┘
```

**Performance:**
- Training accuracy: ~99%
- Test accuracy: ~99%
- Inference time: <10ms per image

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page |
| `/predict` | POST | Predict digit from image/canvas |
| `/predict_top_k` | POST | Get top-k predictions |
| `/health` | GET | Check model status |

### Predict Endpoint

**Request (Canvas Data):**
```json
{
  "canvas_data": "data:image/png;base64,..."
}
```

**Request (File Upload):**
```
POST /predict
Content-Type: multipart/form-data
Image: <file>
```

**Response:**
```json
{
  "digit": 5,
  "confidence": 0.9876,
  "probabilities": {
    "0": 0.001,
    "1": 0.002,
    "2": 0.003,
    "3": 0.004,
    "4": 0.005,
    "5": 0.987,
    "6": 0.001,
    "7": 0.001,
    "8": 0.001,
    "9": 0.001
  }
}
```

## 🎨 UI Features

- **Drawing Canvas**: 280x280 pixel canvas with smooth drawing
- **Touch Support**: Works on mobile/tablet devices
- **Clear Button**: Reset the canvas
- **Confidence Display**: Shows prediction confidence as percentage
- **Probability Bars**: Visual representation of all digit probabilities
- **File Upload**: Alternative to drawing for image files

## 🔍 Tips for Better Results

1. Draw digits clearly in the center of the canvas
2. Use a single stroke when possible
3. Keep the digit proportional (not too small or large)
4. For uploaded images: use clear, high-contrast images

## 🐛 Troubleshooting

**Model not found error:**
- Make sure you've run `python train_model.py` first
- Check that `models/digit_model.h5` exists

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**Canvas not working:**
- Use a modern browser (Chrome, Firefox, Edge)
- Enable JavaScript in your browser

## 📝 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- TensorFlow: https://www.tensorflow.org/
- Flask: https://flask.palletsprojects.com/
