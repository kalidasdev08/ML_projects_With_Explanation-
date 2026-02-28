# 🚦 Traffic Sign Recognition - Deep Learning CNN

A Deep Learning project for recognizing traffic signs using Convolutional Neural Networks (CNN), designed for self-driving car applications.

## 📋 Project Overview

This project implements a CNN-based traffic sign recognition system that can classify 43 different types of traffic signs. The model is trained using TensorFlow/Keras with extensive image augmentation to improve generalization.

### 🎯 Problem Statement

Self-driving cars must be able to recognize and respond to road signs in real-time. This project implements a computer vision solution using deep learning to accurately identify traffic signs from images.

### 🔧 Skills Demonstrated

- **TensorFlow/Keras** - Deep learning framework
- **Convolutional Neural Networks (CNN)** - Image classification architecture
- **Image Augmentation** - Data preprocessing and enhancement
- **Flask** - Web application framework
- **OpenCV** - Image processing

## 📁 Project Structure

```
Traffic_Sign_Recognition (Computer Vision)/
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── traffic_sign_recognition.py     # Model inference class
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── templates/
│   └── index.html                  # Web interface
└── models/
    ├── traffic_sign_model.keras    # Trained model
    ├── class_names.csv             # Class labels
    └── training_history.png        # Training curves
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- 8GB RAM minimum (16GB recommended for training)

### Installation

1. **Clone or download the project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the GTSRB Dataset** (optional):
   - Download from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
   - Extract to a folder named `GTSRB`

### Training the Model

To train the CNN model:

```bash
python train_model.py
```

The training script will:
- Load/Generate training data
- Apply data augmentation
- Train the CNN model
- Save the trained model to `models/traffic_sign_model.keras`
- Generate training history plots

### Running the Web Application

Start the Flask server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Using the Recognition API

```python
from traffic_sign_recognition import TrafficSignRecognizer

# Initialize recognizer
recognizer = TrafficSignRecognizer('models/traffic_sign_model.keras')

# Predict traffic sign
result = recognizer.predict('traffic_sign.jpg')

# Get top prediction
print(result['predictions'][0]['class_name'])
print(result['predictions'][0]['confidence'])
```

## 🧠 Model Architecture

The CNN model consists of:

```
┌─────────────────────────────────────────┐
│         Input (48x48x3 RGB)             │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Conv2D(32) → BatchNorm → ReLU         │
│  Conv2D(32) → BatchNorm → ReLU         │
│  MaxPool(2x2) → Dropout(0.25)          │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Conv2D(64) → BatchNorm → ReLU         │
│  Conv2D(64) → BatchNorm → ReLU         │
│  MaxPool(2x2) → Dropout(0.25)          │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Conv2D(128) → BatchNorm → ReLU        │
│  Conv2D(128) → BatchNorm → ReLU        │
│  MaxPool(2x2) → Dropout(0.25)          │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Flatten                                │
│  Dense(256) → BatchNorm → ReLU         │
│  Dropout(0.5)                          │
│  Dense(128) → BatchNorm → ReLU         │
│  Dropout(0.5)                          │
│  Dense(43) → Softmax                    │
└─────────────────────────────────────────┘
```

## 🖼️ Image Augmentation

The following augmentation techniques are applied during training:

- **Rotation**: ±15 degrees
- **Width Shift**: ±15%
- **Height Shift**: ±15%
- **Zoom**: 0.2x
- **Brightness**: ±20%

## 📊 Supported Traffic Signs (43 Classes)

| ID | Sign Name | ID | Sign Name |
|----|-----------|----|-----------|
| 0 | Speed Limit 20 | 22 | Bumpy Road |
| 1 | Speed Limit 30 | 23 | Slippery Road |
| 2 | Speed Limit 50 | 24 | Road Narrows |
| 3 | Speed Limit 60 | 25 | Road Work |
| 4 | Speed Limit 70 | 26 | Traffic Signals |
| 5 | Speed Limit 80 | 27 | Pedestrians |
| 6 | End of Speed Limit 80 | 28 | Children Crossing |
| 7 | Speed Limit 100 | 29 | Bicycles Crossing |
| 8 | Speed Limit 120 | 30 | Beware of Ice/Snow |
| 9 | No Passing | 31 | Wild Animals |
| 10 | No Passing (3.5t) | 32 | End of All Restrictions |
| 11 | Right-of-Way | 33 | Turn Right Ahead |
| 12 | Priority Road | 34 | Turn Left Ahead |
| 13 | Yield | 35 | Ahead Only |
| 14 | Stop | 36 | Go Straight or Right |
| 15 | No Vehicles | 37 | Go Straight or Left |
| 16 | No Vehicles (3.5t) | 38 | Keep Right |
| 17 | No Entry | 39 | Keep Left |
| 18 | General Danger | 40 | Roundabout |
| 19 | Curve Left | 41 | End of No Passing |
| 20 | Curve Right | 42 | End of No Passing (3.5t) |
| 21 | Double Curve | | |

## 🎨 Web Interface Features

- **Drag & Drop Upload**: Easy image upload
- **Real-time Preview**: See uploaded images
- **Top 5 Predictions**: View multiple predictions
- **Confidence Scores**: Visual confidence bars
- **Responsive Design**: Works on all devices

## 📈 Performance

The model achieves:
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%

## 🔮 Future Enhancements

- [ ] Real-time camera input support
- [ ] Mobile app integration
- [ ] Model optimization for edge deployment
- [ ] YOLO object detection integration
- [ ] Transfer learning with ResNet/VGG

## 📝 License

This project is for educational purposes.

## 👏 Acknowledgments

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/)
- TensorFlow/Keras Documentation
- Various open-source contributors

---

**Built with ❤️ using TensorFlow and Flask**
