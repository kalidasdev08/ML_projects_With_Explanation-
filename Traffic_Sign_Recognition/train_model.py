"""
Traffic Sign Recognition - Model Training
Deep Learning CNN for Self-Driving Cars

This script trains a CNN model to classify traffic signs using the 
German Traffic Sign Recognition Benchmark (GTSRB) dataset.
"""

import os
import numpy as np
import pandas as pd

# Try to import cv2, but make it optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Using PIL fallback.")

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Demo mode only.")

from datetime import datetime

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_DIR = 'GTSRB'
    MODEL_DIR = 'models'
    
    # Image parameters
    IMG_SIZE = 48  # Resize images to 48x48
    NUM_CHANNELS = 3  # RGB images
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Number of classes (43 traffic sign classes in GTSRB)
    NUM_CLASSES = 43


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_traffic_sign_data(data_dir):
    """
    Load traffic sign data from the GTSRB dataset.
    
    The dataset contains:
    - Training images in separate folders per class
    - CSV file with test labels
    
    Args:
        data_dir: Path to the GTSRB dataset directory
        
    Returns:
        X_train, y_train: Training images and labels
        X_test, y_test: Test images and labels
    """
    print("Loading traffic sign data...")
    
    # Try to load from GTSRB folder structure
    train_dir = os.path.join(data_dir, 'train')
    test_csv = os.path.join(data_dir, 'test.csv')
    
    X_train = []
    y_train = []
    
    # Check if GTSRB data exists
    if os.path.exists(train_dir) and CV2_AVAILABLE:
        # Load training data from folders
        for class_id in range(Config.NUM_CLASSES):
            class_dir = os.path.join(train_dir, str(class_id))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                        X_train.append(img)
                        y_train.append(class_id)
                    except:
                        continue
        print(f"Loaded {len(X_train)} training images from folder structure")
    
    # If no folder structure, generate synthetic data for demonstration
    if len(X_train) == 0:
        print("Generating synthetic training data for demonstration...")
        X_train, y_train = generate_synthetic_data()
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Load test data
    X_test = []
    y_test = []
    
    if os.path.exists(test_csv) and CV2_AVAILABLE:
        # Load test data from CSV
        test_df = pd.read_csv(test_csv)
        for _, row in test_df.iterrows():
            try:
                img = cv2.imread(row['Path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                X_test.append(img)
                y_test.append(row['ClassId'])
            except:
                continue
        print(f"Loaded {len(X_test)} test images")
    else:
        # Generate synthetic test data
        X_test, y_test = generate_synthetic_data(test=True)
        print(f"Generated {len(X_test)} synthetic test images")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test


def generate_synthetic_data(test=False):
    """
    Generate synthetic traffic sign data for demonstration.
    Creates sample images with different colors and shapes.
    """
    np.random.seed(42 if not test else 123)
    
    num_samples = 5000 if not test else 1000
    num_classes = Config.NUM_CLASSES
    
    images = []
    labels = []
    
    # Define traffic sign colors (RGB)
    sign_colors = [
        (255, 0, 0),      # Red - stop, prohibition
        (0, 0, 255),      # Blue - information, mandatory
        (255, 255, 0),    # Yellow - warning
        (0, 255, 0),      # Green - information
        (255, 165, 0),    # Orange - warning
        (128, 0, 128),    # Purple - temporary
    ]
    
    for i in range(num_samples):
        # Random class
        class_id = np.random.randint(0, num_classes)
        
        # Create a simple traffic sign-like image
        img = np.ones((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8) * 240
        
        # Add colored circle (traffic sign shape)
        color = sign_colors[class_id % len(sign_colors)]
        center = (Config.IMG_SIZE // 2, Config.IMG_SIZE // 2)
        radius = Config.IMG_SIZE // 3
        
        if CV2_AVAILABLE:
            # Draw colored circle
            cv2.circle(img, center, radius, color, -1)
            
            # Add inner white circle for some signs
            if class_id % 3 == 0:
                cv2.circle(img, center, radius // 2, (255, 255, 255), -1)
            
            # Add some random noise for variation
            noise = np.random.randint(-20, 20, (Config.IMG_SIZE, Config.IMG_SIZE, 3))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Apply random rotation
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (Config.IMG_SIZE, Config.IMG_SIZE))
        else:
            # Use PIL for drawing
            from PIL import Image, ImageDraw
            img_pil = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (240, 240, 240))
            draw = ImageDraw.Draw(img_pil)
            
            # Draw colored circle
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, fill=color)
            
            # Add inner white circle for some signs
            if class_id % 3 == 0:
                inner_radius = radius // 2
                inner_bbox = [center[0] - inner_radius, center[1] - inner_radius, 
                             center[0] + inner_radius, center[1] + inner_radius]
                draw.ellipse(inner_bbox, fill=(255, 255, 255))
            
            # Convert to numpy array
            img = np.array(img_pil)
        
        images.append(img)
        labels.append(class_id)
    
    return np.array(images), np.array(labels)


def preprocess_data(X, y=None):
    """
    Preprocess images for the model.
    
    Args:
        X: Image data
        y: Labels (optional)
        
    Returns:
        X_normalized: Normalized images
        y_encoded: Encoded labels (if provided)
    """
    # Normalize pixel values to [0, 1]
    X_normalized = X.astype('float32') / 255.0
    
    if y is not None:
        # One-hot encode labels
        y_encoded = to_categorical(y, Config.NUM_CLASSES)
        return X_normalized, y_encoded
    
    return X_normalized


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def create_data_augmentation():
    """
    Create an ImageDataGenerator with augmentation for training.
    
    Augmentation techniques:
    - Random rotation
    - Random width/height shift
    - Random zoom
    - Random brightness adjustment
    - Random horizontal flip
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,  # Traffic signs shouldn't be flipped
        fill_mode='nearest',
        preprocessing_function=lambda x: x  # Can add more preprocessing here
    )
    
    # Validation data should not be augmented
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen


# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================

def build_cnn_model(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_CHANNELS),
                   num_classes=Config.NUM_CLASSES):
    """
    Build a CNN model for traffic sign classification.
    
    Architecture:
    - 3 convolutional blocks with batch normalization
    - Max pooling and dropout for regularization
    - Dense layers with softmax output
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """
    Main training function.
    """
    print("=" * 60)
    print("Traffic Sign Recognition - Model Training")
    print("=" * 60)
    
    # Create model directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    X_train, y_train, X_test, y_test = load_traffic_sign_data(Config.DATA_DIR)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    # Preprocess data
    print("\n[2/5] Preprocessing data...")
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )
    print(f"Training split: {len(X_train_split)}")
    print(f"Validation split: {len(X_val)}")
    
    # Create data generators
    print("\n[3/5] Setting up data augmentation...")
    train_datagen, val_datagen = create_data_augmentation()
    
    # Build model
    print("\n[4/5] Building CNN model...")
    model = build_cnn_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, 'traffic_sign_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n[5/5] Training the model...")
    print("-" * 40)
    
    history = model.fit(
        train_datagen.flow(X_train_split, y_train_split, batch_size=Config.BATCH_SIZE),
        steps_per_epoch=len(X_train_split) // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=val_datagen.flow(X_val, y_val, batch_size=Config.BATCH_SIZE),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save training history plot
    plot_training_history(history)
    
    # Save class names
    save_class_names()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {os.path.join(Config.MODEL_DIR, 'traffic_sign_model.keras')}")
    print("=" * 60)
    
    return model, history


def plot_training_history(history):
    """
    Plot training history (accuracy and loss curves).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {os.path.join(Config.MODEL_DIR, 'training_history.png')}")


def save_class_names():
    """
    Save traffic sign class names.
    """
    # GTSRB class names
    class_names = [
        'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60',
        'Speed Limit 70', 'Speed Limit 80', 'End of Speed Limit 80', 'Speed Limit 100',
        'Speed Limit 120', 'No passing', 'No passing for vehicles over 3.5 tons',
        'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 tons prohibited', 'No entry',
        'General danger', 'Curve left', 'Curve right', 'Double curve',
        'Bumpy road', 'Slippery road', 'Road narrows', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals', 'End of all restrictions',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout',
        'End of no passing', 'End of no passing for vehicles over 3.5 tons'
    ]
    
    # Save as CSV
    df = pd.DataFrame({'class_id': range(len(class_names)), 'class_name': class_names})
    df.to_csv(os.path.join(Config.MODEL_DIR, 'class_names.csv'), index=False)
    print(f"Class names saved to {os.path.join(Config.MODEL_DIR, 'class_names.csv')}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model, history = train_model()
