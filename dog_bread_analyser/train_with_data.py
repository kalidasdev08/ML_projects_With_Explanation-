"""
Dog Breed Analyzer - Training Script
Using Transfer Learning with MobileNetV2 on your custom dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import glob


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = 'train'
MODEL_PATH = 'models/dog_breed_model.keras'
LABELS_PATH = 'models/breed_labels.json'


def get_breed_list():
    """
    Get list of all breeds from the train directory
    """
    breeds = []
    for folder in os.listdir(TRAIN_DIR):
        folder_path = os.path.join(TRAIN_DIR, folder)
        if os.path.isdir(folder_path):
            # Clean up the breed name
            breed_name = folder.replace('_', ' ').title()
            breeds.append({
                'folder': folder,
                'name': breed_name
            })
    
    # Sort by folder name for consistency
    breeds.sort(key=lambda x: x['folder'])
    return breeds


def create_model(num_classes):
    """
    Create a CNN model using transfer learning with MobileNetV2
    """
    # Load pre-trained MobileNetV2 model without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build the model
    model = keras.Sequential([
        # Preprocessing input
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Base model (transfer learning)
        base_model,
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers for classification
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def prepare_data():
    """
    Prepare training and validation data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.15  # 15% for validation
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model():
    """
    Train the model with callbacks for best performance
    """
    # Get breed list
    breeds = get_breed_list()
    num_classes = len(breeds)
    
    print(f"\nFound {num_classes} breeds in dataset")
    print("Breeds:", [b['name'] for b in breeds[:10]], "...")
    
    # Create the model
    print("\n[1/5] Creating model with MobileNetV2 backbone...")
    model, base_model = create_model(num_classes)
    model.summary()
    
    # Prepare data
    print("\n[2/5] Loading and preparing data...")
    train_generator, val_generator = prepare_data()
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Number of classes: {num_classes}")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n[3/5] Training the model...")
    print("="*50)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save breed labels
    print("\n[4/5] Saving breed labels...")
    breed_labels = {str(i): breed['name'] for i, breed in enumerate(breeds)}
    breed_folders = {str(i): breed['folder'] for i, breed in enumerate(breeds)}
    
    # Save as JSON
    with open(LABELS_PATH, 'w') as f:
        json.dump(breed_labels, f, indent=2)
    
    # Also save class indices
    class_indices = train_generator.class_indices
    with open('models/class_indices.json', 'w') as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=2)
    
    # Fine-tune the model
    print("\n[5/5] Fine-tuning the model...")
    base_model.trainable = True
    
    # Freeze all but last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    print("\nFine-tuning with unfrozen layers...")
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(MODEL_PATH)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")
    print("="*50)
    
    return model, history


def main():
    """
    Main training function
    """
    print("="*60)
    print("Dog Breed Analyzer - Custom Training")
    print("Using Transfer Learning with MobileNetV2")
    print("="*60)
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model, history = train_model()


if __name__ == "__main__":
    main()
