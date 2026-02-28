"""
Dog Breed Analyzer - Classifier Module
Using the best ML algorithm: Transfer Learning with MobileNetV2
"""

import os
import numpy as np
import json
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras


class DogBreedClassifier:
    """
    Dog Breed Classifier using Transfer Learning
    Best ML algorithm for image classification tasks
    """
    
    def __init__(self, model_path='models/dog_breed_model.keras', labels_path='models/breed_labels.json'):
        """
        Initialize the classifier
        """
        self.IMG_SIZE = 224
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.breed_labels = {}
        self.is_loaded = False
        
    def load_model(self):
        """
        Load the trained model and breed labels
        Priority: 1) Custom trained model, 2) ImageNet pre-trained model
        """
        try:
            # First, try to load custom trained model
            if os.path.exists(self.model_path):
                print(f"Loading custom trained model from {self.model_path}...")
                self.model = keras.models.load_model(self.model_path)
                print("Custom model loaded successfully!")
                
                # Load breed labels from custom model
                if os.path.exists(self.labels_path):
                    with open(self.labels_path, 'r') as f:
                        self.breed_labels = json.load(f)
                    print(f"Loaded {len(self.breed_labels)} breed labels from custom model")
                else:
                    # Try to load class indices
                    if os.path.exists('models/class_indices.json'):
                        with open('models/class_indices.json', 'r') as f:
                            class_indices = json.load(f)
                        self._load_labels_from_indices(class_indices)
                    else:
                        print("No labels found. Using ImageNet fallback.")
                        self._load_imagenet_labels()
            else:
                # Use ImageNet pre-trained model
                print("No custom model found. Using ImageNet pre-trained model...")
                self.model = self._create_imagenet_model()
                self._load_imagenet_labels()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to ImageNet
            try:
                print("Falling back to ImageNet model...")
                self.model = self._create_imagenet_model()
                self._load_imagenet_labels()
                self.is_loaded = True
                return True
            except:
                return False
    
    def _load_labels_from_indices(self, class_indices):
        """
        Load breed labels from class indices
        """
        # Create breed name from folder name
        self.breed_labels = {}
        for idx, folder in class_indices.items():
            # Clean up folder name to get breed name
            breed_name = folder.replace('_', ' ').title()
            self.breed_labels[idx] = breed_name
        print(f"Loaded {len(self.breed_labels)} breed labels from class indices")
    
    def _create_imagenet_model(self):
        """
        Create a model using MobileNetV2 with ImageNet weights
        """
        # Load MobileNetV2 with full classification head
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)
        )
        return base_model
    
    def _load_imagenet_labels(self):
        """
        Load ImageNet class labels including dog breeds
        """
        # ImageNet class indices for dog breeds (151-268 are dog breeds)
        # We use a subset of common dog breeds from ImageNet
        self.breed_labels = {
            # Dog breeds from ImageNet
            "151": "Chihuahua",
            "153": "Maltese Dog",
            "155": "Shih-Tzu",
            "156": "Blenheim Spaniel",
            "158": "Toy Poodle",
            "159": "Miniature Poodle",
            "160": "Standard Poodle",
            "161": "Tibetan Terrier",
            "162": "Lhasa Apso",
            "163": "Old English Sheepdog",
            "164": "Shetland Sheepdog",
            "165": "Collie",
            "166": "Border Collie",
            "167": "Rottweiler",
            "168": "German Shepherd",
            "169": "Doberman",
            "170": "Miniature Pinscher",
            "171": "Great Dane",
            "172": "Saint Bernard",
            "173": "Eskimo Dog",
            "174": "Malamute",
            "175": "Siberian Husky",
            "176": "Affenpinscher",
            "177": "Basenji",
            "178": "Pug",
            "179": "Leonberger",
            "180": "Newfoundland",
            "181": "Samoyed",
            "182": "Pomeranian",
            "183": "Chow Chow",
            "184": "Keeshond",
            "185": "Brussels Griffon",
            "187": "Pembroke",
            "188": "Cardigan",
            "189": "Toy Spaniel",
            "190": "English Springer Spaniel",
            "191": "Welsh Springer Spaniel",
            "192": "Cocker Spaniel",
            "193": "Sussex Spaniel",
            "194": "Irish Water Spaniel",
            "195": "Dalmatian",
            "196": "Akita",
            "197": "Great Pyrenees",
            "198": "Samoyed",
            "199": "French Bulldog",
            "200": "Boxer",
            "201": "Bull Terrier",
            "202": "Staffordshire Bullterrier",
            "205": "Labrador Retriever",
            "206": "Chesapeake Bay Retriever",
            "207": "Curly-coated Retriever",
            "208": "Flat-coated Retriever",
            "209": "Golden Retriever",
            "210": "English Setter",
            "211": "Irish Setter",
            "213": "Brittany",
            "215": "Clumber",
            "216": "English Springer Spaniel",
            "217": "Welsh Springer Spaniel",
            "218": "Cocker Spaniel",
            "219": "Sussex Spaniel",
            "220": "Irish Water Spaniel",
            "221": "Kuvasz",
            "222": "Otterhound",
            "223": "Saluki",
            "224": "Scottish Deerhound",
            "225": "Weimaraner",
            "226": "Staffordshire Bullterrier",
            "227": "American Staffordshire Terrier",
            "228": "Bedlington Terrier",
            "229": "Border Terrier",
            "230": "Kerry Blue Terrier",
            "231": "Irish Terrier",
            "232": "Norfolk Terrier",
            "233": "Norwich Terrier",
            "234": "Yorkshire Terrier",
            "235": "Wire-haired Fox Terrier",
            "236": "Lakeland Terrier",
            "237": "Sealyham Terrier",
            "238": "Airedale Terrier",
            "239": "Cairn Terrier",
            "240": "Australian Terrier",
            "241": "Dandie Dinmont Terrier",
            "242": "Boston Terrier",
            "243": "Miniature Schnauzer",
            "244": "Giant Schnauzer",
            "245": "Standard Schnauzer",
            "246": "Scottish Terrier",
            "247": "Tibetan Mastiff",
            "248": "Tibetan Spaniel",
            "249": "Japanese Chin",
            "250": "Japanese Spaniel",
            "251": "Komondor",
            "252": "Kuvasz",
            "253": "Leonberger",
            "254": "Mastiff",
            "255": "Mexican Hairless",
            "256": "Newfoundland",
            "257": "Old English Sheepdog",
            "258": "Otterhound",
            "259": "Pekingese",
            "260": "Pembroke",
            "261": "Pomeranian",
            "262": "Pug",
            "263": "Saint Bernard",
            "264": "Samoyed",
            "265": "Schipperke",
            "266": "Shiba Inu",
            "267": "Shih-Tzu",
            "268": "Tibetan Terrier"
        }
        
        # Compile with pretrained weights already loaded via base
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL
                img = Image.open(image_path)
                img = np.array(img)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_image_from_array(self, img_array):
        """
        Preprocess image from numpy array
        """
        try:
            # Resize to model input size
            img = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Handle different channel formats
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path, top_k=5):
        """
        Predict dog breed from image
        Returns list of top predictions with probabilities
        """
        if not self.is_loaded:
            self.load_model()
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Check if using custom model or ImageNet
        if hasattr(self, 'model') and self.model is not None:
            # Check if it's the ImageNet model (has 'predictions' attribute from top layer)
            # or a custom model
            try:
                # Try with ImageNet preprocessing first
                img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
                predictions = self.model.predict(img_processed, verbose=0)[0]
            except:
                # Custom model - use simple preprocessing
                predictions = self.model.predict(img, verbose=0)[0]
        else:
            # Fallback
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
            predictions = self.model.predict(img, verbose=0)[0]
        
        # Check if it's ImageNet model (1000 classes) or custom model
        num_classes = len(predictions)
        
        if num_classes > 300:
            # ImageNet model - filter for dog breeds only (class indices 151-268)
            dog_breed_indices = list(range(151, 269))
            
            # Get top predictions and filter for dog breeds
            top_indices = np.argsort(predictions)[-100:][::-1]
            
            results = []
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                
                # Check if it's a dog breed
                if idx in dog_breed_indices:
                    breed_name = self.breed_labels.get(str(idx), self._get_imagenet_class_name(idx))
                    results.append({
                        'breed': breed_name,
                        'confidence': float(predictions[idx]),
                        'index': int(idx)
                    })
            
            # If no dog breeds found, show top predictions
            if len(results) == 0:
                for idx in top_indices[:top_k]:
                    breed_name = self._get_imagenet_class_name(idx)
                    results.append({
                        'breed': breed_name,
                        'confidence': float(predictions[idx]),
                        'index': int(idx)
                    })
        else:
            # Custom model - use breed labels directly
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                breed_name = self.breed_labels.get(str(idx), f"Breed {idx}")
                results.append({
                    'breed': breed_name,
                    'confidence': float(predictions[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def _get_imagenet_class_name(self, idx):
        """
        Get ImageNet class name for index
        """
        # Full ImageNet dog breed class names
        imagenet_names = {
            # Terrier group
            151: "Chihuahua", 152: "Silky Terrier", 153: "Maltese Dog", 
            154: "Shih-Tzu", 155: "Shih-Tzu", 156: "Blenheim Spaniel",
            157: "Australian Terrier", 158: "Toy Poodle", 159: "Miniature Poodle", 
            160: "Standard Poodle", 161: "Tibetan Terrier", 162: "Lhasa Apso", 
            
            # Herding group
            163: "Old English Sheepdog", 164: "Shetland Sheepdog", 165: "Collie", 
            166: "Border Collie", 187: "Pembroke Welsh Corgi", 188: "Cardigan Welsh Corgi",
            
            # Working group
            167: "Rottweiler", 168: "German Shepherd", 169: "Doberman", 
            170: "Miniature Pinscher", 171: "Great Dane", 172: "Saint Bernard",
            173: "Eskimo Dog", 174: "Malamute", 175: "Siberian Husky",
            179: "Leonberger", 180: "Newfoundland", 181: "Samoyed",
            
            # Toy group
            176: "Affenpinscher", 177: "Basenji", 178: "Pug", 
            182: "Pomeranian", 184: "Keeshond", 185: "Brussels Griffon",
            250: "Japanese Chin", 251: "Japanese Spaniel", 259: "Pekingese",
            
            # Non-sporting group
            183: "Chow Chow", 199: "French Bulldog", 200: "French Bulldog",
            201: "Bull Terrier", 202: "Staffordshire Bull Terrier",
            
            # Sporting group
            205: "Labrador Retriever", 206: "Chesapeake Bay Retriever",
            207: "Curly-coated Retriever", 208: "Flat-coated Retriever",
            209: "Golden Retriever", 210: "English Setter", 211: "Irish Setter",
            215: "Clumber Spaniel", 216: "Cocker Spaniel", 217: "Welsh Springer Spaniel",
            218: "English Cocker Spaniel", 219: "Sussex Spaniel", 220: "Irish Water Spaniel",
            
            # Terrier group continued
            229: "Border Terrier", 230: "Kerry Blue Terrier", 231: "Irish Terrier",
            232: "Norfolk Terrier", 233: "Norwich Terrier", 234: "Yorkshire Terrier",
            235: "Wire Fox Terrier", 236: "Lakeland Terrier", 237: "Sealyham Terrier",
            238: "Airedale Terrier", 239: "Cairn Terrier", 240: "Australian Terrier",
            241: "Dandie Dinmont Terrier", 242: "Boston Terrier", 243: "Miniature Schnauzer",
            244: "Giant Schnauzer", 245: "Standard Schnauzer", 246: "Scottish Terrier",
            
            # Hound group
            221: "Kuvasz", 222: "Otterhound", 223: "Saluki", 224: "Scottish Deerhound",
            225: "Weimaraner", 226: "Staffordshire Bull Terrier", 
            227: "American Staffordshire Terrier", 228: "Bedlington Terrier",
            
            # Other
            247: "Tibetan Mastiff", 248: "Tibetan Spaniel", 258: "Otterhound",
            263: "Saint Bernard", 264: "Samoyed", 265: "Schipperke", 
            266: "Shiba Inu", 267: "Shih-Tzu", 268: "Tibetan Terrier",
            
            # Additional common breeds
            195: "Dalmatian", 196: "Akita", 197: "Great Pyrenees", 
            198: "Samoyed", 203: "Staffordshire Bull Terrier", 204: "American Staffordshire Terrier",
            212: "Gordon Setter", 213: "Brittany", 214: "English Cocker Spaniel",
            226: "American Staffordshire Terrier", 254: "Mastiff", 255: "Mexican Hairless",
            256: "Newfoundland", 257: "Old English Sheepdog", 260: "Pembroke",
            261: "Pomeranian", 262: "Pug"
        }
        return imagenet_names.get(idx, f"Class {idx}")
    
    def predict_from_array(self, img_array, top_k=5):
        """
        Predict dog breed from numpy array
        """
        if not self.is_loaded:
            self.load_model()
        
        # Preprocess image
        img = self.preprocess_image_from_array(img_array)
        if img is None:
            return None
        
        # Make prediction using ImageNet model
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img * 255.0)
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Filter for dog breeds only (class indices 151-268)
        dog_breed_indices = list(range(151, 269))
        
        # Get top predictions and filter for dog breeds
        top_indices = np.argsort(predictions)[-100:][::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            # Check if it's a dog breed
            if idx in dog_breed_indices:
                breed_name = self.breed_labels.get(str(idx), self._get_imagenet_class_name(idx))
                results.append({
                    'breed': breed_name,
                    'confidence': float(predictions[idx]),
                    'index': int(idx)
                })
            # Also include some related breeds from similar categories
            elif idx in list(range(150, 280)) and len(results) < 3:
                breed_name = self._get_imagenet_class_name(idx)
                results.append({
                    'breed': breed_name,
                    'confidence': float(predictions[idx]),
                    'index': int(idx)
                })
        
        # If no dog breeds found, show top predictions
        if len(results) == 0:
            for idx in top_indices[:top_k]:
                breed_name = self._get_imagenet_class_name(idx)
                results.append({
                    'breed': breed_name,
                    'confidence': float(predictions[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def get_breed_info(self, breed_name):
        """
        Get additional information about a breed
        """
        # Common breed information
        breed_info = {
            "German Shepherd": {
                "origin": "Germany",
                "temperament": "Intelligent, Loyal, Confident",
                "size": "Large",
                "life_span": "10-14 years"
            },
            "Labrador Retriever": {
                "origin": "Canada",
                "temperament": "Friendly, Active, Outgoing",
                "size": "Large",
                "life_span": "10-12 years"
            },
            "Golden Retriever": {
                "origin": "Scotland",
                "temperament": "Intelligent, Friendly, Devoted",
                "size": "Large",
                "life_span": "10-12 years"
            },
            "Bulldog": {
                "origin": "England",
                "temperament": "Docile, Willful, Friendly",
                "size": "Medium",
                "life_span": "8-10 years"
            },
            "Poodle": {
                "origin": "Germany/France",
                "temperament": "Intelligent, Active, Alert",
                "size": "Small to Large",
                "life_span": "12-15 years"
            }
        }
        
        return breed_info.get(breed_name, {
            "origin": "Unknown",
            "temperament": "Varies",
            "size": "Varies",
            "life_span": "10-15 years"
        })


def predict_breed(image_path, model_path='models/dog_breed_model.keras'):
    """
    Convenience function for predicting dog breed
    """
    classifier = DogBreedClassifier(model_path=model_path)
    classifier.load_model()
    return classifier.predict(image_path)


if __name__ == "__main__":
    # Test the classifier
    print("="*60)
    print("Dog Breed Classifier - Test")
    print("="*60)
    
    # Initialize classifier
    classifier = DogBreedClassifier()
    
    # Try to load model
    if classifier.load_model():
        print("\nModel loaded successfully!")
        print(f"Number of breeds: {len(classifier.breed_labels)}")
        
        # Show sample predictions (no actual image for testing)
        print("\nNote: To test with an image, provide an image path:")
        print("  results = classifier.predict('path/to/dog_image.jpg')")
    else:
        print("\nFailed to load model.")
