"""
Script to train the fake news detection model using available datasets.
"""

import os
import pandas as pd

from fake_news_detector import FakeNewsDetector

def create_from_separate_datasets():
    """
    Create a combined dataset from separate True and Fake news CSV files.
    """
    dataset_dir = 'Fake_new_prediction'
    
    true_path = os.path.join(dataset_dir, 'True.csv')
    fake_path = os.path.join(dataset_dir, 'Fake.csv')
    
    if os.path.exists(true_path) and os.path.exists(fake_path):
        print("\nFound separate True and Fake news datasets. Combining...")
        
        # Load datasets
        df_true = pd.read_csv(true_path)
        df_fake = pd.read_csv(fake_path)
        
        print(f"True news articles: {len(df_true)}")
        print(f"Fake news articles: {len(df_fake)}")
        
        # Add labels
        df_true['label'] = 'REAL'
        df_fake['label'] = 'FAKE'
        
        # Combine - use title and text
        if 'title' in df_true.columns:
            df_true['text'] = df_true['title'].fillna('') + ' ' + df_true['text'].fillna('')
            df_fake['text'] = df_fake['title'].fillna('') + ' ' + df_fake['text'].fillna('')
        
        df_combined = pd.concat([df_true[['text', 'label']], df_fake[['text', 'label']]])
        
        # Shuffle
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_path = os.path.join(dataset_dir, 'combined_news.csv')
        df_combined.to_csv(combined_path, index=False)
        
        print(f"\nCombined dataset saved to: {combined_path}")
        print(f"Total articles: {len(df_combined)}")
        print(f"REAL: {sum(df_combined['label'] == 'REAL')}")
        print(f"FAKE: {sum(df_combined['label'] == 'FAKE')}")
        
        return combined_path
    
    return None


def create_from_archive():
    """
    Create a combined dataset from the ISOT archive data.
    Reads .txt files from Fake_new_prediction/Fake_new_prediction/archive/
    """
    import glob
    
    # First check for the new news.csv file
    news_csv_path = 'Fake_new_prediction/archive (1)/news.csv'
    
    if os.path.exists(news_csv_path):
        print("\n--- Found news.csv dataset ---")
        df = pd.read_csv(news_csv_path)
        
        # Use title + text for better features
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Keep only text and label columns
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']]
            
            # Clean labels
            df['label'] = df['label'].str.strip().str.upper()
            
            print(f"Total articles: {len(df)}")
            print(f"REAL: {sum(df['label'] == 'REAL')}")
            print(f"FAKE: {sum(df['label'] == 'FAKE')}")
            
            # Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save processed dataset
            dataset_dir = 'Fake_new_prediction'
            processed_path = os.path.join(dataset_dir, 'processed_news_dataset.csv')
            df.to_csv(processed_path, index=False)
            
            print(f"\nDataset saved to: {processed_path}")
            return processed_path
    
    # Fallback to old archive processing
    base_dir = 'Fake_new_prediction/Fake_new_prediction/archive'
    
    # Check if archive exists
    if not os.path.exists(base_dir):
        return None
    
    all_articles = []
    all_labels = []
    
    # Process overall dataset (overall/overall/real and overall/overall/fake)
    real_dir = os.path.join(base_dir, 'overall/overall/real')
    fake_dir = os.path.join(base_dir, 'overall/overall/fake')
    
    print("\n--- Processing overall dataset ---")
    
    if os.path.exists(real_dir):
        for filepath in glob.glob(os.path.join(real_dir, '*.txt')):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text and len(text) > 50:  # Minimum text length
                        all_articles.append(text)
                        all_labels.append('REAL')
            except Exception as e:
                pass
        print(f"Real news from overall: {sum(1 for l in all_labels if l == 'REAL')}")
    
    if os.path.exists(fake_dir):
        fake_count = 0
        for filepath in glob.glob(os.path.join(fake_dir, '*.txt')):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text and len(text) > 50:
                        all_articles.append(text)
                        all_labels.append('FAKE')
                        fake_count += 1
            except Exception as e:
                pass
        print(f"Fake news from overall: {fake_count}")
    
    # Process training dataset (training/training/fakeNewsDataset/fake and legit)
    fake_train_dir = os.path.join(base_dir, 'training/training/fakeNewsDataset/fake')
    legit_train_dir = os.path.join(base_dir, 'training/training/fakeNewsDataset/legit')
    
    print("\n--- Processing training dataset ---")
    
    if os.path.exists(fake_train_dir):
        fake_count = 0
        for filepath in glob.glob(os.path.join(fake_train_dir, '*.txt')):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text and len(text) > 50:
                        all_articles.append(text)
                        all_labels.append('FAKE')
                        fake_count += 1
            except Exception as e:
                pass
        print(f"Fake news from training: {fake_count}")
    
    if os.path.exists(legit_train_dir):
        legit_count = 0
        for filepath in glob.glob(os.path.join(legit_train_dir, '*.txt')):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text and len(text) > 50:
                        all_articles.append(text)
                        all_labels.append('REAL')
                        legit_count += 1
            except Exception as e:
                pass
        print(f"Legit news from training: {legit_count}")
    
    # Process celebrity dataset
    celeb_dir = os.path.join(base_dir, 'training/training/celebrityDataset')
    
    print("\n--- Processing celebrity dataset ---")
    
    if os.path.exists(celeb_dir):
        celeb_legit_dir = os.path.join(celeb_dir, 'legit')
        if os.path.exists(celeb_legit_dir):
            legit_count = 0
            for filepath in glob.glob(os.path.join(celeb_legit_dir, '*.txt')):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if text and len(text) > 50:
                            all_articles.append(text)
                            all_labels.append('REAL')
                            legit_count += 1
                except Exception as e:
                    pass
            print(f"Legit news from celebrity: {legit_count}")
    
    print(f"\nTotal articles: {len(all_articles)}")
    print(f"REAL: {sum(1 for l in all_labels if l == 'REAL')}")
    print(f"FAKE: {sum(1 for l in all_labels if l == 'FAKE')}")
    
    if len(all_articles) < 100:
        return None
    
    # Create DataFrame
    df_combined = pd.DataFrame({'text': all_articles, 'label': all_labels})
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    dataset_dir = 'Fake_new_prediction'
    combined_path = os.path.join(dataset_dir, 'isot_archive_dataset.csv')
    df_combined.to_csv(combined_path, index=False)
    
    print(f"\nArchive dataset saved to: {combined_path}")
    
    return combined_path


def create_from_train_csv():
    """
    Check for and prepare the train.csv file.
    """
    dataset_dir = 'Fake_new_prediction'
    train_path = os.path.join(dataset_dir, 'train.csv')
    
    if os.path.exists(train_path):
        print(f"\nFound train.csv at {train_path}")
        df = pd.read_csv(train_path)
        print(f"Total articles: {len(df)}")
        
        # Check columns
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare text column
        if 'text' not in df.columns:
            if 'title' in df.columns:
                df['text'] = df['title'].fillna('')
        
        if 'label' not in df.columns:
            if 'Label' in df.columns:
                df.rename(columns={'Label': 'label'}, inplace=True)
        
        # Convert labels to FAKE/REAL
        if 'label' in df.columns:
            if df['label'].dtype in ['int64', 'float64']:
                # 1 = FAKE, 0 = REAL
                df['label'] = df['label'].map({1: 'FAKE', 0: 'REAL'})
            elif df['label'].dtype == 'object':
                # Try to map string labels
                unique_labels = df['label'].unique()
                label_map = {}
                for label in unique_labels:
                    label_str = str(label).lower()
                    if 'real' in label_str or label_str == '0':
                        label_map[label] = 'REAL'
                    elif 'fake' in label_str or label_str == '1':
                        label_map[label] = 'FAKE'
                df['label'] = df['label'].map(label_map)
        
        # Save processed dataset
        processed_path = os.path.join(dataset_dir, 'processed_train.csv')
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to: {processed_path}")
        
        return processed_path
    
    return None


def main():
    """Main function to train with available dataset."""
    
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*60)
    
    dataset_dir = 'Fake_new_prediction'
    
    # Print dataset instructions
    print_download_instructions()
    
    # Check for existing dataset files
    dataset_path = None
    
    # Try different dataset formats
    print("\n" + "="*60)
    print("LOOKING FOR DATASET FILES")
    print("="*60)
    
    # First try the archive data (ISOT dataset)
    dataset_path = create_from_archive()
    
    if not dataset_path:
        dataset_path = create_from_train_csv()
    
    if not dataset_path:
        dataset_path = create_from_separate_datasets()
    
    if not dataset_path:
        # Use sample data
        print("\nNo dataset found. Using sample data for demonstration...")
        from fake_news_detector import create_sample_dataset
        df = create_sample_dataset()
        dataset_path = os.path.join(dataset_dir, 'sample_dataset.csv')
        df.to_csv(dataset_path, index=False)
        print(f"Sample dataset saved to: {dataset_path}")
    else:
        # Train with the available dataset
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Initialize detector with optimized parameters for larger dataset
        detector = FakeNewsDetector(
            max_features=8000,  # Increased for larger dataset
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        # Load the dataset
        df = detector.load_data(dataset_path)
        
        # Train the model with cross-validation
        results = detector.train(df, text_column='text', label_column='label', test_size=0.2)
        
        # Save the model
        model_path = os.path.join(dataset_dir, 'fake_news_kaggle_model.pkl')
        detector.save_model(model_path)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"Training samples: {results['train_size']}")
        print(f"Test samples: {results['test_size']}")
        print(f"Training accuracy: {results['training_accuracy']*100:.2f}%")
        print(f"Test accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
