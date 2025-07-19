#!/usr/bin/env python3
"""
Script to check and validate the Kaggle IMDB dataset.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.data_loader import IMDBDataLoader


def check_kaggle_data():
    """Check if Kaggle dataset is properly placed and formatted."""
    print("IMDB Dataset Checker")
    print("=" * 40)
    
    # Initialize data loader
    loader = IMDBDataLoader()
    
    # Check if Kaggle dataset exists
    has_kaggle = loader.check_kaggle_dataset()
    
    if not has_kaggle:
        print("ERROR: Kaggle dataset not found!")
        print("\nExpected locations in data/raw/:")
        print("  - IMDB Dataset.csv")
        print("  - imdb_dataset.csv") 
        print("  - IMDB_Dataset.csv")
        print("  - imdb-dataset.csv")
        print("  - movie_reviews.csv")
        print(f"\nPlease place your downloaded CSV file in: {loader.data_dir}")
        return False
    
    print(f"SUCCESS: Found Kaggle dataset: {loader.kaggle_csv_path}")
    
    try:
        # Load and validate dataset
        print("\nLoading dataset...")
        df = loader.load_kaggle_dataset()
        
        print(f"SUCCESS: Dataset loaded successfully!")
        print(f"   Total samples: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check data types
        print(f"\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        # Check sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        print(f"\nSentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            label = "Positive" if sentiment == 1 else "Negative"
            percentage = (count / len(df)) * 100
            print(f"   {label} ({sentiment}): {count:,} ({percentage:.1f}%)")
        
        # Show sample data
        print(f"\nSample reviews:")
        for i in range(min(3, len(df))):
            sentiment_label = "Positive" if df.iloc[i]['sentiment'] == 1 else "Negative"
            review_preview = df.iloc[i]['review'][:100] + "..." if len(df.iloc[i]['review']) > 100 else df.iloc[i]['review']
            print(f"   {i+1}. [{sentiment_label}] {review_preview}")
        
        # Test creating splits
        print(f"\nTesting train/val/test splits...")
        datasets = loader.create_dataset(dataset_size=1000, use_kaggle=True)
        
        total_samples = sum(len(split_df) for split_df in datasets.values())
        print(f"SUCCESS: Splits created successfully! Total: {total_samples} samples")
        
        for split_name, split_df in datasets.items():
            pos_count = (split_df['sentiment'] == 1).sum()
            neg_count = (split_df['sentiment'] == 0).sum()
            print(f"   {split_name}: {len(split_df)} samples ({pos_count} pos, {neg_count} neg)")
        
        print(f"\nSUCCESS: Dataset validation completed successfully!")
        print(f"Your Kaggle dataset is ready to use with the ML pipeline!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error validating dataset: {e}")
        print(f"\nCommon issues:")
        print(f"   - Missing required columns ('review', 'sentiment')")
        print(f"   - Incorrect sentiment format (should be 'positive'/'negative' or 0/1)")
        print(f"   - File encoding issues")
        print(f"   - Corrupted CSV file")
        return False


def show_usage_examples():
    """Show usage examples for the dataset."""
    print(f"\nNext Steps:")
    print(f"1. Test the full pipeline:")
    print(f"   python scripts/test_setup.py")
    print(f"\n2. Run quick training:")
    print(f"   make train-quick")
    print(f"\n3. Run full training:")
    print(f"   python src/models/train.py")
    print(f"\n4. Test batch prediction:")
    print(f"   make predict-sample")


if __name__ == "__main__":
    success = check_kaggle_data()
    
    if success:
        show_usage_examples()
    else:
        print(f"\nPlease fix the issues above and run this script again.")
        sys.exit(1) 