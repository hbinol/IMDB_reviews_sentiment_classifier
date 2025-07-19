"""
Data loading functionality for IMDB sentiment classification.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import tarfile
import urllib.request
import logging

from src.utils import ensure_dir, setup_logging

logger = setup_logging()


class IMDBDataLoader:
    """Data loader for IMDB movie reviews dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize data loader.
        
        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.dataset_path = self.data_dir / "aclImdb_v1.tar.gz"
        self.extracted_path = self.data_dir / "aclImdb"
        
        # Kaggle dataset path
        self.kaggle_csv_path = self.data_dir / "IMDB Dataset.csv"
        
        ensure_dir(str(self.data_dir))
    
    def check_kaggle_dataset(self) -> bool:
        """Check if Kaggle CSV dataset exists.
        
        Returns:
            True if Kaggle dataset exists
        """
        # Check for common Kaggle dataset filenames
        possible_names = [
            "IMDB Dataset.csv",
            "imdb_dataset.csv", 
            "IMDB_Dataset.csv",
            "imdb-dataset.csv",
            "movie_reviews.csv"
        ]
        
        for filename in possible_names:
            csv_path = self.data_dir / filename
            if csv_path.exists():
                self.kaggle_csv_path = csv_path
                logger.info(f"Found Kaggle dataset: {csv_path}")
                return True
        
        return False
    
    def load_kaggle_dataset(self) -> pd.DataFrame:
        """Load IMDB dataset from Kaggle CSV format.
        
        Returns:
            DataFrame with 'review' and 'sentiment' columns
        """
        logger.info(f"Loading Kaggle dataset from {self.kaggle_csv_path}")
        
        # Read CSV file
        df = pd.read_csv(self.kaggle_csv_path)
        
        # Check expected columns
        if 'review' not in df.columns:
            # Try common alternative column names
            if 'text' in df.columns:
                df = df.rename(columns={'text': 'review'})
            elif 'comment' in df.columns:
                df = df.rename(columns={'comment': 'review'})
            else:
                raise ValueError("Could not find review text column. Expected 'review', 'text', or 'comment'")
        
        if 'sentiment' not in df.columns:
            # Try common alternative column names
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'sentiment'})
            elif 'rating' in df.columns:
                df = df.rename(columns={'rating': 'sentiment'})
            else:
                raise ValueError("Could not find sentiment column. Expected 'sentiment', 'label', or 'rating'")
        
        # Convert sentiment to binary (0/1) if it's text
        if df['sentiment'].dtype == 'object':
            sentiment_mapping = {
                'positive': 1, 'pos': 1, 'good': 1, '1': 1, 1: 1,
                'negative': 0, 'neg': 0, 'bad': 0, '0': 0, 0: 0
            }
            
            # Convert to lowercase for mapping
            df['sentiment'] = df['sentiment'].astype(str).str.lower()
            df['sentiment'] = df['sentiment'].map(sentiment_mapping)
            
            # Check if mapping was successful
            if df['sentiment'].isna().any():
                unique_sentiments = df['sentiment'].unique()
                raise ValueError(f"Could not map sentiment values: {unique_sentiments}")
        
        # Ensure sentiment is integer
        df['sentiment'] = df['sentiment'].astype(int)
        
        # Remove any rows with missing values
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} samples from Kaggle dataset")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df

    def download_dataset(self) -> None:
        """Download IMDB dataset if not already present."""
        if self.dataset_path.exists():
            logger.info("Dataset already downloaded")
            return
            
        logger.info("Downloading IMDB dataset...")
        try:
            urllib.request.urlretrieve(self.dataset_url, self.dataset_path)
            logger.info(f"Dataset downloaded to {self.dataset_path}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def extract_dataset(self) -> None:
        """Extract downloaded dataset."""
        if self.extracted_path.exists():
            logger.info("Dataset already extracted")
            return
            
        logger.info("Extracting dataset...")
        try:
            with tarfile.open(self.dataset_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            logger.info(f"Dataset extracted to {self.extracted_path}")
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            raise
    
    def load_data_from_files(self, subset: str = "train") -> pd.DataFrame:
        """Load data from extracted files (original Stanford format).
        
        Args:
            subset: 'train' or 'test'
            
        Returns:
            DataFrame with 'review' and 'sentiment' columns
        """
        subset_path = self.extracted_path / subset
        
        reviews = []
        labels = []
        
        # Load positive reviews
        pos_path = subset_path / "pos"
        if pos_path.exists():
            for file_path in pos_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1)  # Positive sentiment
        
        # Load negative reviews
        neg_path = subset_path / "neg"
        if neg_path.exists():
            for file_path in neg_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(0)  # Negative sentiment
        
        df = pd.DataFrame({
            'review': reviews,
            'sentiment': labels
        })
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} {subset} samples")
        return df
    
    def create_dataset(self, 
                      dataset_size: Optional[int] = None,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42,
                      use_kaggle: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            dataset_size: Number of samples to use (None for full dataset)
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed
            use_kaggle: Whether to use Kaggle dataset (auto-detect if None)
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        # Auto-detect data source if not specified
        if use_kaggle is None:
            use_kaggle = self.check_kaggle_dataset()
        
        if use_kaggle:
            logger.info("Using Kaggle CSV dataset")
            # Load from Kaggle CSV
            df = self.load_kaggle_dataset()
        else:
            logger.info("Using original Stanford dataset")
            # Ensure dataset is downloaded and extracted
            self.download_dataset()
            self.extract_dataset()
            
            # Load training data from original format
            df = self.load_data_from_files("train")
        
        # Limit dataset size if specified
        if dataset_size and dataset_size < len(df):
            df = df.sample(n=dataset_size, random_state=random_state)
            logger.info(f"Limited dataset to {dataset_size} samples")
        
        # Split into train/val/test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['sentiment']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=train_val_df['sentiment']
        )
        
        datasets = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # Log dataset statistics
        for split, df in datasets.items():
            pos_count = (df['sentiment'] == 1).sum()
            neg_count = (df['sentiment'] == 0).sum()
            logger.info(f"{split} set: {len(df)} samples ({pos_count} positive, {neg_count} negative)")
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/processed") -> None:
        """Save datasets to CSV files.
        
        Args:
            datasets: Dictionary of DataFrames to save
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        ensure_dir(str(output_path))
        
        for split, df in datasets.items():
            file_path = output_path / f"{split}_data.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {split} data to {file_path}")
    
    def load_processed_datasets(self, data_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
        """Load previously processed datasets.
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        data_path = Path(data_dir)
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            file_path = data_path / f"{split}_data.csv"
            if file_path.exists():
                datasets[split] = pd.read_csv(file_path)
                logger.info(f"Loaded {split} data: {len(datasets[split])} samples")
            else:
                logger.warning(f"No {split} data found at {file_path}")
        
        return datasets


def main():
    """Main function to demonstrate data loading."""
    loader = IMDBDataLoader()
    
    # Check what data sources are available
    has_kaggle = loader.check_kaggle_dataset()
    print(f"Kaggle dataset available: {has_kaggle}")
    
    # Create datasets (will auto-detect source)
    datasets = loader.create_dataset(dataset_size=1000)  # Small sample for testing
    
    # Save datasets
    loader.save_datasets(datasets)
    
    # Load back datasets
    loaded_datasets = loader.load_processed_datasets()
    
    print("Dataset loading completed successfully!")


if __name__ == "__main__":
    main() 