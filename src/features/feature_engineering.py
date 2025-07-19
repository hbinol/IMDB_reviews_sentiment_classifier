"""
Feature engineering functionality for IMDB sentiment classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import logging

from src.utils import setup_logging, ensure_dir

logger = setup_logging()


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor for text data using TF-IDF or Count Vectorization."""
    
    def __init__(self,
                 vectorizer_type: str = "tfidf",
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_df: float = 0.95,
                 min_df: int = 2,
                 stop_words: str = "english",
                 lowercase: bool = True,
                 binary: bool = False):
        """Initialize feature extractor.
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf' or 'count')
            max_features: Maximum number of features
            ngram_range: N-gram range for tokenization
            max_df: Maximum document frequency
            min_df: Minimum document frequency
            stop_words: Stop words to remove
            lowercase: Whether to convert to lowercase
            binary: Whether to use binary features
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.binary = binary
        
        # Initialize vectorizer
        self._init_vectorizer()
        
        # For storing fitted state
        self.is_fitted = False
        self.feature_names = None
        self.vocabulary_size = None
    
    def _init_vectorizer(self):
        """Initialize the appropriate vectorizer."""
        vectorizer_params = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_df': self.max_df,
            'min_df': self.min_df,
            'stop_words': self.stop_words,
            'lowercase': self.lowercase,
            'binary': self.binary
        }
        
        if self.vectorizer_type.lower() == "tfidf":
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        elif self.vectorizer_type.lower() == "count":
            self.vectorizer = CountVectorizer(**vectorizer_params)
        else:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")
    
    def fit(self, X, y=None):
        """Fit the feature extractor.
        
        Args:
            X: Text data to fit on
            y: Target labels (not used)
            
        Returns:
            self
        """
        logger.info(f"Fitting {self.vectorizer_type} vectorizer on {len(X)} texts...")
        
        # Fit vectorizer
        self.vectorizer.fit(X)
        
        # Store fitted state
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        
        logger.info(f"Fitted vectorizer with {self.vocabulary_size} features")
        
        return self
    
    def transform(self, X):
        """Transform text data to features.
        
        Args:
            X: Text data to transform
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        logger.info(f"Transforming {len(X)} texts to features...")
        
        # Transform text to features
        features = self.vectorizer.transform(X)
        
        logger.info(f"Transformed to feature matrix of shape {features.shape}")
        
        return features
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step.
        
        Args:
            X: Text data
            y: Target labels (not used)
            
        Returns:
            Feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """Get feature names.
        
        Returns:
            Array of feature names
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        return self.feature_names
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get vocabulary information.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        return {
            'vocabulary_size': self.vocabulary_size,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_df': self.max_df,
            'min_df': self.min_df
        }
    
    def save_vectorizer(self, filepath: str):
        """Save fitted vectorizer to file.
        
        Args:
            filepath: Path to save vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        ensure_dir(filepath.rsplit('/', 1)[0])
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Saved vectorizer to {filepath}")
    
    def load_vectorizer(self, filepath: str):
        """Load fitted vectorizer from file.
        
        Args:
            filepath: Path to load vectorizer from
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Update fitted state
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        
        logger.info(f"Loaded vectorizer from {filepath}")


class FeaturePipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_extractor = None
        self.label_encoder = None
        self.is_fitted = False
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize pipeline components."""
        features_config = self.config.get('features', {})
        
        # Initialize feature extractor
        self.feature_extractor = TextFeatureExtractor(
            vectorizer_type="tfidf" if features_config.get('use_tfidf', True) else "count",
            max_features=features_config.get('max_features', 10000),
            ngram_range=tuple(features_config.get('ngram_range', [1, 2])),
            **features_config.get('tfidf_params', {})
        )
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
    
    def fit(self, texts, labels):
        """Fit the complete pipeline.
        
        Args:
            texts: Text data
            labels: Target labels
            
        Returns:
            self
        """
        logger.info("Fitting feature pipeline...")
        
        # Fit feature extractor
        self.feature_extractor.fit(texts)
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        
        self.is_fitted = True
        
        logger.info("Feature pipeline fitted successfully")
        
        return self
    
    def transform(self, texts, labels=None):
        """Transform data using fitted pipeline.
        
        Args:
            texts: Text data to transform
            labels: Optional target labels to encode
            
        Returns:
            Tuple of (features, encoded_labels) or just features if no labels
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Transform texts to features
        features = self.feature_extractor.transform(texts)
        
        if labels is not None:
            # Encode labels
            encoded_labels = self.label_encoder.transform(labels)
            return features, encoded_labels
        
        return features
    
    def fit_transform(self, texts, labels):
        """Fit and transform in one step.
        
        Args:
            texts: Text data
            labels: Target labels
            
        Returns:
            Tuple of (features, encoded_labels)
        """
        return self.fit(texts, labels).transform(texts, labels)
    
    def inverse_transform_labels(self, encoded_labels):
        """Convert encoded labels back to original labels.
        
        Args:
            encoded_labels: Encoded label values
            
        Returns:
            Original label values
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self):
        """Get feature names from extractor.
        
        Returns:
            Array of feature names
        """
        return self.feature_extractor.get_feature_names()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information.
        
        Returns:
            Dictionary with pipeline statistics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        info = {
            'feature_extractor': self.feature_extractor.get_vocabulary_info(),
            'label_classes': self.label_encoder.classes_.tolist(),
            'n_classes': len(self.label_encoder.classes_)
        }
        
        return info
    
    def save_pipeline(self, directory: str):
        """Save fitted pipeline components.
        
        Args:
            directory: Directory to save pipeline components
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        ensure_dir(directory)
        
        # Save feature extractor
        self.feature_extractor.save_vectorizer(f"{directory}/vectorizer.pkl")
        
        # Save label encoder
        with open(f"{directory}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Saved pipeline to {directory}")
    
    def load_pipeline(self, directory: str):
        """Load fitted pipeline components.
        
        Args:
            directory: Directory containing pipeline components
        """
        # Load feature extractor
        self.feature_extractor.load_vectorizer(f"{directory}/vectorizer.pkl")
        
        # Load label encoder
        with open(f"{directory}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_fitted = True
        
        logger.info(f"Loaded pipeline from {directory}")


def analyze_feature_importance(feature_extractor: TextFeatureExtractor, 
                              model, 
                              top_n: int = 20) -> Dict[str, list]:
    """Analyze feature importance for linear models.
    
    Args:
        feature_extractor: Fitted feature extractor
        model: Fitted linear model with coef_ attribute
        top_n: Number of top features to return
        
    Returns:
        Dictionary with top positive and negative features
    """
    if not hasattr(model, 'coef_'):
        logger.warning("Model doesn't have coef_ attribute for feature importance analysis")
        return {'positive': [], 'negative': []}
    
    feature_names = feature_extractor.get_feature_names()
    coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    
    # Get indices of top positive and negative coefficients
    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefficients)[:top_n]
    
    # Get feature names and coefficients
    top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]
    top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]
    
    return {
        'positive': top_positive,
        'negative': top_negative
    }


def main():
    """Main function to demonstrate feature engineering."""
    # Sample data
    texts = [
        "This movie was fantastic! Great acting and plot.",
        "Terrible movie. Boring and poorly made.",
        "Good story but average execution.",
        "Amazing cinematography and excellent performances!"
    ]
    labels = [1, 0, 1, 1]  # 1=positive, 0=negative
    
    # Create sample config
    config = {
        'features': {
            'max_features': 1000,
            'ngram_range': [1, 2],
            'use_tfidf': True,
            'tfidf_params': {
                'max_df': 0.95,
                'min_df': 1,
                'stop_words': 'english'
            }
        }
    }
    
    # Initialize and fit pipeline
    pipeline = FeaturePipeline(config)
    features, encoded_labels = pipeline.fit_transform(texts, labels)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Encoded labels: {encoded_labels}")
    print(f"Pipeline info: {pipeline.get_pipeline_info()}")


if __name__ == "__main__":
    main() 