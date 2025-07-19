"""
Simplified text preprocessor for IMDB sentiment classification.
Avoids complex NLTK dependencies that may have version conflicts.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List
import logging

from src.utils import setup_logging

logger = setup_logging()


class SimpleTextPreprocessor:
    """Simplified text preprocessing pipeline that avoids NLTK complexities."""
    
    def __init__(self, 
                 remove_html: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = True,
                 lowercase: bool = True,
                 min_word_length: int = 2,
                 max_word_length: int = 50):
        """Initialize simple text preprocessor.
        
        Args:
            remove_html: Whether to remove HTML tags
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stop words
            lowercase: Whether to convert to lowercase
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Basic English stop words (no NLTK dependency)
        if self.remove_stopwords:
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'through', 'during', 'before', 'after', 'above', 'below', 
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
                'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
                'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
            }
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, ' ', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub(' ', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        # Replace multiple spaces/tabs/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing whitespace
        return text.strip()
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization using regex (no NLTK dependency)."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to a single text."""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove HTML tags
        if self.remove_html:
            text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation (optional)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Simple tokenization
        tokens = self.simple_tokenize(text)
        
        # Filter tokens by length
        tokens = [
            token for token in tokens 
            if self.min_word_length <= len(token) <= self.max_word_length
        ]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'review') -> pd.DataFrame:
        """Preprocess text in a DataFrame."""
        df_copy = df.copy()
        
        logger.info(f"Preprocessing {len(df_copy)} texts...")
        
        # Apply preprocessing to text column
        df_copy[f'{text_column}_processed'] = df_copy[text_column].apply(
            self.preprocess_text
        )
        
        # Remove empty texts after preprocessing
        initial_count = len(df_copy)
        df_copy = df_copy[df_copy[f'{text_column}_processed'].str.len() > 0]
        final_count = len(df_copy)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} empty texts after preprocessing")
        
        logger.info(f"Preprocessing completed. {final_count} texts remaining.")
        
        return df_copy
    
    def get_vocabulary_stats(self, texts: List[str]) -> dict:
        """Get vocabulary statistics from preprocessed texts."""
        all_words = []
        text_lengths = []
        
        for text in texts:
            words = text.split()
            all_words.extend(words)
            text_lengths.append(len(words))
        
        unique_words = set(all_words)
        
        stats = {
            'total_texts': len(texts),
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'avg_text_length': np.mean(text_lengths),
            'median_text_length': np.median(text_lengths),
            'max_text_length': np.max(text_lengths),
            'min_text_length': np.min(text_lengths)
        }
        
        return stats


def main():
    """Main function to demonstrate simple text preprocessing."""
    # Sample IMDB review
    sample_text = """
    This movie was absolutely fantastic! I loved every minute of it. 
    The acting was <b>superb</b> and the plot was very engaging. 
    Would definitely recommend to anyone who enjoys good cinema.
    Check out more reviews at http://example.com/reviews
    """
    
    # Initialize preprocessor
    preprocessor = SimpleTextPreprocessor()
    
    # Preprocess text
    processed_text = preprocessor.preprocess_text(sample_text)
    
    print("Original text:")
    print(sample_text)
    print("\nProcessed text:")
    print(processed_text)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'review': [sample_text, "This was a terrible movie. I hated it!"],
        'sentiment': [1, 0]
    })
    
    # Preprocess DataFrame
    processed_df = preprocessor.preprocess_dataframe(df)
    print("\nProcessed DataFrame:")
    print(processed_df[['review', 'review_processed', 'sentiment']])


if __name__ == "__main__":
    main() 