"""
Text preprocessing functionality for IMDB sentiment classification.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import logging

from src.utils import setup_logging

logger = setup_logging()


class TextPreprocessor:
    """Text preprocessing pipeline for IMDB reviews."""
    
    def __init__(self, 
                 remove_html: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = True,
                 lowercase: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 min_word_length: int = 2,
                 max_word_length: int = 50):
        """Initialize text preprocessor.
        
        Args:
            remove_html: Whether to remove HTML tags
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stop words
            lowercase: Whether to convert to lowercase
            lemmatize: Whether to lemmatize words
            stem: Whether to stem words
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.stem:
            self.stemmer = PorterStemmer()
            
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {item}")
                nltk.download(item, quiet=True)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed
        """
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, ' ', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub(' ', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized spacing
        """
        # Replace multiple spaces/tabs/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing whitespace
        return text.strip()
    
    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character used by WordNetLemmatizer.
        
        Args:
            word: Input word
            
        Returns:
            WordNet POS tag
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
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
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens by length
        tokens = [
            token for token in tokens 
            if self.min_word_length <= len(token) <= self.max_word_length
        ]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [
                self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) 
                for token in tokens
            ]
        
        # Stem
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'review') -> pd.DataFrame:
        """Preprocess text in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to preprocess
            
        Returns:
            DataFrame with preprocessed text
        """
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
        """Get vocabulary statistics from preprocessed texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Dictionary with vocabulary statistics
        """
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
    """Main function to demonstrate text preprocessing."""
    # Sample IMDB review
    sample_text = """
    This movie was absolutely fantastic! I loved every minute of it. 
    The acting was <b>superb</b> and the plot was very engaging. 
    Would definitely recommend to anyone who enjoys good cinema.
    Check out more reviews at http://example.com/reviews
    """
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
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
    print(processed_df)


if __name__ == "__main__":
    main() 