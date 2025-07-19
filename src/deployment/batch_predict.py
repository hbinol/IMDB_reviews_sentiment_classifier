"""
Batch prediction service for IMDB sentiment classification.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging
from datetime import datetime
import json

from src.utils import setup_logging, ensure_dir, save_predictions, setup_environment
from src.data.simple_preprocessor import SimpleTextPreprocessor
from src.features.feature_engineering import FeaturePipeline

logger = setup_logging()


class BatchPredictor:
    """Batch prediction service for IMDB sentiment classification."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 pipeline_path: Optional[str] = None,
                 mlflow_model_uri: Optional[str] = None):
        """Initialize batch predictor.
        
        Args:
            model_path: Path to saved model file
            pipeline_path: Path to saved feature pipeline
            mlflow_model_uri: MLflow model URI (alternative to file paths)
        """
        self.model = None
        self.feature_pipeline = None
        self.preprocessor = None
        self.model_metadata = {}
        
        # Load model and pipeline
        if mlflow_model_uri:
            self._load_from_mlflow(mlflow_model_uri)
        else:
            self._load_from_files(model_path, pipeline_path)
        
        # Initialize preprocessor with same settings as training
        self.preprocessor = SimpleTextPreprocessor(
            remove_html=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lowercase=True
        )
    
    def _load_from_files(self, model_path: str, pipeline_path: str):
        """Load model and pipeline from file paths.
        
        Args:
            model_path: Path to model file
            pipeline_path: Path to feature pipeline directory
        """
        if not model_path or not pipeline_path:
            raise ValueError("Both model_path and pipeline_path must be provided")
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info(f"Loading feature pipeline from {pipeline_path}")
        config = setup_environment()  # Get default config
        self.feature_pipeline = FeaturePipeline(config)
        self.feature_pipeline.load_pipeline(pipeline_path)
        
        self.model_metadata = {
            'model_path': model_path,
            'pipeline_path': pipeline_path,
            'loaded_at': datetime.now().isoformat(),
            'source': 'file'
        }
    
    def _load_from_mlflow(self, model_uri: str):
        """Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI
        """
        logger.info(f"Loading model from MLflow: {model_uri}")
        
        # Load model
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # For MLflow models, we need to load the feature pipeline separately
        # This would typically be stored as artifacts in MLflow
        logger.warning("Feature pipeline loading from MLflow not fully implemented")
        logger.warning("Falling back to default pipeline configuration")
        
        config = setup_environment()
        self.feature_pipeline = FeaturePipeline(config)
        
        # Try to load from default location
        try:
            self.feature_pipeline.load_pipeline("models/feature_pipeline")
            logger.info("Loaded feature pipeline from default location")
        except:
            logger.error("Could not load feature pipeline. Predictions may fail.")
        
        self.model_metadata = {
            'model_uri': model_uri,
            'loaded_at': datetime.now().isoformat(),
            'source': 'mlflow'
        }
    
    def preprocess_input(self, data: Union[str, List[str], pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data for prediction.
        
        Args:
            data: Input data (string, list of strings, or DataFrame)
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert input to DataFrame
        if isinstance(data, str):
            df = pd.DataFrame({'review': [data]})
        elif isinstance(data, list):
            df = pd.DataFrame({'review': data})
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'review' not in df.columns:
                raise ValueError("DataFrame must contain 'review' column")
        else:
            raise ValueError("Input must be string, list of strings, or DataFrame")
        
        # Preprocess text
        processed_df = self.preprocessor.preprocess_dataframe(df, text_column='review')
        
        return processed_df
    
    def predict(self, 
                data: Union[str, List[str], pd.DataFrame],
                return_probabilities: bool = True) -> Dict[str, Any]:
        """Make predictions on input data.
        
        Args:
            data: Input data to predict on
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None or self.feature_pipeline is None:
            raise ValueError("Model and feature pipeline must be loaded first")
        
        logger.info(f"Making predictions on {len(data) if hasattr(data, '__len__') else 1} samples")
        
        # Preprocess input
        processed_df = self.preprocess_input(data)
        
        # Extract features
        features = self.feature_pipeline.transform(processed_df['review_processed'].values)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        # Convert to sentiment labels
        sentiment_labels = ['negative' if pred == 0 else 'positive' for pred in predictions]
        
        results = {
            'predictions': predictions.tolist(),
            'sentiment_labels': sentiment_labels,
            'n_samples': len(predictions)
        }
        
        # Add probabilities if requested and available
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            results['probabilities'] = probabilities.tolist()
            results['confidence_scores'] = np.max(probabilities, axis=1).tolist()
        
        # Add metadata
        results['metadata'] = {
            'model_info': self.model_metadata,
            'prediction_time': datetime.now().isoformat(),
            'feature_shape': features.shape
        }
        
        logger.info(f"Predictions completed. Positive: {sum(predictions)}, "
                   f"Negative: {len(predictions) - sum(predictions)}")
        
        return results
    
    def predict_batch_file(self,
                          input_file: str,
                          output_file: str,
                          text_column: str = 'review',
                          batch_size: int = 1000,
                          output_format: str = 'csv') -> str:
        """Process a file in batches and save predictions.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output file
            text_column: Name of text column in input file
            batch_size: Number of samples to process at once
            output_format: Output format ('csv', 'json', 'pickle')
            
        Returns:
            Path to output file
        """
        logger.info(f"Processing batch file: {input_file}")
        
        # Read input file
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Input file must be CSV or JSON")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input file")
        
        # Prepare output DataFrame
        results_df = df.copy()
        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []
        
        # Process in batches
        n_samples = len(df)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_df = df.iloc[i:batch_end]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{n_batches}")
            
            # Make predictions on batch
            batch_results = self.predict(
                batch_df[text_column].tolist(),
                return_probabilities=True
            )
            
            all_predictions.extend(batch_results['predictions'])
            
            if 'probabilities' in batch_results:
                all_probabilities.extend(batch_results['probabilities'])
                all_confidence_scores.extend(batch_results['confidence_scores'])
        
        # Add predictions to DataFrame
        results_df['prediction'] = all_predictions
        results_df['sentiment'] = ['negative' if pred == 0 else 'positive' 
                                  for pred in all_predictions]
        
        if all_probabilities:
            results_df['confidence'] = all_confidence_scores
            results_df['prob_negative'] = [prob[0] for prob in all_probabilities]
            results_df['prob_positive'] = [prob[1] for prob in all_probabilities]
        
        # Add metadata
        results_df['processed_at'] = datetime.now().isoformat()
        
        # Save results
        ensure_dir(str(Path(output_file).parent))
        save_predictions(results_df, output_file, output_format)
        
        logger.info(f"Batch processing completed. Results saved to {output_file}")
        
        return output_file


def create_sample_data(output_file: str = "data/external/sample_reviews.csv"):
    """Create sample data for testing batch prediction.
    
    Args:
        output_file: Path to save sample data
    """
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I really enjoyed this film. Great cinematography and excellent character development.",
        "What a terrible movie. The plot was boring and the acting was awful.",
        "Worst film I've ever seen. Complete waste of time and money.",
        "It was an okay movie. Not great but not terrible either. Average entertainment.",
        "Amazing movie! One of the best I've seen this year. Highly recommended!",
        "The movie was disappointing. Had high expectations but it fell short.",
        "Brilliant storytelling and outstanding performances. A masterpiece!",
        "Not my cup of tea. The genre doesn't appeal to me but others might enjoy it.",
        "Excellent film with great attention to detail. Worth watching multiple times."
    ]
    
    df = pd.DataFrame({
        'review_id': range(1, len(sample_reviews) + 1),
        'review': sample_reviews,
        'source': 'sample_data'
    })
    
    ensure_dir(str(Path(output_file).parent))
    df.to_csv(output_file, index=False)
    
    logger.info(f"Sample data created: {output_file}")
    return output_file


def main():
    """Main function for command-line batch prediction."""
    parser = argparse.ArgumentParser(description="IMDB Sentiment Batch Prediction")
    
    parser.add_argument("--input-file", type=str, required=True,
                       help="Path to input CSV file with reviews")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Path to output file for predictions")
    parser.add_argument("--model-path", type=str,
                       help="Path to saved model file")
    parser.add_argument("--pipeline-path", type=str,
                       help="Path to saved feature pipeline")
    parser.add_argument("--mlflow-uri", type=str,
                       help="MLflow model URI (alternative to file paths)")
    parser.add_argument("--text-column", type=str, default="review",
                       help="Name of text column in input file")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for processing")
    parser.add_argument("--output-format", type=str, default="csv",
                       choices=['csv', 'json', 'pickle'],
                       help="Output format")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data for testing")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        sample_file = create_sample_data()
        print(f"Sample data created: {sample_file}")
        return
    
    # Initialize batch predictor
    try:
        if args.mlflow_uri:
            predictor = BatchPredictor(mlflow_model_uri=args.mlflow_uri)
        else:
            # Use default paths if not provided
            model_path = args.model_path or "models/best_model_logistic_regression.joblib"
            pipeline_path = args.pipeline_path or "models/feature_pipeline"
            
            predictor = BatchPredictor(
                model_path=model_path,
                pipeline_path=pipeline_path
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Process batch file
    try:
        output_file = predictor.predict_batch_file(
            input_file=args.input_file,
            output_file=args.output_file,
            text_column=args.text_column,
            batch_size=args.batch_size,
            output_format=args.output_format
        )
        
        print(f"Batch prediction completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise


if __name__ == "__main__":
    main() 