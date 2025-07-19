"""
Prefect workflow for orchestrating IMDB sentiment classification training pipeline.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import setup_environment, ensure_dir
from src.data.data_loader import IMDBDataLoader
from src.data.preprocessor import TextPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.models.train import ModelTrainer, run_training_pipeline


@task(name="load_configuration")
def load_config_task() -> Dict[str, Any]:
    """Load project configuration."""
    logger = get_run_logger()
    logger.info("Loading configuration...")
    
    config = setup_environment()
    logger.info(f"Loaded configuration for project: {config['project']['name']}")
    
    return config


@task(name="download_and_prepare_data")
def download_data_task(config: Dict[str, Any], dataset_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Download and prepare IMDB dataset."""
    logger = get_run_logger()
    logger.info("Downloading and preparing IMDB dataset...")
    
    data_config = config.get('data', {})
    
    # Initialize data loader
    loader = IMDBDataLoader()
    
    # Create datasets with train/val/test splits
    datasets = loader.create_dataset(
        dataset_size=dataset_size or data_config.get('dataset_size'),
        test_size=data_config.get('test_size', 0.2),
        val_size=data_config.get('val_size', 0.1),
        random_state=data_config.get('random_state', 42)
    )
    
    # Save raw datasets
    loader.save_datasets(datasets, "data/processed")
    
    logger.info(f"Data preparation completed. Train: {len(datasets['train'])}, "
                f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return datasets


@task(name="preprocess_text")
def preprocess_text_task(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Preprocess text data."""
    logger = get_run_logger()
    logger.info("Preprocessing text data...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        remove_html=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lowercase=True,
        lemmatize=True,
        stem=False
    )
    
    # Preprocess each dataset split
    processed_datasets = {}
    for split, df in datasets.items():
        logger.info(f"Preprocessing {split} data...")
        processed_df = preprocessor.preprocess_dataframe(df, text_column='review')
        processed_datasets[split] = processed_df
        
        # Log vocabulary statistics
        vocab_stats = preprocessor.get_vocabulary_stats(
            processed_df['review_processed'].tolist()
        )
        logger.info(f"{split} vocabulary stats: {vocab_stats}")
    
    logger.info("Text preprocessing completed")
    
    return processed_datasets


@task(name="feature_engineering")
def feature_engineering_task(datasets: Dict[str, pd.DataFrame], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform feature engineering on preprocessed data."""
    logger = get_run_logger()
    logger.info("Performing feature engineering...")
    
    # Initialize feature pipeline
    feature_pipeline = FeaturePipeline(config)
    
    # Fit on training data
    X_train, y_train = feature_pipeline.fit_transform(
        datasets['train']['review_processed'].values,
        datasets['train']['sentiment'].values
    )
    
    # Transform validation and test data
    X_val, y_val = feature_pipeline.transform(
        datasets['val']['review_processed'].values,
        datasets['val']['sentiment'].values
    )
    
    X_test, y_test = feature_pipeline.transform(
        datasets['test']['review_processed'].values,
        datasets['test']['sentiment'].values
    )
    
    # Log pipeline information
    pipeline_info = feature_pipeline.get_pipeline_info()
    logger.info(f"Feature pipeline info: {pipeline_info}")
    
    # Save feature pipeline
    pipeline_path = "models/feature_pipeline"
    ensure_dir(pipeline_path)
    feature_pipeline.save_pipeline(pipeline_path)
    
    logger.info(f"Feature engineering completed. Train shape: {X_train.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_pipeline': feature_pipeline
    }


@task(name="train_models")
def train_models_task(feature_data: Dict[str, Any], 
                     config: Dict[str, Any]) -> Dict[str, Any]:
    """Train all configured models with hyperparameter tuning."""
    logger = get_run_logger()
    logger.info("Training models with hyperparameter tuning...")
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Train all models
    training_results = trainer.train_all_models(
        feature_data['X_train'],
        feature_data['y_train'],
        feature_data['X_val'],
        feature_data['y_val'],
        feature_data['feature_pipeline']
    )
    
    logger.info(f"Model training completed. Best model: {trainer.best_model_name}")
    
    return {
        'training_results': training_results,
        'trainer': trainer
    }


@task(name="evaluate_best_model")
def evaluate_model_task(training_results: Dict[str, Any],
                       feature_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the best model on test set."""
    logger = get_run_logger()
    logger.info("Evaluating best model on test set...")
    
    trainer = training_results['trainer']
    
    if trainer.best_model is None:
        logger.error("No best model available for evaluation")
        return {}
    
    # Evaluate on test set
    test_results = trainer.evaluate_model(
        trainer.best_model,
        feature_data['X_test'],
        feature_data['y_test']
    )
    
    # Log test metrics
    test_metrics = test_results['metrics']
    logger.info(f"Test set evaluation results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return test_results


@task(name="save_model_artifacts")
def save_artifacts_task(training_results: Dict[str, Any],
                       test_results: Dict[str, Any]) -> str:
    """Save model artifacts and metadata."""
    logger = get_run_logger()
    logger.info("Saving model artifacts...")
    
    trainer = training_results['trainer']
    
    if trainer.best_model is None:
        logger.error("No best model to save")
        return ""
    
    # Save best model
    model_path = trainer.save_best_model("models")
    
    # Save test results
    import pickle
    results_path = "models/test_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    
    logger.info(f"Model artifacts saved. Best model: {model_path}")
    
    return model_path


@flow(name="imdb-sentiment-training-pipeline", 
      task_runner=SequentialTaskRunner(),
      description="End-to-end training pipeline for IMDB sentiment classification")
def training_flow(dataset_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Main training pipeline flow for IMDB sentiment classification.
    
    Args:
        dataset_size: Optional limit on dataset size for faster testing
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting IMDB sentiment classification training pipeline")
    
    # Load configuration
    config = load_config_task()
    
    # Download and prepare data
    datasets = download_data_task(config, dataset_size)
    
    # Preprocess text
    processed_datasets = preprocess_text_task(datasets)
    
    # Feature engineering
    feature_data = feature_engineering_task(processed_datasets, config)
    
    # Train models
    training_results = train_models_task(feature_data, config)
    
    # Evaluate best model
    test_results = evaluate_model_task(training_results, feature_data)
    
    # Save artifacts
    model_path = save_artifacts_task(training_results, test_results)
    
    # Compile final results
    pipeline_results = {
        'best_model_name': training_results['trainer'].best_model_name,
        'best_model_path': model_path,
        'training_results': training_results['training_results'],
        'test_results': test_results,
        'status': 'completed'
    }
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model: {pipeline_results['best_model_name']}")
    
    if test_results:
        logger.info(f"Test accuracy: {test_results['metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-score: {test_results['metrics']['f1']:.4f}")
    
    return pipeline_results


@flow(name="quick-training-pipeline",
      description="Quick training pipeline with small dataset for testing")
def quick_training_flow() -> Dict[str, Any]:
    """Quick training flow with small dataset for testing."""
    return training_flow(dataset_size=1000)


@flow(name="full-training-pipeline",
      description="Full training pipeline with complete dataset")
def full_training_flow() -> Dict[str, Any]:
    """Full training flow with complete dataset."""
    return training_flow(dataset_size=None)


# Flow for retraining with new data
@flow(name="retrain-pipeline",
      description="Retrain models with new configuration or data")
def retrain_flow(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrain models with optional configuration overrides.
    
    Args:
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Dictionary with retraining results
    """
    logger = get_run_logger()
    logger.info("Starting model retraining pipeline")
    
    # Load base configuration
    config = load_config_task()
    
    # Apply overrides if provided
    if config_overrides:
        logger.info(f"Applying configuration overrides: {config_overrides}")
        # Deep merge configuration overrides
        for key, value in config_overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Run training with updated configuration
    return training_flow()


if __name__ == "__main__":
    # Run quick training for testing
    result = quick_training_flow()
    print(f"Training completed. Results: {result}") 