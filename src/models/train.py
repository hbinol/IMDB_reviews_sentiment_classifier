"""
Model training functionality with MLflow tracking for IMDB sentiment classification.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
import logging
import time
from pathlib import Path

from src.utils import setup_logging, ensure_dir, calculate_metrics
from src.data.data_loader import IMDBDataLoader
from src.data.preprocessor import TextPreprocessor
from src.features.feature_engineering import FeaturePipeline

logger = setup_logging()


class ModelTrainer:
    """Model trainer with MLflow tracking and hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Initialize models
        self._init_models()
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = mlflow_config.get('experiment_name', 'imdb-sentiment-classification')
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
    def _init_models(self):
        """Initialize models with hyperparameter grids."""
        models_config = self.config.get('models', {})
        random_state = models_config.get('random_state', 42)
        
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state),
                'params': models_config.get('logistic_regression', {
                    'C': [0.1, 1, 10],
                    'max_iter': [1000],
                    'solver': ['liblinear']
                })
            },
            'svm': {
                'model': SVC(random_state=random_state, probability=True),
                'params': models_config.get('svm', {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                })
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': models_config.get('random_forest', {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                })
            }
        }
    
    def train_model(self, 
                   model_name: str,
                   X_train, y_train,
                   X_val, y_val,
                   feature_pipeline: FeaturePipeline) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_pipeline: Fitted feature pipeline
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        param_grid = model_info['params']
        
        logger.info(f"Training {model_name} with hyperparameter tuning...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log model configuration
            mlflow.log_params({
                'model_name': model_name,
                'model_type': type(model).__name__,
                'feature_extractor_type': feature_pipeline.feature_extractor.vectorizer_type,
                'vocabulary_size': feature_pipeline.feature_extractor.vocabulary_size,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val)
            })
            
            # Hyperparameter tuning with cross-validation
            models_config = self.config.get('models', {})
            cv_folds = models_config.get('cv_folds', 5)
            scoring = models_config.get('scoring_metric', 'f1')
            
            grid_search = GridSearchCV(
                model, param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # Train model
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            # Log best parameters
            mlflow.log_params(best_params)
            mlflow.log_metric('cv_score', best_cv_score)
            mlflow.log_metric('training_time', training_time)
            
            # Evaluate on validation set
            val_predictions = best_model.predict(X_val)
            val_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            val_metrics = calculate_metrics(y_val, val_predictions)
            
            if val_proba is not None:
                val_metrics['roc_auc'] = roc_auc_score(y_val, val_proba)
            
            # Log validation metrics
            for metric, value in val_metrics.items():
                mlflow.log_metric(f'val_{metric}', value)
            
            # Log model
            mlflow.sklearn.log_model(
                best_model, 
                model_name,
                registered_model_name=f"imdb_{model_name}"
            )
            
            # Log feature pipeline
            pipeline_path = f"models/pipeline_{model_name}"
            ensure_dir(pipeline_path)
            feature_pipeline.save_pipeline(pipeline_path)
            mlflow.log_artifacts(pipeline_path, "feature_pipeline")
            
            # Generate classification report
            class_report = classification_report(y_val, val_predictions, output_dict=True)
            
            # Log confusion matrix
            conf_matrix = confusion_matrix(y_val, val_predictions)
            
            logger.info(f"{model_name} training completed:")
            logger.info(f"  Best CV score: {best_cv_score:.4f}")
            logger.info(f"  Validation F1: {val_metrics['f1']:.4f}")
            logger.info(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Store results
            results = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_cv_score,
                'val_metrics': val_metrics,
                'training_time': training_time,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'run_id': mlflow.active_run().info.run_id
            }
            
            # Update best model if this one is better
            if val_metrics['f1'] > self.best_score:
                self.best_model = best_model
                self.best_model_name = model_name
                self.best_score = val_metrics['f1']
                
                # Tag as best model
                mlflow.set_tag("best_model", "true")
            
            return results
    
    def train_all_models(self, 
                        X_train, y_train,
                        X_val, y_val,
                        feature_pipeline: FeaturePipeline) -> Dict[str, Dict[str, Any]]:
        """Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_pipeline: Fitted feature pipeline
            
        Returns:
            Dictionary with results for all models
        """
        logger.info("Starting training for all models...")
        
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.train_model(
                    model_name, X_train, y_train, X_val, y_val, feature_pipeline
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Log comparison metrics
        with mlflow.start_run(run_name="model_comparison"):
            comparison_metrics = {}
            for model_name, results in all_results.items():
                for metric, value in results['val_metrics'].items():
                    comparison_metrics[f"{model_name}_{metric}"] = value
            
            mlflow.log_metrics(comparison_metrics)
            
            # Log best model info
            if self.best_model is not None:
                mlflow.log_params({
                    'best_model_name': self.best_model_name,
                    'best_f1_score': self.best_score
                })
        
        logger.info(f"Training completed. Best model: {self.best_model_name} (F1: {self.best_score:.4f})")
        
        return all_results
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with test metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Make predictions
        test_predictions = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, test_predictions)
        
        if test_proba is not None:
            test_metrics['roc_auc'] = roc_auc_score(y_test, test_proba)
        
        # Generate detailed report
        class_report = classification_report(y_test, test_predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, test_predictions)
        
        return {
            'metrics': test_metrics,
            'predictions': test_predictions,
            'probabilities': test_proba,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def save_best_model(self, output_dir: str = "models"):
        """Save the best model to disk.
        
        Args:
            output_dir: Directory to save model
        """
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        ensure_dir(output_dir)
        
        model_path = f"{output_dir}/best_model_{self.best_model_name}.joblib"
        joblib.dump(self.best_model, model_path)
        
        logger.info(f"Saved best model to {model_path}")
        
        return model_path


def run_training_pipeline(config: Dict[str, Any], 
                         dataset_size: Optional[int] = None) -> Dict[str, Any]:
    """Run the complete training pipeline.
    
    Args:
        config: Configuration dictionary
        dataset_size: Optional dataset size limit
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting training pipeline...")
    
    # Load and preprocess data
    data_config = config.get('data', {})
    
    # Load data
    loader = IMDBDataLoader()
    datasets = loader.create_dataset(
        dataset_size=dataset_size or data_config.get('dataset_size'),
        test_size=data_config.get('test_size', 0.2),
        val_size=data_config.get('val_size', 0.1),
        random_state=data_config.get('random_state', 42)
    )
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    for split in datasets:
        datasets[split] = preprocessor.preprocess_dataframe(datasets[split])
    
    # Feature engineering
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
    
    # Train models
    trainer = ModelTrainer(config)
    training_results = trainer.train_all_models(X_train, y_train, X_val, y_val, feature_pipeline)
    
    # Evaluate best model on test set
    if trainer.best_model is not None:
        test_results = trainer.evaluate_model(trainer.best_model, X_test, y_test)
        
        # Log test results in MLflow
        with mlflow.start_run(run_name=f"test_evaluation_{trainer.best_model_name}"):
            for metric, value in test_results['metrics'].items():
                mlflow.log_metric(f'test_{metric}', value)
        
        # Save best model
        model_path = trainer.save_best_model()
        
        return {
            'training_results': training_results,
            'test_results': test_results,
            'best_model_name': trainer.best_model_name,
            'best_model_path': model_path,
            'feature_pipeline': feature_pipeline
        }
    
    return {
        'training_results': training_results,
        'best_model_name': None,
        'feature_pipeline': feature_pipeline
    }


def main():
    """Main function to run training."""
    from src.utils import setup_environment
    
    # Load configuration
    config = setup_environment()
    
    # Run training pipeline
    results = run_training_pipeline(config, dataset_size=1000)  # Small dataset for testing
    
    print("Training completed successfully!")
    print(f"Best model: {results['best_model_name']}")
    
    if 'test_results' in results:
        print(f"Test metrics: {results['test_results']['metrics']}")


if __name__ == "__main__":
    main() 