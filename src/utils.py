"""
Utility functions for IMDB sentiment classification project.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def setup_environment() -> Dict[str, Any]:
    """Set up environment variables and return configuration.
    
    Returns:
        Configuration dictionary from environment and config file
    """
    # Load environment variables
    load_dotenv()
    
    # Load base configuration
    config = load_config()
    
    # Override with environment variables where available
    config.setdefault('mlflow', {})
    config['mlflow']['tracking_uri'] = os.getenv(
        'MLFLOW_TRACKING_URI', 
        config['mlflow'].get('tracking_uri', './mlruns')
    )
    config['mlflow']['experiment_name'] = os.getenv(
        'MLFLOW_EXPERIMENT_NAME',
        config['mlflow'].get('experiment_name', 'imdb-sentiment-classification')
    )
    
    config.setdefault('data', {})
    config['data']['dataset_size'] = int(os.getenv(
        'DATASET_SIZE',
        config['data'].get('dataset_size', 50000)
    ))
    
    return config


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Project root path
    """
    return Path(__file__).parent.parent


def save_predictions(predictions, output_path: str, format: str = "csv") -> None:
    """Save predictions to file.
    
    Args:
        predictions: Predictions to save
        output_path: Output file path
        format: Output format (csv, json, pickle)
    """
    import pandas as pd
    import json
    import pickle
    
    ensure_dir(os.path.dirname(output_path))
    
    if format.lower() == "csv":
        if isinstance(predictions, (list, tuple)):
            pd.DataFrame({"predictions": predictions}).to_csv(output_path, index=False)
        else:
            predictions.to_csv(output_path, index=False)
    elif format.lower() == "json":
        with open(output_path, 'w') as f:
            if hasattr(predictions, 'to_dict'):
                json.dump(predictions.to_dict(), f, indent=2)
            else:
                json.dump(predictions, f, indent=2)
    elif format.lower() == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(predictions, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate common classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted')
    } 