# IMDB Reviews Sentiment Classifier - MLOps Project

This is an end-to-end MLOps project for sentiment classification of IMDB movie reviews, built as part of the MLOps Zoomcamp final project.

## Project Overview

This project demonstrates MLOps best practices by building a complete machine learning pipeline for binary sentiment classification (positive/negative) of IMDB movie reviews using traditional ML approaches.

## Technology Stack

- **Machine Learning**: scikit-learn, pandas, numpy
- **Experiment Tracking**: MLflow (optional)
- **Workflow Orchestration**: Prefect (optional)
- **Model Deployment**: Batch prediction service
- **Cloud Simulation**: LocalStack (AWS services locally, optional)
- **Monitoring**: Evidently AI (framework ready, optional)
- **CI/CD**: GitHub Actions (optional)
- **Containerization**: Docker
- **Text Processing**: Custom preprocessing (NLTK-free)

## Project Structure

```
├── data/
│   ├── raw/              # Raw IMDB dataset
│   ├── processed/        # Cleaned and preprocessed data
│   └── external/         # External datasets if needed
├── models/               # Saved model artifacts
├── notebooks/            # Jupyter notebooks for EDA
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and evaluation
│   ├── monitoring/      # Model monitoring and drift detection
│   └── deployment/      # Batch prediction service
├── tests/               # Unit and integration tests
├── config/              # Configuration files
├── logs/                # Application logs
├── docker/              # Docker configurations
└── scripts/             # Setup and utility scripts
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IMDB_reviews_sentiment_classifier
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

### Dataset Setup

Place your Kaggle IMDB dataset CSV file in the `data/raw/` directory:

```bash
data/raw/
└── IMDB Dataset.csv  # Your downloaded Kaggle dataset
```

Supported filenames:
- `IMDB Dataset.csv`
- `imdb_dataset.csv`
- `IMDB_Dataset.csv`
- `imdb-dataset.csv`
- `movie_reviews.csv`

## Pipeline Overview

1. **Data Ingestion**: Load IMDB dataset from Kaggle CSV
2. **Data Preprocessing**: Clean text, remove HTML, stopwords, tokenization
3. **Feature Engineering**: TF-IDF vectorization with n-grams
4. **Model Training**: Train traditional ML models (Logistic Regression, SVM, Random Forest)
5. **Model Evaluation**: Cross-validation and test metrics
6. **Batch Deployment**: Deploy model for batch predictions
7. **Monitoring**: Framework for performance tracking

## Running the Project

### Basic Usage

1. **Validate your dataset**:
```bash
make check-data
# or: PYTHONPATH=. python scripts/check_data.py
```

2. **Test the setup**:
```bash
PYTHONPATH=. python scripts/test_setup.py
```

3. **Run basic training**:
```bash
PYTHONPATH=. python -c "
from src.data.data_loader import IMDBDataLoader
from src.data.simple_preprocessor import SimpleTextPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.utils import setup_environment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess data
loader = IMDBDataLoader()
datasets = loader.create_dataset(dataset_size=2000)
preprocessor = SimpleTextPreprocessor()
for split in datasets:
    datasets[split] = preprocessor.preprocess_dataframe(datasets[split])

# Train model
config = setup_environment()
pipeline = FeaturePipeline(config)
X_train, y_train = pipeline.fit_transform(
    datasets['train']['review_processed'].values,
    datasets['train']['sentiment'].values
)
X_test, y_test = pipeline.transform(
    datasets['test']['review_processed'].values,
    datasets['test']['sentiment'].values
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions):.3f}')
print(f'F1-Score: {f1_score(y_test, predictions):.3f}')
"
```

### Batch Predictions

1. **Create sample data**:
```bash
PYTHONPATH=. python src/deployment/batch_predict.py --create-sample
```

2. **Make predictions** (requires trained model):
```bash
PYTHONPATH=. python src/deployment/batch_predict.py \
    --input-file data/external/sample_reviews.csv \
    --output-file results/predictions.csv \
    --model-path models/your_model.joblib \
    --pipeline-path models/feature_pipeline
```

### Using Make Commands

```bash
# Data operations
make check-data              # Validate Kaggle dataset
make help                   # Show all available commands

# Docker operations (optional)
make docker-up              # Start all services
make docker-down           # Stop all services
```

## Model Performance

Initial results with basic setup:
- **Accuracy**: ~80-85%
- **F1-Score**: ~80-85%
- **Training time**: <1 second (for 2k samples)
- **Features**: 10,000 TF-IDF features

## Architecture

### Core Components

1. **Data Pipeline**: 
   - `IMDBDataLoader`: Handles Kaggle CSV loading and train/val/test splits
   - `SimpleTextPreprocessor`: HTML removal, stopwords, tokenization

2. **Feature Engineering**:
   - `FeaturePipeline`: TF-IDF vectorization with configurable parameters
   - `TextFeatureExtractor`: Scikit-learn compatible transformer

3. **Model Training**:
   - Support for multiple algorithms (LogReg, SVM, Random Forest)
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation and comprehensive metrics

4. **Deployment**:
   - `BatchPredictor`: Process CSV files in batches
   - Model loading from joblib or MLflow
   - Configurable output formats

### Optional MLOps Components

The project includes framework for advanced MLOps features:

- **MLflow**: Experiment tracking and model registry
- **Prefect**: Workflow orchestration
- **LocalStack**: AWS services simulation
- **Evidently**: Model monitoring and drift detection
- **Docker**: Containerization

To enable these features, install additional dependencies:
```bash
pip install mlflow prefect evidently localstack
```

## Configuration

The project uses YAML configuration in `config/config.yaml`:

```yaml
# Data configuration
data:
  dataset_size: 50000  # null for full dataset
  test_size: 0.2
  val_size: 0.1
  random_state: 42

# Feature engineering
features:
  max_features: 10000
  ngram_range: [1, 2]
  use_tfidf: true

# Model parameters
models:
  random_state: 42
  cv_folds: 5
  scoring_metric: "f1"
```

## Testing

Run the test suite to validate your setup:

```bash
PYTHONPATH=. python scripts/test_setup.py
```

This tests:
- Dependencies installation
- Directory structure
- Module imports
- Configuration loading
- Text preprocessing
- Feature extraction
- Batch prediction setup

## Project Status

### Working Components
- Data loading and preprocessing
- Feature engineering with TF-IDF
- Model training (Logistic Regression, SVM, Random Forest)
- Batch prediction service
- Configuration management
- Basic testing framework

### Optional Enhancements
- MLflow experiment tracking
- Prefect workflow orchestration
- Docker containerization
- Model monitoring with Evidently
- CI/CD with GitHub Actions
- Cloud deployment with LocalStack

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

MLOps Zoomcamp Final Project
Project Link: [https://github.com/yourusername/IMDB_reviews_sentiment_classifier](https://github.com/yourusername/IMDB_reviews_sentiment_classifier) 