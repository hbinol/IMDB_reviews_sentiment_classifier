# MLOps Concepts Explained: A Beginner's Guide

## Table of Contents
1. [What is MLOps?](#what-is-mlops)
2. [Project Structure & Organization](#project-structure--organization)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Experiment Tracking](#experiment-tracking)
7. [Model Deployment](#model-deployment)
8. [Workflow Orchestration](#workflow-orchestration)
9. [Containerization](#containerization)
10. [Monitoring](#monitoring)
11. [Best Practices](#best-practices)
12. [Cloud & Infrastructure](#cloud--infrastructure)

---

## What is MLOps?

### Definition
**MLOps** (Machine Learning Operations) is a set of practices that combines Machine Learning (ML) and DevOps to standardize and streamline the ML lifecycle.

### Why Do We Need MLOps?

#### Traditional Data Science Challenges:
```
Data Scientist's Laptop → Jupyter Notebook → Model.pkl → ???
```

**Problems:**
- Models work on laptops but fail in production
- No version control for data or models
- Manual, error-prone processes
- Difficult to reproduce results
- Hard to monitor model performance over time

#### MLOps Solution:
```
Data → Pipeline → Training → Testing → Deployment → Monitoring → Feedback
```

### Key Benefits:
1. **Reproducibility**: Same results every time
2. **Scalability**: Handle large datasets and many models
3. **Reliability**: Automated testing and validation
4. **Collaboration**: Teams can work together effectively
5. **Monitoring**: Track model performance in production

---

## Project Structure & Organization

### Why Structure Matters
In our project, we organized code like this:
```
├── data/                 # All data-related files
├── src/                  # Source code
│   ├── data/            # Data handling code
│   ├── features/        # Feature engineering
│   ├── models/          # Model training
│   └── deployment/      # Deployment code
├── config/              # Configuration files
├── scripts/             # Utility scripts
└── tests/               # Test files
```

### Benefits of This Structure:

#### 1. **Separation of Concerns**
Each directory has a specific purpose:
- **Data scientists** work in `src/models/`
- **Data engineers** focus on `src/data/`
- **ML engineers** handle `src/deployment/`

#### 2. **Modularity**
```python
# Instead of one giant notebook:
from src.data.data_loader import IMDBDataLoader
from src.features.feature_engineering import FeaturePipeline
from src.models.train import ModelTrainer
```

#### 3. **Reusability**
Code can be imported and reused across different scripts and notebooks.

### Configuration Management

Instead of hardcoding values:
```python
# BAD: Hard-coded values
max_features = 10000
test_size = 0.2
```

We use configuration files:
```yaml
# config/config.yaml
features:
  max_features: 10000
data:
  test_size: 0.2
```

```python
# GOOD: Configuration-driven
config = load_config()
max_features = config['features']['max_features']
```

**Benefits:**
- Easy to change parameters without modifying code
- Different configurations for development/production
- Version control for hyperparameters

---

## Data Pipeline

### What is a Data Pipeline?
A data pipeline is a series of processes that move and transform data from source to destination.

### Our Data Pipeline Steps:

#### 1. **Data Loading**
```python
# src/data/data_loader.py
class IMDBDataLoader:
    def load_kaggle_dataset(self):
        # Load CSV file
        # Validate columns
        # Check data quality
```

**What it does:**
- Loads IMDB dataset from CSV
- Validates data format
- Handles missing values
- Creates train/validation/test splits

#### 2. **Data Validation**
```python
# Check data quality
if 'review' not in df.columns:
    raise ValueError("Missing review column")
```

**Why it's important:**
- Catches data issues early
- Prevents downstream failures
- Ensures data quality

#### 3. **Train/Validation/Test Splits**
```
Original Dataset (50k samples)
├── Training Set (70% = 35k)     # Learn patterns
├── Validation Set (10% = 5k)    # Tune hyperparameters  
└── Test Set (20% = 10k)         # Final evaluation
```

**Why split?**
- **Training**: Model learns from this data
- **Validation**: Used to tune hyperparameters
- **Test**: Unbiased evaluation of final model

### Data Versioning Concepts
Though not fully implemented, we prepared for:
- **Data Lineage**: Track where data comes from
- **Data Versioning**: Track changes to datasets
- **Schema Evolution**: Handle changes in data structure

---

## Feature Engineering

### What is Feature Engineering?
Converting raw data into features that machine learning algorithms can understand.

### Text Preprocessing Pipeline

#### 1. **HTML Cleaning**
```python
# Raw text
"<b>This movie is great!</b>"

# After cleaning
"This movie is great!"
```

#### 2. **Tokenization**
```python
# Text
"This movie is great!"

# After tokenization
["This", "movie", "is", "great"]
```

#### 3. **Stopword Removal**
```python
# Tokens
["This", "movie", "is", "great"]

# After stopword removal (remove "is")
["This", "movie", "great"]
```

#### 4. **Normalization**
```python
# Mixed case
["This", "Movie", "GREAT"]

# After normalization
["this", "movie", "great"]
```

### TF-IDF Vectorization

#### What is TF-IDF?
**Term Frequency-Inverse Document Frequency** converts text to numbers.

#### Example:
```
Reviews:
1. "This movie is great"
2. "This movie is terrible"

TF-IDF creates features like:
- "this": [0.2, 0.2]     # Common word, lower score
- "movie": [0.2, 0.2]    # Common word, lower score  
- "great": [0.8, 0.0]    # Rare word, higher score
- "terrible": [0.0, 0.8] # Rare word, higher score
```

#### Why TF-IDF?
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare a word is across all documents
- Gives higher weights to important, distinguishing words

### Feature Pipeline
```python
class FeaturePipeline:
    def fit_transform(self, texts, labels):
        # 1. Create vocabulary
        # 2. Transform texts to vectors
        # 3. Encode labels
        # 4. Save pipeline for reuse
```

**Benefits:**
- Consistent preprocessing
- Reusable for new data
- Version controlled transformations

---

## Model Development

### Traditional ML vs Deep Learning

#### Our Approach: Traditional ML
```python
# Models we use:
- Logistic Regression
- Support Vector Machine (SVM)  
- Random Forest
```

**Why Traditional ML?**
- **Faster training**: Minutes vs hours/days
- **Less data needed**: Works with smaller datasets
- **Interpretable**: Can understand feature importance
- **Stable**: Consistent results

#### Model Training Process
```python
# 1. Initialize model
model = LogisticRegression(max_iter=1000)

# 2. Train on training data
model.fit(X_train, y_train)

# 3. Evaluate on test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

### Hyperparameter Tuning

#### What are Hyperparameters?
Settings you choose before training:
```python
# Examples:
LogisticRegression(
    C=1.0,           # Regularization strength
    max_iter=1000,   # Maximum iterations
    random_state=42  # For reproducibility
)
```

#### Grid Search
```python
# Try different combinations:
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 1000]
}

# Best combination based on validation scores
```

### Cross-Validation
Instead of one train/test split:
```
Fold 1: Train[2,3,4,5] → Test[1]
Fold 2: Train[1,3,4,5] → Test[2]  
Fold 3: Train[1,2,4,5] → Test[3]
Fold 4: Train[1,2,3,5] → Test[4]
Fold 5: Train[1,2,3,4] → Test[5]

Average performance across all folds
```

**Benefits:**
- More robust evaluation
- Better estimate of model performance
- Reduces overfitting

---

## Experiment Tracking

### Why Track Experiments?

#### The Problem:
```
experiment_1.ipynb
experiment_2_final.ipynb  
experiment_2_final_FINAL.ipynb
experiment_actually_final.ipynb
```

**What got lost:**
- Which parameters were used?
- What was the accuracy?
- Which dataset version?
- How to reproduce results?

### MLflow Solution

#### What MLflow Tracks:
```python
import mlflow

with mlflow.start_run():
    # Parameters
    mlflow.log_param("max_features", 10000)
    mlflow.log_param("model_type", "logistic_regression")
    
    # Metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.84)
    
    # Model
    mlflow.sklearn.log_model(model, "model")
    
    # Artifacts (files)
    mlflow.log_artifact("feature_importance.png")
```

#### MLflow UI:
```
Experiment: IMDB-Sentiment
├── Run 1: LogReg, max_features=5000  → Accuracy: 0.82
├── Run 2: LogReg, max_features=10000 → Accuracy: 0.85  
└── Run 3: SVM, C=1.0                 → Accuracy: 0.84
```

### Model Registry
```python
# Register best model
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="imdb-sentiment-classifier"
)

# Version management
# v1: Logistic Regression
# v2: SVM  
# v3: Random Forest
```

---

## Model Deployment

### Batch vs Real-time Inference

#### Batch Prediction (Our Approach)
```
Input: CSV file with 1000 reviews
Process: Run predictions on all at once
Output: CSV file with predictions
```

**Use cases:**
- Daily/weekly predictions
- Large datasets
- Non-urgent results

#### Real-time Prediction (Alternative)
```
Input: Single review via API
Process: Immediate prediction  
Output: Instant result
```

**Use cases:**
- Web applications
- Mobile apps
- Interactive systems

### Our Batch Predictor

#### How it Works:
```python
class BatchPredictor:
    def __init__(self, model_path):
        # 1. Load trained model
        self.model = joblib.load(model_path)
        
    def predict_batch_file(self, input_file):
        # 2. Load input data
        # 3. Preprocess text
        # 4. Extract features  
        # 5. Make predictions
        # 6. Save results
```

#### Example Usage:
```bash
python batch_predict.py \
    --input-file reviews.csv \
    --output-file predictions.csv
```

### Model Serialization
```python
# Save model
joblib.dump(model, "model.joblib")

# Load model later
model = joblib.load("model.joblib")
```

**Why serialize?**
- Save trained models for later use
- Share models between team members
- Deploy models to production

---

## Workflow Orchestration

### The Problem: Manual Steps
```
1. Download data (manual)
2. Preprocess data (manual)
3. Train model (manual)  
4. Evaluate model (manual)
5. Deploy if good (manual)
```

**Issues:**
- Error-prone
- Time-consuming
- Hard to reproduce
- No automatic retries

### Prefect Solution

#### What is Prefect?
A workflow orchestration tool that automates ML pipelines.

#### Example Pipeline:
```python
from prefect import flow, task

@task
def load_data():
    return IMDBDataLoader().create_dataset()

@task  
def preprocess_data(datasets):
    # Clean and prepare data
    return processed_datasets

@task
def train_model(processed_data):
    # Train and return model
    return trained_model

@flow
def training_pipeline():
    # Define workflow
    data = load_data()
    processed = preprocess_data(data)
    model = train_model(processed)
    return model
```

#### Benefits:
- **Automation**: Runs without manual intervention
- **Retry Logic**: Automatically retries failed tasks
- **Monitoring**: Track pipeline progress
- **Scheduling**: Run daily/weekly automatically
- **Parallel Execution**: Run independent tasks simultaneously

#### Task Dependencies:
```
load_data() → preprocess_data() → train_model()
     ↓              ↓                 ↓
   Raw Data → Processed Data → Trained Model
```

---

## Containerization

### The Problem: "It Works on My Machine"
```
Data Scientist: "My model works perfectly!"
Production: "Model crashes with dependency errors"
```

**Common issues:**
- Different Python versions
- Missing libraries
- OS differences
- Environment variables

### Docker Solution

#### What is Docker?
Packages your application with all its dependencies into a "container."

#### Dockerfile Example:
```dockerfile
# Start with Python base image
FROM python:3.8

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY src/ src/

# Set entry point
CMD ["python", "src/models/train.py"]
```

#### Benefits:
- **Consistency**: Same environment everywhere
- **Portability**: Runs on any system with Docker
- **Isolation**: No conflicts with other applications
- **Reproducibility**: Exact same setup every time

### Docker Compose

#### Managing Multiple Services:
```yaml
# docker-compose.yml
services:
  mlflow:
    image: mlflow-server
    ports:
      - "5000:5000"
  
  app:
    build: .
    depends_on:
      - mlflow
    
  database:
    image: postgres:13
```

#### Benefits:
- Start all services with one command: `docker-compose up`
- Automatic service discovery
- Network isolation
- Volume management

---

## Monitoring

### Why Monitor ML Models?

#### Models Degrade Over Time:
```
Month 1: 85% accuracy
Month 3: 82% accuracy  
Month 6: 78% accuracy
Month 12: 65% accuracy
```

**Why does this happen?**
- **Data Drift**: Input data changes over time
- **Concept Drift**: Relationship between input and output changes
- **Model Staleness**: Model becomes outdated

### Types of Monitoring

#### 1. **Data Drift Detection**
```python
# Training data vocabulary
train_words = ["great", "amazing", "terrible", "awful"]

# Production data vocabulary  
prod_words = ["epic", "fire", "mid", "cringe"]  # New slang!

# Drift detected: New vocabulary not seen in training
```

#### 2. **Model Performance Monitoring**
```python
# Track metrics over time
accuracy_over_time = [0.85, 0.84, 0.82, 0.78]

# Alert if performance drops
if current_accuracy < threshold:
    send_alert("Model performance degraded!")
```

#### 3. **Feature Distribution Monitoring**
```python
# Training: Average review length = 150 words
# Production: Average review length = 50 words

# Distribution shift detected!
```

### Evidently AI (Our Framework)

#### What it Monitors:
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Create monitoring report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=prod_data)
```

#### Output:
- Data drift reports
- Model performance dashboards
- Alerts and notifications
- Historical trend analysis

---

## Best Practices

### Code Quality

#### 1. **Linting (Flake8)**
Checks for code style issues:
```python
# BAD
def train_model(data,model_type,params):
    if model_type=='lr':
        model=LogisticRegression(**params)
    return model

# GOOD  
def train_model(data, model_type, params):
    if model_type == 'lr':
        model = LogisticRegression(**params)
    return model
```

#### 2. **Code Formatting (Black)**
Automatically formats code:
```bash
# Before: Inconsistent formatting
# After: Consistent, readable code
black src/
```

#### 3. **Import Sorting (isort)**
Organizes imports:
```python
# GOOD
import os
import sys

import pandas as pd
import numpy as np

from src.data.loader import DataLoader
```

### Testing

#### Unit Tests
```python
def test_data_loader():
    loader = IMDBDataLoader()
    data = loader.load_kaggle_dataset()
    
    # Test data shape
    assert len(data) > 0
    assert 'review' in data.columns
    assert 'sentiment' in data.columns
```

#### Integration Tests
```python
def test_full_pipeline():
    # Test complete workflow
    loader = IMDBDataLoader()
    preprocessor = SimpleTextPreprocessor()
    pipeline = FeaturePipeline()
    
    # Should run without errors
    datasets = loader.create_dataset(dataset_size=100)
    processed = preprocessor.preprocess_dataframe(datasets['train'])
    features = pipeline.fit_transform(processed['review_processed'])
```

### Documentation

#### Docstrings
```python
def preprocess_text(self, text: str) -> str:
    """Apply all preprocessing steps to a single text.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text string
        
    Example:
        >>> preprocessor = SimpleTextPreprocessor()
        >>> result = preprocessor.preprocess_text("<b>Great movie!</b>")
        >>> print(result)
        "great movie"
    """
```

#### README Files
- Project overview
- Setup instructions  
- Usage examples
- API documentation

### Version Control

#### Git Best Practices
```bash
# Meaningful commit messages
git commit -m "Add TF-IDF feature extraction pipeline"

# Feature branches
git checkout -b feature/batch-prediction

# Small, focused commits
git add src/data/preprocessor.py
git commit -m "Add text preprocessing functionality"
```

#### .gitignore
```
# Don't commit
*.pyc                 # Compiled Python files
__pycache__/          # Python cache
data/raw/*.csv        # Large data files
models/*.joblib       # Trained models
.env                  # Secrets
```

---

## Cloud & Infrastructure

### LocalStack (Local AWS Simulation)

#### Why LocalStack?
Develop and test AWS services locally without costs:

```python
# Instead of real AWS S3
import boto3
s3 = boto3.client('s3', endpoint_url='http://localhost:4566')

# Upload model to local S3
s3.upload_file('model.joblib', 'ml-models', 'sentiment-model.joblib')
```

#### Services We Can Simulate:
- **S3**: Store models and data
- **Lambda**: Serverless prediction functions
- **SQS**: Message queues for batch processing
- **CloudWatch**: Monitoring and logging

### Infrastructure as Code (IaC)

#### The Problem: Manual Setup
```
1. Create S3 bucket (manual)
2. Set up IAM permissions (manual)  
3. Configure VPC (manual)
4. Deploy application (manual)
```

#### IaC Solution (Terraform Example):
```hcl
# infrastructure/main.tf
resource "aws_s3_bucket" "ml_models" {
  bucket = "my-ml-models"
}

resource "aws_lambda_function" "predict" {
  filename = "prediction_function.zip"
  function_name = "sentiment-predictor"
  runtime = "python3.8"
}
```

**Benefits:**
- **Reproducible**: Same infrastructure every time
- **Version Controlled**: Track infrastructure changes
- **Automated**: Deploy with one command
- **Cost Effective**: Destroy when not needed

### CI/CD for ML

#### Continuous Integration
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies  
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
      - name: Run linting
        run: flake8 src/
```

#### Continuous Deployment
```yaml
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Train model
        run: python src/models/train.py
      - name: Deploy to production
        run: ./scripts/deploy.sh
```

---

## Putting It All Together

### Our MLOps Pipeline Flow

```
1. Data Ingestion
   ├── Load Kaggle CSV
   ├── Validate data quality
   └── Create train/val/test splits

2. Data Processing  
   ├── Clean text (remove HTML, etc.)
   ├── Tokenization and normalization
   └── Feature extraction (TF-IDF)

3. Model Training
   ├── Train multiple models (LogReg, SVM, RF)
   ├── Hyperparameter tuning with GridSearch
   ├── Cross-validation evaluation
   └── Select best model

4. Experiment Tracking (MLflow)
   ├── Log parameters and metrics
   ├── Save model artifacts
   └── Register best model

5. Model Deployment
   ├── Serialize model and pipeline
   ├── Create batch prediction service
   └── Test with sample data

6. Monitoring (Evidently)
   ├── Track model performance
   ├── Detect data drift
   └── Alert on issues

7. Orchestration (Prefect)
   ├── Automate training pipeline
   ├── Schedule regular retraining
   └── Handle failures gracefully

8. Infrastructure (Docker + LocalStack)
   ├── Containerize application
   ├── Simulate cloud services locally
   └── Prepare for production deployment
```

### Key Learnings

#### 1. **Start Simple, Scale Gradually**
- Begin with basic ML pipeline
- Add MLOps components incrementally
- Don't over-engineer from the start

#### 2. **Automation is Key**
- Manual processes don't scale
- Automate testing, training, deployment
- Use orchestration tools for complex workflows

#### 3. **Monitoring is Critical**
- Models degrade over time
- Catch issues before they impact users
- Plan monitoring from the beginning

#### 4. **Reproducibility Matters**
- Version everything: code, data, models
- Use configuration management
- Document all processes

#### 5. **Collaboration Enables Success**
- Structure code for team collaboration
- Use consistent tools and practices
- Document for future team members

---

## Next Steps for Learning

### Beginner Level
1. **Practice Git**: Version control fundamentals
2. **Learn Docker**: Containerization basics
3. **Understand ML Lifecycle**: From data to deployment
4. **Practice Python**: OOP, modules, testing

### Intermediate Level
1. **Experiment Tracking**: Deep dive into MLflow
2. **Pipeline Orchestration**: Prefect/Airflow
3. **Cloud Platforms**: AWS/GCP/Azure basics
4. **Monitoring Tools**: Evidently, Weights & Biases

### Advanced Level
1. **Infrastructure as Code**: Terraform, CloudFormation
2. **Kubernetes**: Container orchestration
3. **CI/CD for ML**: GitHub Actions, Jenkins
4. **Model Serving**: FastAPI, TorchServe, MLflow serving

### Recommended Resources
1. **Books**: 
   - "Building Machine Learning Pipelines" by Hannes Hapke
   - "ML Engineering" by Andriy Burkov

2. **Courses**:
   - MLOps Zoomcamp (DataTalks.Club)
   - Machine Learning Engineering for Production (Coursera)

3. **Practice Projects**:
   - Build end-to-end ML projects
   - Contribute to open-source ML tools
   - Deploy models to cloud platforms

---

## Conclusion

MLOps transforms machine learning from experimental code to production-ready systems. By implementing these practices:

- **Your models become reliable and reproducible**
- **Your team can collaborate effectively**  
- **Your solutions scale to handle real-world demands**
- **Your career advances to ML Engineering roles**

The concepts in this project provide a solid foundation for building robust ML systems. Start with the basics, practice regularly, and gradually incorporate more advanced techniques as you grow your expertise.

Remember: MLOps is a journey, not a destination. Keep learning, experimenting, and improving your practices! 