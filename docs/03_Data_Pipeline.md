# Data Pipeline

## What is a Data Pipeline?

A **data pipeline** is a series of processes that move and transform data from source to destination, ensuring data is clean, validated, and ready for machine learning.

Think of it as an **assembly line for data** - raw materials (data) go in, processed products (features) come out.

## Why Data Pipelines Matter

### The Problem: Manual Data Handling
```python
# Typical notebook approach:
df = pd.read_csv("data.csv")  # Manual download
df = df.dropna()              # Manual cleaning
df['text'] = df['text'].str.lower()  # Manual preprocessing
# What if data changes? Start over manually!
```

### The Solution: Automated Pipeline
```python
# Pipeline approach:
pipeline = DataPipeline()
processed_data = pipeline.execute(input_data)
# Works reliably every time!
```

## Our Data Pipeline Architecture

### Overview
```
Raw Data (CSV) → Validation → Cleaning → Splitting → Processed Data
       ↓              ↓          ↓         ↓            ↓
  Kaggle File    Check Format  Remove   Train/Val/   Ready for
                              HTML/etc   Test        Training
```

### Step-by-Step Breakdown

#### Step 1: Data Loading
```python
class IMDBDataLoader:
    def load_kaggle_dataset(self):
        # 1. Read CSV file
        df = pd.read_csv(self.kaggle_csv_path)
        
        # 2. Validate expected columns exist
        if 'review' not in df.columns:
            raise ValueError("Missing review column")
        
        # 3. Return validated data
        return df
```

**What it does:**
- Loads data from Kaggle CSV format
- Handles different possible file names
- Validates data structure
- Converts sentiment labels to binary format

#### Step 2: Data Validation
```python
def validate_data(self, df):
    """Ensure data quality before processing."""
    
    # Check for required columns
    required_cols = ['review', 'sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    # Check sentiment values
    valid_sentiments = {0, 1}
    invalid_sentiments = set(df['sentiment'].unique()) - valid_sentiments
    if invalid_sentiments:
        raise ValueError(f"Invalid sentiment values: {invalid_sentiments}")
```

**Why validation matters:**
- Catches data quality issues early
- Prevents downstream pipeline failures
- Ensures consistent data format

#### Step 3: Data Cleaning
```python
def clean_text(self, text):
    """Clean individual text sample."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

**Common cleaning steps:**
- Remove HTML tags: `<br/>`, `<b>`, etc.
- Remove URLs and email addresses
- Fix encoding issues
- Normalize whitespace
- Handle special characters

#### Step 4: Data Splitting
```python
def create_splits(self, df, test_size=0.2, val_size=0.1, random_state=42):
    """Create train/validation/test splits."""
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['sentiment']  # Maintain class balance
    )
    
    # Second split: separate validation from training
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=train_val_df['sentiment']
    )
    
    return {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
```

**Why split data?**
- **Training Set (70%)**: Model learns from this
- **Validation Set (10%)**: Tune hyperparameters
- **Test Set (20%)**: Final unbiased evaluation

## Data Pipeline Patterns

### 1. Extract, Transform, Load (ETL)

#### Extract
```python
# Extract data from various sources
csv_data = pd.read_csv("reviews.csv")
api_data = requests.get("api.com/reviews").json()
db_data = pd.read_sql("SELECT * FROM reviews", connection)
```

#### Transform
```python
# Transform data to consistent format
def standardize_data(data):
    data['review'] = data['review'].str.lower()
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data
```

#### Load
```python
# Load into destination (file, database, etc.)
processed_data.to_csv("processed_reviews.csv")
```

### 2. Data Quality Checks

#### Schema Validation
```python
def validate_schema(df):
    expected_schema = {
        'review': 'object',
        'sentiment': 'int64'
    }
    
    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if df[col].dtype != expected_type:
            raise ValueError(f"Wrong type for {col}: expected {expected_type}")
```

#### Data Profiling
```python
def profile_data(df):
    """Generate data quality report."""
    report = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'avg_review_length': df['review'].str.len().mean()
    }
    return report
```

### 3. Data Versioning

#### Why Version Data?
```python
# Model performance drops suddenly
# Which data version was used for training?
# What changed in the data?
# How to reproduce the issue?
```

#### Simple Data Versioning
```python
def save_data_version(df, version):
    """Save data with version metadata."""
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'rows': len(df),
        'columns': list(df.columns),
        'checksum': hashlib.md5(df.to_string().encode()).hexdigest()
    }
    
    # Save data
    df.to_csv(f"data/v{version}/dataset.csv", index=False)
    
    # Save metadata
    with open(f"data/v{version}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

## Handling Different Data Sources

### CSV Files (Our Current Approach)
```python
def load_csv_data(file_path):
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(file_path)
        validate_csv_structure(df)
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise
```

### APIs
```python
def load_api_data(api_url, api_key):
    """Load data from REST API."""
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['reviews'])
        return df
    except Exception as e:
        logger.error(f"Failed to load API data: {e}")
        raise
```

### Databases
```python
def load_database_data(connection_string, query):
    """Load data from database."""
    try:
        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Failed to load database data: {e}")
        raise
```

## Error Handling in Pipelines

### Graceful Failure Handling
```python
class DataPipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

def robust_data_loading(source):
    """Load data with error handling."""
    try:
        if source.endswith('.csv'):
            return load_csv_data(source)
        elif source.startswith('http'):
            return load_api_data(source)
        else:
            raise ValueError(f"Unsupported source: {source}")
            
    except FileNotFoundError:
        logger.error(f"Data file not found: {source}")
        raise DataPipelineError("Data source unavailable")
    
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {source}")
        raise DataPipelineError("No data to process")
    
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise DataPipelineError("Data loading failed")
```

### Retry Logic
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    """Decorator to retry function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return wrapper
        return decorator

@retry(max_attempts=3, delay=2)
def download_data(url):
    """Download data with automatic retries."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

## Pipeline Monitoring

### Logging Pipeline Steps
```python
import logging

def create_pipeline_logger():
    """Set up logging for pipeline monitoring."""
    logger = logging.getLogger('data_pipeline')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler('logs/pipeline.log')
    file_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Usage in pipeline
logger = create_pipeline_logger()

def load_data(source):
    logger.info(f"Starting data load from {source}")
    start_time = time.time()
    
    try:
        data = pd.read_csv(source)
        duration = time.time() - start_time
        logger.info(f"Data loaded successfully in {duration:.2f}s. Rows: {len(data)}")
        return data
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
```

### Data Quality Metrics
```python
def calculate_data_quality_metrics(df):
    """Calculate metrics to monitor data quality."""
    metrics = {
        'completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
        'uniqueness': len(df.drop_duplicates()) / len(df),
        'validity': {
            'sentiment_valid': df['sentiment'].isin([0, 1]).mean(),
            'review_not_empty': (df['review'].str.len() > 0).mean()
        },
        'consistency': {
            'review_length_std': df['review'].str.len().std(),
            'sentiment_balance': abs(df['sentiment'].mean() - 0.5)  # Closer to 0 = more balanced
        }
    }
    return metrics
```

## Best Practices

### 1. **Make Pipelines Idempotent**
```python
def idempotent_data_processing(input_path, output_path):
    """Pipeline that produces same result when run multiple times."""
    
    # Check if output already exists and is up-to-date
    if os.path.exists(output_path):
        input_mtime = os.path.getmtime(input_path)
        output_mtime = os.path.getmtime(output_path)
        
        if output_mtime > input_mtime:
            logger.info("Output is up-to-date, skipping processing")
            return pd.read_csv(output_path)
    
    # Process data
    data = process_data(input_path)
    data.to_csv(output_path, index=False)
    return data
```

### 2. **Use Configuration Files**
```yaml
# config/data_pipeline.yaml
data_sources:
  primary: "data/raw/IMDB Dataset.csv"
  backup: "s3://backup-bucket/imdb-data.csv"

validation:
  required_columns: ["review", "sentiment"]
  min_rows: 1000
  max_missing_percent: 5

splits:
  test_size: 0.2
  val_size: 0.1
  random_state: 42
```

### 3. **Test Your Pipelines**
```python
def test_data_pipeline():
    """Test pipeline with small sample data."""
    # Create test data
    test_data = pd.DataFrame({
        'review': ['Great movie!', 'Terrible film.'],
        'sentiment': [1, 0]
    })
    
    # Run pipeline
    pipeline = DataPipeline()
    result = pipeline.process(test_data)
    
    # Assert expected outcomes
    assert len(result) == 2
    assert 'review_processed' in result.columns
    assert result['sentiment'].dtype == 'int64'
```

### 4. **Document Data Lineage**
```python
def track_data_lineage(data, transformations):
    """Track where data came from and how it was transformed."""
    lineage = {
        'source': data.attrs.get('source', 'unknown'),
        'transformations': transformations,
        'timestamp': datetime.now().isoformat(),
        'shape': data.shape,
        'columns': list(data.columns)
    }
    
    # Attach lineage to data
    data.attrs['lineage'] = lineage
    return data
```

## Common Pitfalls

### 1. **Data Leakage**
```python
# BAD: Using future information
df['target'] = df['future_column'].shift(-1)  # Looks into future!

# GOOD: Only use past information
df['target'] = df['past_column'].shift(1)
```

### 2. **Inconsistent Preprocessing**
```python
# BAD: Different preprocessing for train/test
train_data = preprocess_v1(train_data)  # Old version
test_data = preprocess_v2(test_data)    # New version

# GOOD: Same preprocessing for all data
preprocessor = DataPreprocessor()
train_data = preprocessor.transform(train_data)
test_data = preprocessor.transform(test_data)
```

### 3. **Ignoring Data Distribution Changes**
```python
# Monitor data distribution over time
def check_distribution_shift(reference_data, current_data):
    """Detect if data distribution has changed."""
    from scipy import stats
    
    # Compare review length distributions
    _, p_value = stats.ks_2samp(
        reference_data['review'].str.len(),
        current_data['review'].str.len()
    )
    
    if p_value < 0.05:
        logger.warning("Significant distribution shift detected!")
        return True
    return False
```

## Next Steps

After mastering data pipelines:
1. **Feature Engineering**: Transform raw data into ML features
2. **Model Development**: Use your clean data for training
3. **Monitoring**: Track data quality in production
4. **Workflow Orchestration**: Automate your pipelines

Remember: **Good data pipelines are the foundation of reliable ML systems.** Invest time in building robust, well-tested pipelines - they'll save you countless hours debugging model issues later! 