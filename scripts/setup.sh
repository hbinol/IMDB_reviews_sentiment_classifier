#!/bin/bash

# IMDB Sentiment Classifier - Setup Script
echo "Setting up IMDB Sentiment Classifier MLOps Project..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Setting up environment variables..."
    cp .env.example .env
    echo "SUCCESS: Copied .env.example to .env - please update with your settings"
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p tmp/{localstack,prefect}
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p logs

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('NLTK data downloaded successfully')
"

echo "SUCCESS: Setup completed successfully!"
echo ""
echo "Usage:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Check data: make check-data"
echo "  3. Test setup: python scripts/test_setup.py"
echo "  4. Run training: make train-quick"
echo "  5. View help: make help"
echo ""
echo "Make sure to:"
echo "  - Place your IMDB Dataset.csv in data/raw/"
echo "  - Update .env file with your configurations"
echo "  - Run 'make check-data' to validate your dataset" 