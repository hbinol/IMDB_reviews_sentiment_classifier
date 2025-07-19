#!/usr/bin/env python3
"""
Test script to verify IMDB Sentiment Classifier setup and functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import setup_logging, load_config
        from src.data.data_loader import IMDBDataLoader
        from src.data.simple_preprocessor import SimpleTextPreprocessor
        from src.features.feature_engineering import FeaturePipeline
        from src.deployment.batch_predict import BatchPredictor
        print("SUCCESS: All imports successful")
        return True
    except Exception as e:
        print(f"ERROR: Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        from src.utils import setup_environment
        config = setup_environment()
        
        # Check if essential config sections exist
        required_sections = ['project', 'data', 'models']
        for section in required_sections:
            if section not in config:
                print(f"ERROR: Missing config section: {section}")
                return False
        
        print("SUCCESS: Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR: Configuration error: {e}")
        return False

def test_text_processing():
    """Test text preprocessing functionality."""
    print("Testing text preprocessing...")
    
    try:
        from src.data.simple_preprocessor import SimpleTextPreprocessor
        
        preprocessor = SimpleTextPreprocessor()
        
        # Test sample text
        sample_text = "<b>This is a great movie!</b> I really enjoyed it."
        processed = preprocessor.preprocess_text(sample_text)
        
        if processed and len(processed) > 0:
            print(f"SUCCESS: Text preprocessing successful: '{processed}'")
            return True
        else:
            print("ERROR: Text preprocessing failed - empty result")
            return False
            
    except Exception as e:
        print(f"ERROR: Text preprocessing error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction."""
    print("Testing feature extraction...")
    
    try:
        from src.features.feature_engineering import FeaturePipeline
        from src.utils import setup_environment
        
        config = setup_environment()
        
        # Create sample data
        texts = [
            "This movie is fantastic and amazing!",
            "Terrible movie, waste of time.",
            "Average film, nothing special.",
            "Excellent cinematography and acting."
        ]
        labels = [1, 0, 0, 1]
        
        # Initialize and test pipeline
        pipeline = FeaturePipeline(config)
        features, encoded_labels = pipeline.fit_transform(texts, labels)
        
        print(f"SUCCESS: Feature extraction successful - Shape: {features.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR: Feature extraction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction functionality (without trained model)."""
    print("Testing batch prediction setup...")
    
    try:
        from src.deployment.batch_predict import create_sample_data
        
        # Create sample data
        sample_file = create_sample_data("data/external/test_reviews.csv")
        
        if Path(sample_file).exists():
            print("SUCCESS: Sample data creation successful")
            
            # Test BatchPredictor initialization (will fail without model, but tests structure)
            from src.deployment.batch_predict import BatchPredictor
            print("SUCCESS: BatchPredictor class accessible")
            return True
        else:
            print("ERROR: Sample data creation failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Batch prediction test error: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "logs",
        "src/data",
        "src/features",
        "src/models",
        "src/deployment",
        "config"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"ERROR: Missing directories: {missing_dirs}")
        return False
    else:
        print("SUCCESS: All required directories exist")
        return True

def test_dependencies():
    """Test if key dependencies are installed."""
    print("Testing dependencies...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"ERROR: Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("SUCCESS: All required packages available")
        return True

def main():
    """Run all tests."""
    print("IMDB Sentiment Classifier - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Text Processing", test_text_processing),
        ("Feature Extraction", test_feature_extraction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\nAll tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run training: make train-quick")
        print("2. Start services: make docker-up")
        print("3. View MLflow: http://localhost:5000")
    else:
        print(f"\n{len(tests) - passed} tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 