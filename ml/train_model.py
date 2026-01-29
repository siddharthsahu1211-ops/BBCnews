#!/usr/bin/env python3
"""
Train and save BBC News classifier model for backend use
"""

import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bbc_classifier import BBCNewsClassifier

def train_and_save_model():
    """Train the best model and save it for backend use"""
    print("Training BBC News Classifier for backend...")
    
    # Initialize and train classifier
    classifier = BBCNewsClassifier()
    
    # Load and preprocess data
    classifier.load_data()
    classifier.preprocess_data()
    classifier.prepare_features()
    
    # Train models
    results = classifier.train_models()
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Create label mapping
    id_to_label = {0: 'tech', 1: 'business', 2: 'sport', 3: 'entertainment', 4: 'politics'}
    
    # Save model bundle
    model_bundle = {
        'model': best_model,
        'vectorizer': classifier.vectorizer,
        'id_to_label': id_to_label
    }
    
    model_path = os.path.join('..', 'ml', 'bbc_model.joblib')
    joblib.dump(model_bundle, model_path)
    print(f"Model saved to {model_path}")
    
    return classifier, best_model_name, best_accuracy

if __name__ == "__main__":
    train_and_save_model()