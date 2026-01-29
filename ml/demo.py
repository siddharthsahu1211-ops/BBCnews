#!/usr/bin/env python3
"""
BBC News Classifier Demo
Run this script to create the dataset and train models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bbc_classifier import BBCNewsClassifier

def quick_demo():
    """Run a quick demo of the classifier"""
    print("=== BBC News Classifier Demo ===\n")
    
    # Initialize classifier
    classifier = BBCNewsClassifier()
    
    # Load data
    print("1. Loading dataset...")
    df = classifier.load_data()
    print(f"   Loaded {len(df)} articles")
    print(f"   Categories: {list(df['category'].value_counts().index)}\n")
    
    # Preprocess
    print("2. Preprocessing data...")
    classifier.preprocess_data()
    print("   Text cleaning completed\n")
    
    # Prepare features
    print("3. Preparing features...")
    classifier.prepare_features()
    print("   Vectorization completed\n")
    
    # Train models
    print("4. Training models...")
    results = classifier.train_models()
    print("\n5. Model Results:")
    for name, result in results.items():
        print(f"   {name}: {result['accuracy']:.3f}")
    
    # Test predictions
    print("\n6. Testing predictions:")
    test_articles = [
        "Apple launches new iPhone with advanced camera technology",
        "Manchester United wins Premier League championship",
        "Government announces new tax policy for businesses",
        "Netflix releases new original series this weekend",
        "Scientists develop breakthrough AI algorithm"
    ]
    
    for article in test_articles:
        prediction = classifier.predict_article(article)
        print(f"   '{article[:50]}...' -> {prediction}")
    
    print("\n=== Demo Complete ===")
    return classifier

if __name__ == "__main__":
    classifier = quick_demo()