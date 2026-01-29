# BBC News Classification ML Module

This module contains machine learning scripts for classifying BBC news articles into categories.

## Files

- `create_dataset.py` - Creates CSV dataset from BBC folder structure
- `bbc_classifier.py` - Main classifier class with preprocessing and training
- `demo.py` - Quick demo script to test the classifier
- `train_model.py` - Trains and saves model for backend use
- `bbc-text.csv` - Generated CSV dataset (created automatically)
- `bbc_model.joblib` - Trained model for backend (created by train_model.py)

## Categories

The classifier predicts 5 categories:
- **tech** - Technology news
- **business** - Business news  
- **sport** - Sports news
- **entertainment** - Entertainment news
- **politics** - Political news

## Usage

### Quick Demo
```bash
cd ml
python demo.py
```

### Train Model for Backend
```bash
cd ml
python train_model.py
```

### Use Classifier in Code
```python
from bbc_classifier import BBCNewsClassifier

classifier = BBCNewsClassifier()
classifier.load_data()
classifier.preprocess_data()
classifier.prepare_features()
classifier.train_models()

# Predict new article
prediction = classifier.predict_article("Apple releases new iPhone")
print(prediction)  # Output: tech
```

## Models Included

1. **Logistic Regression** - Fast and interpretable
2. **Random Forest** - Ensemble method with good performance
3. **Naive Bayes** - Classic text classification algorithm
4. **Linear SVM** - Support Vector Machine for text

## Features

- Automatic dataset creation from folder structure
- Text preprocessing (tokenization, stop word removal, stemming)
- Multiple model training and comparison
- Visualization of data distribution and word clouds
- Model evaluation with accuracy and classification reports
- Easy prediction interface for new articles

## Dependencies

All dependencies are listed in the main `requirements.txt` file.