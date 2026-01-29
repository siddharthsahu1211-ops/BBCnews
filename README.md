# BBCnews

A machine learning project for classifying BBC news articles into categories using various ML algorithms.

## Project Structure

```
BBCnews/
├── bbc/                    # Original BBC dataset (5 categories)
│   ├── business/
│   ├── entertainment/
│   ├── politics/
│   ├── sport/
│   └── tech/
├── ml/                     # Machine Learning module
│   ├── simple_classifier.py    # Main classifier (recommended)
│   ├── bbc_classifier.py      # Advanced classifier with visualizations
│   ├── create_dataset.py      # Dataset creation utility
│   ├── demo.py               # Quick demo script
│   ├── train_model.py        # Train model for backend
│   ├── bbc-text.csv          # Generated CSV dataset
│   ├── bbc_model.joblib      # Trained model for backend
│   └── README.md             # ML module documentation
├── beckend/                # Flask backend API
│   ├── app.py
│   ├── ml_model.py
│   ├── routes.py
│   └── db.py
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

- **Text Classification**: Classify BBC news articles into 5 categories:
  - Technology
  - Business
  - Sports
  - Entertainment
  - Politics

- **Multiple ML Models**: Compare performance of:
  - Logistic Regression (98.5% accuracy)
  - Linear SVM (97.8% accuracy)
  - Naive Bayes (97.6% accuracy)
  - Random Forest (96.7% accuracy)

- **Text Preprocessing**: Automatic text cleaning and feature extraction
- **API Backend**: Flask API for real-time predictions
- **Easy Integration**: Trained models saved for production use

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Classifier
```bash
cd ml
python simple_classifier.py
```

### 3. Test Predictions
```python
from ml.simple_classifier import SimpleBBCClassifier

classifier = SimpleBBCClassifier()
classifier.load_data()
classifier.preprocess_data()
classifier.train_models()

# Predict new article
prediction, probabilities = classifier.predict_article(
    "Apple launches new iPhone with advanced AI features"
)
print(f"Category: {prediction}")
```

### 4. Start Backend API
```bash
cd beckend
python app.py
```

## Dataset

The BBC News dataset contains 2,225 articles:
- **Sport**: 511 articles
- **Business**: 510 articles
- **Politics**: 417 articles
- **Tech**: 401 articles
- **Entertainment**: 386 articles

## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.5% |
| Linear SVM | 97.8% |
| Naive Bayes | 97.6% |
| Random Forest | 96.7% |

## Usage Examples

### Command Line
```bash
# Quick demo
cd ml && python demo.py

# Train and save model
cd ml && python train_model.py

# Create dataset from folders
cd ml && python create_dataset.py
```

### Python API
```python
# Load and use trained model
import joblib

bundle = joblib.load('ml/bbc_model.joblib')
model = bundle['model']
vectorizer = bundle['vectorizer']

# Predict new text
text_vec = vectorizer.transform(["Your news article here"])
prediction = model.predict(text_vec)[0]
print(f"Category: {bundle['id_to_label'][prediction]}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
