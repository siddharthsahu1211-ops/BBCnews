import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import re
import joblib

class SimpleBBCClassifier:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.models = {}
        self.category_mapping = {'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4}
        self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
    
    def load_data(self):
        """Load or create the BBC dataset"""
        if not os.path.exists('bbc-text.csv'):
            self._create_dataset()
        
        self.df = pd.read_csv('bbc-text.csv')
        print(f"Dataset loaded: {len(self.df)} articles")
        print(f"Categories: {self.df['category'].value_counts()}")
        return self.df
    
    def _create_dataset(self):
        """Create dataset from BBC folder structure"""
        texts = []
        categories = []
        base_path = "../bbc"
        
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if not os.path.isdir(category_path):
                continue
            
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        texts.append(f.read())
                        categories.append(category)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        df = pd.DataFrame({"text": texts, "category": categories})
        df.to_csv("bbc-text.csv", index=False)
        print(f"Dataset created with {len(df)} articles")
    
    def clean_text(self, text):
        """Simple text cleaning"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def preprocess_data(self):
        """Preprocess the text data"""
        print("Preprocessing data...")
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        self.df['category_encoded'] = self.df['category'].map(self.category_mapping)
        print("Preprocessing completed!")
        return self.df
    
    def train_models(self, test_size=0.3, random_state=42):
        """Train multiple models"""
        print("Training models...")
        
        X = self.df['clean_text']
        y = self.df['category_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(random_state=random_state, max_iter=2000)
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_vec, y_train)
            
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def predict_article(self, text, model_name='Logistic Regression'):
        """Predict category for new article"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        clean_text = self.clean_text(text)
        text_vec = self.vectorizer.transform([clean_text])
        
        model = self.models[model_name]['model']
        prediction = model.predict(text_vec)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vec)[0]
            prob_dict = {self.reverse_mapping[i]: prob for i, prob in enumerate(probabilities)}
            return self.reverse_mapping[prediction], prob_dict
        
        return self.reverse_mapping[prediction], None
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.models:
            return None
        
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        return best_model_name, self.models[best_model_name]
    
    def save_model(self, model_name='Logistic Regression', filename='bbc_model.joblib'):
        """Save model for backend use"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model_bundle = {
            'model': self.models[model_name]['model'],
            'vectorizer': self.vectorizer,
            'id_to_label': self.reverse_mapping
        }
        
        joblib.dump(model_bundle, filename)
        print(f"Model saved to {filename}")

def main():
    classifier = SimpleBBCClassifier()
    
    classifier.load_data()
    classifier.preprocess_data()
    classifier.train_models()
    
    print("\nModel Results:")
    for name, result in classifier.models.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    best_model_name, _ = classifier.get_best_model()
    print(f"\nBest model: {best_model_name}")
    
    test_articles = [
        "Apple launches new iPhone with advanced camera technology",
        "Manchester United wins Premier League championship",
        "Government announces new tax policy for businesses",
        "Netflix releases new original series this weekend",
        "Scientists develop breakthrough AI algorithm"
    ]
    
    print(f"\nTesting predictions with {best_model_name}:")
    for article in test_articles:
        prediction, probabilities = classifier.predict_article(article, best_model_name)
        print(f"'{article[:50]}...' -> {prediction}")
        if probabilities:
            top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3: {', '.join([f'{cat}({prob:.2f})' for cat, prob in top_3])}")
    
    # Save the best model
    classifier.save_model(best_model_name)

if __name__ == "__main__":
    main()