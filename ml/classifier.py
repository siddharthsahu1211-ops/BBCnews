import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import re
import joblib

class BBCClassifier:
    def __init__(self):
        self.model = LogisticRegression()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.categories = {'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4}
        
    def load_data(self):
        if not os.path.exists('data.csv'):
            self._create_dataset()
        return pd.read_csv('data.csv')
    
    def _create_dataset(self):
        texts, labels = [], []
        for cat in os.listdir('../bbc'):
            if os.path.isdir(f'../bbc/{cat}'):
                for file in os.listdir(f'../bbc/{cat}'):
                    with open(f'../bbc/{cat}/{file}', 'r', encoding='latin-1') as f:
                        texts.append(re.sub(r'[^a-zA-Z\s]', '', f.read().lower()))
                        labels.append(cat)
        pd.DataFrame({'text': texts, 'label': labels}).to_csv('data.csv', index=False)
    
    def train(self):
        df = self.load_data()
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label'].map(self.categories)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Accuracy: {accuracy:.3f}")
        return accuracy
    
    def predict(self, text):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        X = self.vectorizer.transform([clean_text])
        pred = self.model.predict(X)[0]
        return list(self.categories.keys())[pred]
    
    def save(self, path='model.pkl'):
        joblib.dump({'model': self.model, 'vectorizer': self.vectorizer}, path)

if __name__ == "__main__":
    classifier = BBCClassifier()
    classifier.train()
    
    # Test
    test_texts = [
        "Apple releases new iPhone",
        "Football match results",
        "Stock market update"
    ]
    
    for text in test_texts:
        print(f"'{text}' -> {classifier.predict(text)}")
    
    classifier.save()