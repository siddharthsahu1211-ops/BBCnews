import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class BBCNewsClassifier:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Download NLTK data
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_data = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
            except:
                pass
    
    def load_data(self, csv_path='bbc-text.csv'):
        """Load the BBC news dataset"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found. Creating from folder structure...")
            self._create_dataset_from_folders()
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df.rename(columns={'text': 'News_Headline'})
        print(f"Dataset loaded with {len(self.df)} articles")
        return self.df
    
    def _create_dataset_from_folders(self):
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
    
    def visualize_data(self):
        """Create visualizations of the dataset"""
        plt.figure(figsize=(12, 8))
        
        # Category distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=self.df, y="category", hue="category", palette="Set1", legend=False)
        plt.title("Number of Articles per Category")
        
        # Stop words analysis
        plt.subplot(2, 2, 2)
        self._plot_stopwords()
        plt.title("Top Stop Words")
        
        # Frequent words
        plt.subplot(2, 2, 3)
        self._plot_frequent_words()
        plt.title("Top Non-Stop Words")
        
        plt.tight_layout()
        plt.show()
        
        # Word cloud
        self._plot_wordcloud()
    
    def _plot_stopwords(self):
        """Plot top stop words"""
        stop = set(stopwords.words('english'))
        data_split = self.df['News_Headline'].str.split()
        data_list = data_split.values.tolist()
        corpus = [word for i in data_list for word in i]
        
        dictionary_stopwords = {}
        for word in corpus:
            if word.lower() in stop:
                dictionary_stopwords[word.lower()] = dictionary_stopwords.get(word.lower(), 0) + 1
        
        top = sorted(dictionary_stopwords.items(), key=lambda x: x[1], reverse=True)[:10]
        if top:
            x, y = zip(*top)
            plt.bar(x, y)
            plt.xticks(rotation=45)
    
    def _plot_frequent_words(self):
        """Plot frequent non-stop words"""
        stop = set(stopwords.words('english'))
        data_split = self.df['News_Headline'].str.lower().str.split()
        data_list = data_split.values.tolist()
        corpus = [word for i in data_list for word in i]
        
        counter = Counter(corpus)
        most_common = counter.most_common()
        
        words, counts = [], []
        for word, count in most_common:
            if word not in stop and word.isalpha() and len(word) > 2:
                words.append(word)
                counts.append(count)
            if len(words) == 10:
                break
        
        if words:
            plt.barh(words, counts)
    
    def _plot_wordcloud(self):
        """Generate and display word cloud"""
        stop = set(stopwords.words("english"))
        
        corpus = []
        for text in self.df['News_Headline']:
            tokens = word_tokenize(str(text).lower())
            tokens = [w for w in tokens if w.isalpha() and w not in stop and len(w) > 2]
            corpus.extend(tokens)
        
        wc = WordCloud(
            background_color="white",
            stopwords=STOPWORDS,
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=1
        ).generate(" ".join(corpus))
        
        plt.figure(figsize=(12, 8))
        plt.axis("off")
        plt.imshow(wc)
        plt.title("Word Cloud of BBC News Articles")
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the text data"""
        print("Preprocessing data...")
        
        # Convert to lowercase
        self.df['News_Headline'] = self.df['News_Headline'].str.lower()
        
        # Tokenization
        self.df['text_clean'] = self.df['News_Headline'].apply(word_tokenize)
        
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        self.df['text_clean'] = self.df['text_clean'].apply(
            lambda x: [item for item in x if item not in stop_words and item.isalpha()]
        )
        
        # Stemming
        stemmer = PorterStemmer()
        self.df['text_clean'] = self.df['text_clean'].apply(
            lambda x: [stemmer.stem(y) for y in x]
        )
        
        # Convert back to string for vectorization
        self.df['text_clean_str'] = self.df['text_clean'].apply(lambda x: " ".join(x))
        
        # Encode categories
        category_mapping = {'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4}
        self.df['category_encoded'] = self.df['category'].map(category_mapping)
        
        print("Preprocessing completed!")
        return self.df
    
    def prepare_features(self):
        """Prepare features for model training"""
        # Split data
        X = self.df['text_clean_str']
        y = self.df['category_encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.6, random_state=1
        )
        
        # Vectorize text
        self.vectorizer = CountVectorizer(stop_words="english", lowercase=False)
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"Training set size: {self.X_train_vec.shape}")
        print(f"Test set size: {self.X_test_vec.shape}")
    
    def train_models(self):
        """Train multiple ML models"""
        print("Training models...")
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=1),
            'Random Forest': RandomForestClassifier(random_state=1),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(max_iter=5000, random_state=1)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train_vec, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_vec)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Evaluate and compare models"""
        category_names = ['tech', 'business', 'sport', 'entertainment', 'politics']
        
        for name, result in self.models.items():
            print(f"\n{name} Results:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("\nClassification Report:")
            print(classification_report(
                self.y_test, 
                result['predictions'], 
                target_names=category_names
            ))
    
    def plot_model_comparison(self):
        """Plot model accuracy comparison"""
        accuracies = [result['accuracy'] for result in self.models.values()]
        model_names = list(self.models.keys())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def predict_article(self, text, model_name='Logistic Regression'):
        """Predict category for a new article"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        # Preprocess text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()
        
        clean_tokens = [stemmer.stem(token) for token in tokens 
                       if token not in stop_words and token.isalpha()]
        clean_text = " ".join(clean_tokens)
        
        # Vectorize
        text_vec = self.vectorizer.transform([clean_text])
        
        # Predict
        model = self.models[model_name]['model']
        prediction = model.predict(text_vec)[0]
        
        category_names = ['tech', 'business', 'sport', 'entertainment', 'politics']
        return category_names[prediction]

def main():
    # Initialize classifier
    classifier = BBCNewsClassifier()
    
    # Load and explore data
    df = classifier.load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Categories: {df['category'].value_counts()}")
    
    # Visualize data
    classifier.visualize_data()
    
    # Preprocess data
    classifier.preprocess_data()
    
    # Prepare features
    classifier.prepare_features()
    
    # Train models
    results = classifier.train_models()
    
    # Evaluate models
    classifier.evaluate_models()
    
    # Plot comparison
    classifier.plot_model_comparison()
    
    # Example prediction
    sample_text = "Apple releases new iPhone with advanced AI features"
    prediction = classifier.predict_article(sample_text)
    print(f"\nSample prediction:")
    print(f"Text: {sample_text}")
    print(f"Predicted category: {prediction}")

if __name__ == "__main__":
    main()