from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load model
try:
    bundle = joblib.load('ml/model.pkl')
    model = bundle['model']
    vectorizer = bundle['vectorizer']
    categories = ['tech', 'business', 'sport', 'entertainment', 'politics']
except:
    model = vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    text = request.json.get('text', '')
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    X = vectorizer.transform([clean_text])
    pred = model.predict(X)[0]
    
    return jsonify({'category': categories[pred]})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)