import os
import joblib

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "bbc_model.joblib"))

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]
id_to_label = bundle.get("id_to_label")

def predict_text(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    if id_to_label is not None:
        pred = id_to_label[int(pred)]
    return str(pred)

def predict_scores(text: str):
    """
    Returns (category, scores_dict_or_None)
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    scores = None

    if id_to_label is not None:
        pred = id_to_label[int(pred)]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        if id_to_label is not None:
            classes = [id_to_label[int(c)] for c in classes]
        scores = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

    return str(pred), scores