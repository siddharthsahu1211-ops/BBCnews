from flask import Blueprint, jsonify
import json

# from ml_model import predict_scores, vectorise,model

bp = Blueprint('routes', __name__)

@bp.get("/health")
def health():
    return{"status":"ok"}

#  