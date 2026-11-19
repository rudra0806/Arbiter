"""
ML-based router: predicts whether the local model will answer a prompt
correctly, using the trained TF-IDF + logistic regression model.
"""
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

_MODEL = None
_VECTORIZER = None

MODEL_PATH = Path(__file__).parent / "router_model.pkl"
VECT_PATH = Path(__file__).parent / "router_vectorizer.pkl"

def _load():
    global _MODEL, _VECTORIZER
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
        _VECTORIZER = joblib.load(VECT_PATH)

def _numeric_features(prompt):
    length = len(prompt)
    digits = sum(c.isdigit() for c in prompt)
    ops = sum(prompt.count(c) for c in "+-*/%")
    has_q = 1 if "?" in prompt else 0
    words = len(prompt.split())
    return np.array([[length, digits, ops, has_q, words]], dtype=float)

def route_ml(prompt: str, threshold: float = 0.5) -> dict:
    _load()
    if not prompt or not prompt.strip():
        return {"target": "api", "reason": "ml: empty prompt, escalating by default"}
    X_text = _VECTORIZER.transform([prompt])
    X_num = csr_matrix(_numeric_features(prompt))
    X = hstack([X_text, X_num]).tocsr()
    prob_correct = _MODEL.predict_proba(X)[0, 1]
    target = "local" if prob_correct >= threshold else "api"
    reason = f"ml: p_correct={prob_correct:.3f} (threshold={threshold})"
    return {"target": target, "reason": reason}