"""
Train an ML router: predicts whether the LOCAL model will answer a prompt
correctly, using only the prompt text as input. Trains on dev+pool splits
only. Never touches test/final splits.
"""
import json
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
MODEL_OUT = "gateway/router_model.pkl"
VECT_OUT = "gateway/router_vectorizer.pkl"
META_OUT = "gateway/router_ml_meta.json"

def load_data():
    tasks = json.load(open(TASKS_PATH, encoding="utf-8"))
    local = json.load(open(LOCAL_RESULTS_PATH, encoding="utf-8"))
    local_by_id = {r["id"]: r for r in local}

    rows = []
    for t in tasks:
        if t.get("invalid"):
            continue
        if t.get("split") not in ("dev", "pool"):
            continue
        rec = local_by_id.get(t["id"])
        if rec is None or rec.get("error"):
            continue
        rows.append({
            "id": t["id"],
            "prompt": t["prompt"],
            "label": 1 if rec.get("correct") else 0,
            "group": t.get("template") or t.get("category") or "unknown",
            "category": t.get("category"),
        })
    df = pd.DataFrame(rows)
    return df

def extract_numeric_features(prompts):
    feats = []
    for p in prompts:
        length = len(p)
        digits = sum(c.isdigit() for c in p)
        ops = sum(p.count(c) for c in "+-*/%")
        has_q = 1 if "?" in p else 0
        words = len(p.split())
        feats.append([length, digits, ops, has_q, words])
    return np.array(feats, dtype=float)

def main():
    df = load_data()
    print(f"Loaded {len(df)} eligible tasks (dev+pool, excluding invalid/errors).")
    print(f"Label balance: {df['label'].value_counts().to_dict()}")
    print(f"Distinct groups used for CV: {df['group'].nunique()}")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X_text = vectorizer.fit_transform(df["prompt"])
    X_num = extract_numeric_features(df["prompt"].tolist())
    X = hstack([X_text, csr_matrix(X_num)]).tocsr()
    y = df["label"].values
    groups = df["group"].values

    n_splits = min(5, df["group"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        acc = accuracy_score(y[val_idx], preds)
        prec = precision_score(y[val_idx], preds, zero_division=0)
        rec = recall_score(y[val_idx], preds, zero_division=0)
        try:
            auc = roc_auc_score(y[val_idx], probs)
        except ValueError:
            auc = float("nan")
        cm = confusion_matrix(y[val_idx], preds).tolist()
        fold_results.append({"fold": fold, "accuracy": acc, "precision": prec,
                              "recall": rec, "auc": auc, "confusion_matrix": cm,
                              "n_val": len(val_idx)})
        print(f"Fold {fold}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} "
              f"auc={auc:.3f} n_val={len(val_idx)} confmat={cm}")

    mean_acc = np.mean([f["accuracy"] for f in fold_results])
    std_acc = np.std([f["accuracy"] for f in fold_results])
    mean_auc = np.nanmean([f["auc"] for f in fold_results])
    std_auc = np.nanstd([f["auc"] for f in fold_results])
    baseline_acc = df["label"].mean()
    print("=" * 60)
    print(f"CROSS-VAL SUMMARY ({n_splits} folds, grouped by template/category)")
    print(f"Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}  (always-local baseline: {baseline_acc:.3f})")
    print(f"ROC-AUC:  {mean_auc:.3f} +/- {std_auc:.3f}")
    print("=" * 60)

    final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    final_clf.fit(X, y)

    Path("gateway").mkdir(exist_ok=True)
    joblib.dump(final_clf, MODEL_OUT)
    joblib.dump(vectorizer, VECT_OUT)
    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_training_examples": len(df),
        "cv_folds": n_splits,
        "cv_accuracy_mean": mean_acc,
        "cv_accuracy_std": std_acc,
        "cv_auc_mean": float(mean_auc),
        "cv_auc_std": float(std_auc),
        "always_local_baseline_accuracy": float(baseline_acc),
        "features": ["tfidf_char_2to4gram", "prompt_length", "digit_count",
                     "operator_count", "has_question_mark", "word_count",
                     "category_onehot"],
        "fold_details": fold_results,
    }
    json.dump(meta, open(META_OUT, "w", encoding="utf-8"), indent=2)
    print(f"Saved model to {MODEL_OUT}, vectorizer to {VECT_OUT}, meta to {META_OUT}")

if __name__ == "__main__":
    main()