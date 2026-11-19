"""
Router v3: combines v2's TF-IDF/numeric text features with a new
self-consistency feature (self_agreement: 0-3, how many of 3
repeated local-model samples at temperature 0.7 were graded correct).
Trains and validates on dev+pool only, same grouped CV as v2.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
SELF_CONSISTENCY_PATH = "results/self_consistency.json"
MODEL_OUT = "gateway/router_model_v3.pkl"
VECT_OUT = "gateway/router_vectorizer_v3.pkl"
META_OUT = "gateway/router_ml_meta_v3.json"

import re

def extract_candidate_answer(text):
    """Pull a rough 'final answer' out of a reply, with no access to the
    true key. Strips markdown/LaTeX noise, then returns the LAST number
    found, or failing that, the last short word (<=12 chars) as a fallback.
    This is intentionally simple and imperfect — same limitation a real
    router would face."""
    if not text:
        return None
    cleaned = text.replace("**", "").replace("*", "").replace("`", "")
    cleaned = re.sub(r"\\[\(\)\[\]]", "", cleaned)
    cleaned = re.sub(r"\{,\}", ",", cleaned)
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", cleaned)
    if numbers:
        return numbers[-1].replace(",", "")
    words = re.findall(r"[A-Za-z]+", cleaned)
    short_words = [w.lower() for w in words if len(w) <= 12]
    return short_words[-1] if short_words else None

def compute_self_agreement(replies, n_samples):
    subset = replies[:n_samples]
    candidates = [extract_candidate_answer(r) for r in subset]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0.0
    from collections import Counter
    most_common_count = Counter(candidates).most_common(1)[0][1]
    return most_common_count / len(subset)

def load_data(n_samples=3):
    tasks = json.load(open(TASKS_PATH, encoding="utf-8"))
    local = json.load(open(LOCAL_RESULTS_PATH, encoding="utf-8"))
    sc = json.load(open(SELF_CONSISTENCY_PATH, encoding="utf-8"))
    local_by_id = {r["id"]: r for r in local}
    sc_by_id = {r["id"]: r for r in sc}

    rows = []
    for t in tasks:
        if t.get("invalid"):
            continue
        if t.get("split") not in ("dev", "pool"):
            continue
        rec = local_by_id.get(t["id"])
        screc = sc_by_id.get(t["id"])
        if rec is None or rec.get("error") or screc is None:
            continue
        self_agreement = compute_self_agreement(screc["replies"], n_samples)
        rows.append({
            "id": t["id"],
            "prompt": t["prompt"],
            "label": 1 if rec.get("correct") else 0,
            "group": t.get("template") or t.get("category") or "unknown",
            "self_agreement": self_agreement,
        })
    return pd.DataFrame(rows)


def numeric_feats(df, include_sc):
    feats = []
    for i, p in enumerate(df["prompt"]):
        base = [len(p), sum(c.isdigit() for c in p),
                sum(p.count(c) for c in "+-*/%"),
                1 if "?" in p else 0, len(p.split())]
        if include_sc:
            base.append(df["self_agreement"].iloc[i])
        feats.append(base)
    return np.array(feats, dtype=float)

def run_cv(X, y, groups, label):
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, aucs = [], []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        prob = clf.predict_proba(X[va])[:, 1]
        acc = accuracy_score(y[va], pred)
        try:
            auc = roc_auc_score(y[va], prob)
        except ValueError:
            auc = float("nan")
        accs.append(acc); aucs.append(auc)
        print(f"  [{label}] Fold {fold}: acc={acc:.3f} auc={auc:.3f}")
    print(f"  [{label}] MEAN: acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
          f"auc={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}")
    return np.mean(accs), np.nanmean(aucs)

def main():
    for n in (1, 2, 3):
        print(f"\n{'='*60}\nSELF-AGREEMENT USING {n} SAMPLE(S)\n{'='*60}")
        df = load_data(n_samples=n)
        baseline = df["label"].mean()
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
        X_text = vectorizer.fit_transform(df["prompt"])
        y = df["label"].values
        groups = df["group"].values

        X_v3 = hstack([X_text, csr_matrix(numeric_feats(df, include_sc=True))]).tocsr()
        print(f"[text + self_agreement({n})]")
        acc_v3, auc_v3 = run_cv(X_v3, y, groups, f"v3-{n}samples")

        X_sc_only = csr_matrix(df[["self_agreement"]].values.astype(float))
        print(f"[self_agreement({n}) only]")
        run_cv(X_sc_only, y, groups, f"SConly-{n}samples")

if __name__ == "__main__":
    main()