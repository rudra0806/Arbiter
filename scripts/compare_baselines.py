"""
Compare the trained ML router against simple baselines using the SAME
grouped cross-validation folds, so the comparison is fair. This tells us
whether the ML model is adding real value beyond a hand-written rule.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"

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
            "difficulty": t.get("difficulty"),
        })
    return pd.DataFrame(rows)

def numeric_feats(prompts):
    feats = []
    for p in prompts:
        feats.append([len(p), sum(c.isdigit() for c in p),
                      sum(p.count(c) for c in "+-*/%"),
                      1 if "?" in p else 0, len(p.split())])
    return np.array(feats, dtype=float)

def main():
    df = load_data()
    n_groups = df["group"].nunique()
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # ML features (text-only version, no category, matches the better run)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X_text = vectorizer.fit_transform(df["prompt"])
    X_num = numeric_feats(df["prompt"].tolist())
    X = hstack([X_text, csr_matrix(X_num)]).tocsr()
    y = df["label"].values
    groups = df["group"].values

    # Difficulty rule: predict "1" (local correct) only for "easy",
    # predict "0" (local will fail) for "medium" and "hard".
    diff_pred = (df["difficulty"] == "easy").astype(int).values
    # "Probability" for AUC purposes: easy=1.0, medium=0.5, hard=0.0
    diff_score = df["difficulty"].map({"easy": 1.0, "medium": 0.5, "hard": 0.0}).values

    always_local_acc = y.mean()

    ml_accs, ml_aucs = [], []
    rule_accs, rule_aucs = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        ml_pred = clf.predict(X[val_idx])
        ml_prob = clf.predict_proba(X[val_idx])[:, 1]
        ml_accs.append(accuracy_score(y[val_idx], ml_pred))
        try:
            ml_aucs.append(roc_auc_score(y[val_idx], ml_prob))
        except ValueError:
            ml_aucs.append(float("nan"))

        rule_pred_fold = diff_pred[val_idx]
        rule_score_fold = diff_score[val_idx]
        rule_accs.append(accuracy_score(y[val_idx], rule_pred_fold))
        try:
            rule_aucs.append(roc_auc_score(y[val_idx], rule_score_fold))
        except ValueError:
            rule_aucs.append(float("nan"))

        print(f"Fold {fold}: ML acc={ml_accs[-1]:.3f} auc={ml_aucs[-1]:.3f}  |  "
              f"DifficultyRule acc={rule_accs[-1]:.3f} auc={rule_aucs[-1]:.3f}")

    print("=" * 70)
    print(f"Always-local baseline accuracy: {always_local_acc:.3f}")
    print(f"ML model:        acc={np.mean(ml_accs):.3f} +/- {np.std(ml_accs):.3f}   "
          f"auc={np.nanmean(ml_aucs):.3f} +/- {np.nanstd(ml_aucs):.3f}")
    print(f"DifficultyRule:  acc={np.mean(rule_accs):.3f} +/- {np.std(rule_accs):.3f}   "
          f"auc={np.nanmean(rule_aucs):.3f} +/- {np.nanstd(rule_aucs):.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()