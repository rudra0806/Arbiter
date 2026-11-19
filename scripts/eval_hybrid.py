"""
Hybrid: use the verifier's agree/disagree signal as the PRIMARY decision.
When the samples disagree, instead of automatically escalating to API,
ask whether including v2's text-based classifier prediction for those
disagreement cases would have been more accurate than blanket escalation.
This only checks disagreement cases - agreement cases are handled exactly
as the plain verifier does (trust local, never touch API).
"""
import json
import re
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
API_RESULTS_PATH = "results/api_groq_openai-gpt-oss-120b_20261114_130009.json"
SELF_CONSISTENCY_PATHS = {
    "dev": "results/self_consistency.json",
    "pool": "results/self_consistency.json",
    "test": "results/self_consistency_test.json",
}
MODEL_PATH = "gateway/router_model.pkl"
VECT_PATH = "gateway/router_vectorizer.pkl"

def extract_candidate_answer(text):
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

def numeric_feats_single(prompt):
    length = len(prompt)
    digits = sum(c.isdigit() for c in prompt)
    ops = sum(prompt.count(c) for c in "+-*/%")
    has_q = 1 if "?" in prompt else 0
    words = len(prompt.split())
    return np.array([[length, digits, ops, has_q, words]], dtype=float)

def main(split="dev", n_samples=3, clf_threshold=0.5):
    tasks = {t["id"]: t for t in json.load(open(TASKS_PATH, encoding="utf-8"))
             if not t.get("invalid") and t.get("split") == split}
    local = {r["id"]: r for r in json.load(open(LOCAL_RESULTS_PATH, encoding="utf-8"))}
    api = {r["id"]: r for r in json.load(open(API_RESULTS_PATH, encoding="utf-8"))}
    sc_path = SELF_CONSISTENCY_PATHS.get(split, "results/self_consistency.json")
    sc = {r["id"]: r for r in json.load(open(sc_path, encoding="utf-8"))}

    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)

    n = correct = sent_local = api_cost = total_latency = 0
    disagree_cases = 0
    disagree_clf_wouldve_been_right = 0
    disagree_escalate_was_right = 0

    for tid, t in tasks.items():
        lrec, arec, screc = local.get(tid), api.get(tid), sc.get(tid)
        if not lrec or not arec or not screc:
            continue
        n += 1
        candidates = [extract_candidate_answer(r) for r in screc["replies"][:n_samples]]
        candidates = [c for c in candidates if c is not None]
        agree = len(candidates) >= 2 and len(set(candidates)) == 1

        if agree:
            sent_local += 1
            correct += 1 if lrec.get("correct") else 0
            total_latency += (lrec.get("latency_s") or 0) * n_samples
        else:
            disagree_cases += 1
            # What the plain verifier does: escalate.
            escalate_correct = arec.get("correct")
            if escalate_correct:
                disagree_escalate_was_right += 1

            # What the classifier would have predicted for this disagreement case:
            X_text = vectorizer.transform([t["prompt"]])
            X_num = csr_matrix(numeric_feats_single(t["prompt"]))
            X = hstack([X_text, X_num]).tocsr()
            prob_local_correct = clf.predict_proba(X)[0, 1]
            clf_says_local = prob_local_correct >= clf_threshold

            if clf_says_local:
                # Hybrid would trust local here instead of escalating.
                final_correct = lrec.get("correct")
                sent_local += 1
                total_latency += (lrec.get("latency_s") or 0) * n_samples
            else:
                final_correct = escalate_correct
                total_latency += (lrec.get("latency_s") or 0) * n_samples + (arec.get("latency_s") or 0)
                api_cost += arec.get("est_cost_usd") or 0

            correct += 1 if final_correct else 0
            if final_correct and clf_says_local and lrec.get("correct"):
                disagree_clf_wouldve_been_right += 1

    print(f"Split: {split}  Samples/task: {n_samples}  Tasks: {n}")
    print(f"Disagreement cases: {disagree_cases}")
    print(f"  Of those, plain-escalate was correct: {disagree_escalate_was_right}/{disagree_cases}")
    print(f"  Of those, classifier would trust local AND be right: {disagree_clf_wouldve_been_right}/{disagree_cases}")
    print(f"HYBRID Accuracy: {correct/n:.3f}   %Local: {sent_local/n:.1%}   "
          f"APICost: ${api_cost:.6f}   AvgLatency: {total_latency/n:.2f}s")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--clf-threshold", type=float, default=0.5)
    args = p.parse_args()
    main(args.split, args.n_samples, args.clf_threshold)