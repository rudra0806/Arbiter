"""
True verifier approach: call local 2x. If they agree (via extract_candidate_
answer), trust that answer and never call the API. If they disagree, escalate
to API. No classifier needed for this — agreement itself IS the decision.
Compares this against v2/v3 classifiers and the plain always-local/always-api
baselines, all on the dev split, all offline using saved data.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.answer_matching import extract_candidate_answer, is_grounded, extract_numbers

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
API_RESULTS_PATH = "results/api_groq_openai-gpt-oss-120b_20261114_130009.json"
SELF_CONSISTENCY_PATHS = {
    "dev": "results/self_consistency.json",
    "pool": "results/self_consistency.json",
    "test": "results/self_consistency_test.json",
    "final": "results/self_consistency_final.json",
}

def main(split="dev", n_samples=2):
    tasks = {t["id"]: t for t in json.load(open(TASKS_PATH, encoding="utf-8"))
             if not t.get("invalid") and t.get("split") == split}
    local = {r["id"]: r for r in json.load(open(LOCAL_RESULTS_PATH, encoding="utf-8"))}
    api = {r["id"]: r for r in json.load(open(API_RESULTS_PATH, encoding="utf-8"))}
    sc_path = SELF_CONSISTENCY_PATHS.get(split, "results/self_consistency.json")
    sc = {r["id"]: r for r in json.load(open(sc_path, encoding="utf-8"))}
    n = correct = sent_local = api_cost = total_latency = agreements = 0
    for tid, t in tasks.items():
        lrec, arec, screc = local.get(tid), api.get(tid), sc.get(tid)
        if not lrec or not arec or not screc:
            continue
        n += 1
        candidates = [extract_candidate_answer(r) for r in screc["replies"][:n_samples]]
        candidates = [c for c in candidates if c is not None]
        agree = len(candidates) >= 2 and len(set(candidates)) == 1

        if agree:
            agreements += 1
            sent_local += 1
            correct += 1 if lrec.get("correct") else 0
            total_latency += (lrec.get("latency_s") or 0) * n_samples
        else:
            correct += 1 if arec.get("correct") else 0
            total_latency += (lrec.get("latency_s") or 0) * n_samples + (arec.get("latency_s") or 0)
            api_cost += arec.get("est_cost_usd") or 0

    print(f"Split: {split}  Samples/task: {n_samples}  Tasks: {n}")
    print(f"Agreed (sent local): {agreements}/{n} ({agreements/n:.1%})")
    print(f"Accuracy: {correct/n:.3f}   AvgLatency: {total_latency/n:.2f}s   APICost: ${api_cost:.6f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--n-samples", type=int, default=2)
    args = p.parse_args()
    main(args.split, args.n_samples)