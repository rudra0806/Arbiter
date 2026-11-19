"""
Evaluate the trained ML router (v2) on the dev split, offline, using saved
results — same comparison style as simulate_router.py, without needing to
touch that file.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.router_ml import route_ml

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
API_RESULTS_PATH = "results/api_groq_openai-gpt-oss-120b_20261114_130009.json"

def load(path):
    return json.load(open(path, encoding="utf-8"))

def main(threshold=0.5, split="dev"):
    tasks = {t["id"]: t for t in load(TASKS_PATH) if not t.get("invalid") and t.get("split") == split}
    local = {r["id"]: r for r in load(LOCAL_RESULTS_PATH)}
    api = {r["id"]: r for r in load(API_RESULTS_PATH)}

    n = correct = local_sent = api_cost = total_latency = 0
    for tid, t in tasks.items():
        lrec, arec = local.get(tid), api.get(tid)
        if not lrec or not arec:
            continue
        decision = route_ml(t["prompt"], threshold=threshold)
        n += 1
        if decision["target"] == "local":
            local_sent += 1
            correct += 1 if lrec.get("correct") else 0
            total_latency += lrec.get("latency_s") or 0
        else:
            correct += 1 if arec.get("correct") else 0
            total_latency += arec.get("latency_s") or 0
            api_cost += arec.get("est_cost_usd") or 0

    print(f"Split: {split}  Threshold: {threshold}  Tasks: {n}")
    print(f"Accuracy: {correct/n:.3f}   %Local: {local_sent/n:.3f}   "
          f"AvgLatency: {total_latency/n:.2f}s   APICost: ${api_cost:.6f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--split", default="dev")
    args = p.parse_args()
    main(args.threshold, args.split)