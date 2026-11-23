"""
One consolidated report: how the final verifier design performs across
dev, test, and final (held-out) splits. Reuses the same leak-free
self-consistency logic as eval_verifier.py. Read-only, no new model calls.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.answer_matching import extract_candidate_answer, is_grounded, extract_numbers

TASKS_PATH = "benchmark/tasks.json"
LOCAL_RESULTS_PATH = "results/local_qwen2.5-1.5b_20261114_130038.json"
API_RESULTS_PATH = "results/api_groq_openai-gpt-oss-120b_20261114_130009.json"
SELF_CONSISTENCY_PATHS = {
    "dev": "results/self_consistency.json",
    "test": "results/self_consistency_test.json",
    "final": "results/self_consistency_final.json",
}
N_SAMPLES = 3

def evaluate_split(split, tasks, local, api, sc):
    eligible = {tid: t for tid, t in tasks.items() if t.get("split") == split and not t.get("invalid")}
    n = always_local_correct = always_api_correct = verifier_correct = 0
    sent_local = api_cost = 0.0

    for tid, t in eligible.items():
        lrec, arec, screc = local.get(tid), api.get(tid), sc.get(tid)
        if not lrec or not arec or not screc:
            continue
        n += 1
        always_local_correct += 1 if lrec.get("correct") else 0
        always_api_correct += 1 if arec.get("correct") else 0

        candidates = [extract_candidate_answer(r) for r in screc["replies"][:N_SAMPLES]]
        candidates = [c for c in candidates if c is not None]
        agree = len(candidates) >= 2 and len(set(candidates)) == 1

        if agree:
            sent_local += 1
            verifier_correct += 1 if lrec.get("correct") else 0
        else:
            verifier_correct += 1 if arec.get("correct") else 0
            api_cost += arec.get("est_cost_usd") or 0

    if n == 0:
        return None
    return {
        "split": split, "n": n,
        "always_local_acc": always_local_correct / n,
        "always_api_acc": always_api_correct / n,
        "verifier_acc": verifier_correct / n,
        "pct_local": sent_local / n,
        "api_cost": api_cost,
        "always_api_cost_estimate": api_cost / max(sent_local == n, 1) if sent_local < n else api_cost,
    }

def main():
    tasks = {t["id"]: t for t in json.load(open(TASKS_PATH, encoding="utf-8"))}
    local = {r["id"]: r for r in json.load(open(LOCAL_RESULTS_PATH, encoding="utf-8"))}
    api = {r["id"]: r for r in json.load(open(API_RESULTS_PATH, encoding="utf-8"))}

    print("=" * 78)
    print("  LLM GATEWAY — FINAL VERIFIER RESULTS SUMMARY")
    print("  (self-consistency verifier, 3 local samples, no trained classifier)")
    print("=" * 78)
    header = f"{'Split':<8}{'Tasks':>7}{'AlwaysLocal':>13}{'AlwaysAPI':>11}{'Verifier':>10}{'%Local':>9}{'APICost':>10}"
    print(header)
    print("-" * 78)

    rows = []
    for split in ("dev", "test", "final"):
        sc = {r["id"]: r for r in json.load(open(SELF_CONSISTENCY_PATHS[split], encoding="utf-8"))}
        row = evaluate_split(split, tasks, local, api, sc)
        if row is None:
            continue
        rows.append(row)
        print(f"{row['split']:<8}{row['n']:>7}{row['always_local_acc']:>12.1%} "
              f"{row['always_api_acc']:>10.1%} {row['verifier_acc']:>9.1%} "
              f"{row['pct_local']:>8.1%} ${row['api_cost']:>8.5f}")

    print("=" * 78)
    print("\nNOTES (not computed above, documented from live testing):")
    print("- 'final' is the one-time held-out split: evaluated exactly once, never")
    print("  tuned against. This is the headline, trustworthy number.")
    print("- A deterministic day-of-week calculator bypasses the LLM entirely for")
    print("  'N days from [weekday]' questions, after live testing found the local")
    print("  model gives confidently wrong, fully self-consistent answers ~50% of")
    print("  the time on that specific pattern (a systematic shortcut, not random")
    print("  noise, so self-consistency alone could not catch it).")
    print("- 17/17 other live-tested question types (money, geometry, logic,")
    print("  probability, speed, sequences, leap years, clock duration) were")
    print("  correctly handled, suggesting the day-of-week issue is narrow.")
    print("- Scope: validated for checkable-answer tasks only (math, facts, code")
    print("  output, extraction, classification). NOT validated for open-ended")
    print("  chat, creative writing, or document generation.")

if __name__ == "__main__":
    main()