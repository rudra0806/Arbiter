"""
benchmark/simulate_router.py
----------------------------
Offline evaluation: compare routing strategies WITHOUT calling any model.
Looks up pre-saved result JSON files and grades the router against them.

Usage
-----
    python -m benchmark.simulate_router \\
        --local  results/local_<tag>.json \\
        --api    results/api_<tag>.json   \\
        [--split dev]                      (default; safe)
        [--split test|final --final]       (test/final set; prints one-time warning)

Output
------
  - Comparison table: always-local, always-api, router, oracle, random-router
  - Per-category accuracy for the router
  - Dev-split only: tasks routed local that local got wrong
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from gateway.router import route

TASKS_PATH = Path(__file__).parent / "tasks.json"

# Cost variables from .env (USD per million tokens).
# Used to estimate what the router would have spent vs always-API.
_COST_PER_M_IN  = float(os.getenv("COST_PER_M_IN",  0.0))
_COST_PER_M_OUT = float(os.getenv("COST_PER_M_OUT", 0.0))

RANDOM_TRIALS = 200   # number of random-router trials for the control row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(path: str) -> dict:
    """Load a result JSON file and return a dict keyed by task id."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in records}


def _est_cost(result: dict) -> float:
    """Estimate USD cost from a single result record."""
    tin  = result.get("tokens_in",  0) or 0
    tout = result.get("tokens_out", 0) or 0
    return (tin / 1_000_000 * _COST_PER_M_IN +
            tout / 1_000_000 * _COST_PER_M_OUT)


def _strategy_stats(tasks: list, local_map: dict, api_map: dict,
                    decisions: list[str]) -> dict:
    """
    Compute stats for a routing strategy given a list of 'local'/'api' decisions
    (one per task, in the same order as `tasks`).

    Returns dict with keys: correct, total, sent_local, latency_s, api_cost_usd.
    """
    correct = 0
    latency = 0.0
    api_cost = 0.0
    sent_local = 0

    for task, target in zip(tasks, decisions):
        tid = task["id"]
        res = (local_map if target == "local" else api_map).get(tid, {})
        if res.get("correct") is True:
            correct += 1
        latency += res.get("latency_s", 0) or 0
        if target == "api":
            api_cost += _est_cost(res)
        else:
            sent_local += 1

    return {
        "correct":      correct,
        "total":        len(tasks),
        "sent_local":   sent_local,
        "latency_s":    latency / len(tasks) if tasks else 0,
        "api_cost_usd": api_cost,
    }


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  -  "
    return f"{100 * num / denom:.1f}%"


def _cost_saved(router_cost: float, always_api_cost: float) -> str:
    saved = always_api_cost - router_cost
    if always_api_cost == 0:
        return "n/a"
    return f"${saved:.6f} ({100*saved/always_api_cost:.1f}%)"


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(local_path: str, api_path: str, split: str, final: bool) -> None:

    # -- Guard: test or final split requires --final flag ------------------
    if split in ("test", "final") and not final:
        print(f"ERROR: To run on the {split} split you must pass both "
              f"--split {split} AND --final. Aborting.")
        sys.exit(1)

    if split in ("test", "final") and final:
        print(
            "\n⚠️  WARNING: You are evaluating on the HELD-OUT TEST SET.\n"
            "   This should only be done once, after all rule development is complete.\n"
        )

    # -- Load data ---------------------------------------------------------
    tasks     = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    raw_split_tasks = [t for t in tasks if t.get("split") == split]

    if not raw_split_tasks:
        print(f"No tasks found for split='{split}'. Check tasks.json.")
        sys.exit(1)

    # Exclude tasks marked invalid
    split_tasks = [t for t in raw_split_tasks if not t.get("invalid")]
    skipped_invalid = len(raw_split_tasks) - len(split_tasks)

    local_map = _load_results(local_path)
    api_map   = _load_results(api_path)

    n = len(split_tasks)
    print(f"\nSplit: {split}  |  Tasks: {n}")
    print(f"Skipped (invalid): {skipped_invalid}")

    # -- Router decisions --------------------------------------------------
    router_decisions = []
    router_reasons   = []
    for t in split_tasks:
        d = route(t["prompt"])
        router_decisions.append(d["target"])
        router_reasons.append(d["reason"])

    # -- Strategy stats ----------------------------------------------------
    always_local_dec = ["local"] * n
    always_api_dec   = ["api"]   * n
    oracle_dec = [
        "local" if local_map.get(t["id"], {}).get("correct") is True else "api"
        for t in split_tasks
    ]

    # Fraction sent local by the router → used to match the random baseline
    frac_local = router_decisions.count("local") / n

    # Random router: average over RANDOM_TRIALS random assignments with the
    # same fraction sent local, as a control showing what chance would do.
    rng = random.Random(42)   # fixed seed for reproducibility
    random_total_correct = 0
    random_total_cost    = 0.0
    random_total_latency = 0.0
    for _ in range(RANDOM_TRIALS):
        # Shuffle local/api labels with the same ratio as our router
        n_local  = round(frac_local * n)
        n_api    = n - n_local
        rand_dec = ["local"] * n_local + ["api"] * n_api
        rng.shuffle(rand_dec)
        s = _strategy_stats(split_tasks, local_map, api_map, rand_dec)
        random_total_correct += s["correct"]
        random_total_cost    += s["api_cost_usd"]
        random_total_latency += s["latency_s"]

    random_stats = {
        "correct":      random_total_correct / RANDOM_TRIALS,
        "total":        n,
        "sent_local":   round(frac_local * n),
        "latency_s":    random_total_latency / RANDOM_TRIALS,
        "api_cost_usd": random_total_cost    / RANDOM_TRIALS,
    }

    al_stats = _strategy_stats(split_tasks, local_map, api_map, always_local_dec)
    aa_stats = _strategy_stats(split_tasks, local_map, api_map, always_api_dec)
    ro_stats = _strategy_stats(split_tasks, local_map, api_map, router_decisions)
    or_stats = _strategy_stats(split_tasks, local_map, api_map, oracle_dec)

    aa_cost = aa_stats["api_cost_usd"]

    # -- Print comparison table -------------------------------------------
    hdr  = f"  {'Strategy':<20} {'Accuracy':>10} {'%Local':>8} {'AvgLat':>9} {'APICost':>12} {'CostSaved':>16}"
    sep  = "  " + "-" * (len(hdr) - 2)

    def row(name: str, s: dict, is_float_correct: bool = False) -> str:
        acc = (f"{s['correct']:.1f}/{s['total']}" if is_float_correct
               else f"{s['correct']}/{s['total']}")
        pct_acc  = (_pct(int(s['correct']), s['total']) if not is_float_correct
                    else f"{100*s['correct']/s['total']:.1f}%")
        pct_loc  = _pct(s["sent_local"], s["total"])
        lat      = f"{s['latency_s']:.2f}s"
        cost     = f"${s['api_cost_usd']:.6f}"
        saved    = _cost_saved(s["api_cost_usd"], aa_cost)
        return f"  {name:<20} {pct_acc:>10} {pct_loc:>8} {lat:>9} {cost:>12} {saved:>16}"

    print(f"\n{'='*78}")
    print(f"  ROUTING STRATEGY COMPARISON  —  split: {split}")
    print(f"  Skipped (invalid): {skipped_invalid}")
    print(f"{'='*78}")
    print(hdr)
    print(sep)
    print(row("always-local",   al_stats))
    print(row("always-api",     aa_stats))
    print(row("router",         ro_stats))
    print(row("oracle",         or_stats))
    print(row(f"random({RANDOM_TRIALS} trials)", random_stats, is_float_correct=True))
    print(f"{'='*78}\n")

    # -- Per-category accuracy for the router -----------------------------
    from collections import defaultdict
    cat_stats: dict[str, list] = defaultdict(lambda: [0, 0])  # [correct, total]
    for task, target in zip(split_tasks, router_decisions):
        cat = task.get("category", "?")
        res = (local_map if target == "local" else api_map).get(task["id"], {})
        cat_stats[cat][1] += 1
        if res.get("correct") is True:
            cat_stats[cat][0] += 1

    print("  Router accuracy per category:")
    print(f"  {'Category':<14} {'Correct':>8} {'Total':>7} {'Acc':>7} {'Via':>8}")
    print("  " + "-" * 44)
    for cat in sorted(cat_stats):
        c, t = cat_stats[cat]
        # Count how many in this category went to API
        via_api = sum(1 for task, dec in zip(split_tasks, router_decisions)
                      if task["category"] == cat and dec == "api")
        print(f"  {cat:<14} {c:>8} {t:>7} {_pct(c, t):>7} {via_api:>5}→api")
    print()

    # -- Dev split: tasks routed local but local got wrong ----------------
    if split == "dev":
        mistakes = [
            (task, reason)
            for task, target, reason in zip(split_tasks, router_decisions, router_reasons)
            if target == "local"
            and local_map.get(task["id"], {}).get("correct") is False
            and local_map.get(task["id"], {}).get("error") is None
        ]
        print("  Tasks routed → local but local got WRONG:")
        if not mistakes:
            print("    (none)")
        else:
            print(f"  {'id':>4}  {'category':<12} {'diff':<8}  reason")
            print("  " + "-" * 60)
            for task, reason in mistakes:
                print(f"  {task['id']:>4}  {task['category']:<12} {task['difficulty']:<8}  {reason}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline router simulation — no model calls."
    )
    parser.add_argument("--local", required=True,
                        help="Path to the local model results JSON file.")
    parser.add_argument("--api",   required=True,
                        help="Path to the API model results JSON file.")
    parser.add_argument("--split", choices=["dev", "test", "final"], default="dev",
                        help="Which split to evaluate (default: dev).")
    parser.add_argument("--final", action="store_true",
                        help="Required together with --split test or --split final to unlock test-set evaluation.")
    args = parser.parse_args()
    simulate(args.local, args.api, args.split, args.final)


if __name__ == "__main__":
    main()
