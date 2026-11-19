"""
benchmark/run_eval.py
---------------------
CLI benchmark runner.

Usage:
    python -m benchmark.run_eval --model local|api [--limit N] [--no-cache]

Output:
    results/<model>_<timestamp>.json   – full per-task records
    Console summary table              – accuracy, latency, cost
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path

# Allow running as  python -m benchmark.run_eval  from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from gateway.models import LocalModel, APIModel
from benchmark.checkers import check as grade

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASKS_PATH  = Path(__file__).parent / "tasks.json"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR   = PROJECT_ROOT / ".cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters (/, \\, :, *, ?, ", <, >, |) with '-'."""
    return re.sub(r'[/\\:*?"<>|\x00-\x1f]', '-', name)


def _cache_key(model_label: str, prompt: str) -> str:
    """SHA-256 of model label + prompt → used as cache filename."""
    raw = f"{model_label}||{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(key: str, data: dict) -> None:
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  -  "
    return f"{100 * num / denom:.1f}%"


def _print_summary(records: list, model_label: str, skipped_invalid: int = 0) -> None:
    """Print an overall + per-category + per-difficulty accuracy table."""
    # Exclude any invalid tasks if present in records (e.g. from existing result files)
    try:
        all_tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        invalid_ids = {t["id"] for t in all_tasks if t.get("invalid")}
    except Exception:
        invalid_ids = set()

    prev_len = len(records)
    records = [r for r in records if r.get("id") not in invalid_ids and not r.get("invalid")]
    skipped_count = skipped_invalid + (prev_len - len(records))

    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    errors  = sum(1 for r in records if r.get("error"))
    truncated = sum(1 for r in records if r.get("truncated") is True)
    total_latency = sum(r.get("latency_s") or 0 for r in records)
    total_cost    = sum(r.get("est_cost_usd", 0) for r in records)
    avg_lat = total_latency / total if total else 0

    print("\n" + "=" * 60)
    print(f"  RESULTS  –  model: {model_label}")
    print("=" * 60)
    print(f"  Tasks run : {total}  |  Correct: {correct}  |  Errors: {errors}")
    print(f"  Truncated : {truncated}")
    print(f"  Skipped (invalid): {skipped_count}")
    print(f"  Accuracy  : {_pct(correct, total - errors)}")
    print(f"  Avg latency : {avg_lat:.2f}s")
    print(f"  Est. total cost : ${total_cost:.6f}")

    # --- Per category ---
    categories: dict[str, dict] = {}
    for r in records:
        cat = r.get("category", "unknown")
        bucket = categories.setdefault(cat, {"correct": 0, "total": 0})
        if r.get("error") is None:
            bucket["total"] += 1
            if r.get("correct"):
                bucket["correct"] += 1

    print("\n  By category:")
    print(f"  {'Category':<18} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print("  " + "-" * 42)
    for cat, b in sorted(categories.items()):
        print(f"  {cat:<18} {b['correct']:>8} {b['total']:>7} {_pct(b['correct'], b['total']):>7}")

    # --- Per difficulty ---
    difficulties: dict[str, dict] = {}
    for r in records:
        diff = r.get("difficulty", "unknown")
        bucket = difficulties.setdefault(diff, {"correct": 0, "total": 0})
        if r.get("error") is None:
            bucket["total"] += 1
            if r.get("correct"):
                bucket["correct"] += 1

    print("\n  By difficulty:")
    print(f"  {'Difficulty':<14} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print("  " + "-" * 42)
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            b = difficulties[diff]
            print(f"  {diff:<14} {b['correct']:>8} {b['total']:>7} {_pct(b['correct'], b['total']):>7}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(model_label: str, limit: int | None, use_cache: bool) -> None:
    # Load model
    if model_label == "local":
        model = LocalModel()
        # Include the sanitized model name in output filenames
        local_model = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
        model_name_tag = _sanitize_filename(local_model)
        file_label = f"local_{model_name_tag}"
    else:
        model = APIModel()
        provider = _sanitize_filename(os.getenv("PROVIDER", "gemini"))
        api_model = os.getenv("API_MODEL", "unknown")
        model_name_tag = _sanitize_filename(api_model)
        file_label = f"api_{provider}_{model_name_tag}"

    # Load tasks, skipping any task marked invalid
    raw_tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if limit:
        raw_tasks = raw_tasks[:limit]

    tasks = [t for t in raw_tasks if not t.get("invalid")]
    skipped_invalid = len(raw_tasks) - len(tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{file_label}_{timestamp}.json"

    records = []
    total = len(tasks)

    print(f"\nRunning {total} tasks with model '{model_label}' …\n")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", i)
        prompt  = task["prompt"]
        print(f"  [{i:>3}/{total}] Task {task_id:>3} ({task['category']}/{task['difficulty']}) … ", end="", flush=True)

        record = {
            "id":         task_id,
            "category":   task.get("category", ""),
            "difficulty": task.get("difficulty", ""),
            "prompt":     prompt,
            "reply":      None,
            "correct":    None,
            "latency_s":  None,
            "tokens_in":  None,
            "tokens_out": None,
            "est_cost_usd": 0.0,
            "truncated":  False,
            "error":      None,
        }

        try:
            # Try cache first
            cache_key = _cache_key(file_label, prompt)
            cached = _load_cache(cache_key) if use_cache else None

            if cached:
                result = cached
                print("(cached) ", end="", flush=True)
            else:
                result = model(prompt)
                if use_cache:
                    _save_cache(cache_key, result)

            record["reply"]       = result["text"]
            record["latency_s"]   = result.get("latency_s")
            record["tokens_in"]   = result.get("tokens_in")
            record["tokens_out"]  = result.get("tokens_out")
            record["est_cost_usd"]= result.get("est_cost_usd", 0.0)
            record["truncated"]   = bool(result.get("truncated", False))

            correct = grade(task, result["text"])
            record["correct"] = correct
            print("✓" if correct else "✗")

        except Exception as exc:
            record["error"] = str(exc)
            # Full traceback to stderr so it doesn't clutter the progress line
            print(f"ERROR – {exc}")
            traceback.print_exc(file=sys.stderr)

        records.append(record)

        # Save incrementally so a crash doesn't lose everything
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    _print_summary(records, file_label, skipped_invalid=skipped_invalid)
    print(f"Full results saved to: {out_path}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM benchmark evaluation."
    )
    parser.add_argument(
        "--model", choices=["local", "api"], required=True,
        help="Which model to evaluate: 'local' (Ollama) or 'api' (hosted)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N tasks (useful for a quick smoke-test)."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore and overwrite any cached replies."
    )
    args = parser.parse_args()
    run_eval(args.model, args.limit, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
