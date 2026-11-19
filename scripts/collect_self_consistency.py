"""
For each dev/pool task, call the LOCAL model 3 times at temperature 0.7
(not 0, so answers can vary) and record how often the 3 replies agree.
This agreement signal is a new feature: self-consistency.
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.models import LocalModel as _LocalModel
from benchmark.checkers import check

TASKS_PATH = "benchmark/tasks.json"

N_SAMPLES = 3

class LocalModelSampled(_LocalModel):
    """Same as LocalModel, but overrides temperature for self-consistency sampling.
    Does not modify gateway/models.py; the main benchmark path is untouched."""
    def __call__(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": self.num_predict,
                "num_ctx": self.NUM_CTX,
            },
        }
        from gateway.models import _post_with_retry
        import requests
        try:
            resp = _post_with_retry(
                url=self.OLLAMA_URL,
                headers={},
                payload=payload,
                timeout=self.TIMEOUT,
            )
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            return {"text": text, "latency_s": 0, "tokens_in": 0, "tokens_out": 0,
                    "truncated": data.get("done_reason") == "length"}
        except RuntimeError as exc:
            raise

def normalize_for_agreement(text):
    # Loose normalization just to compare the 3 replies to each other,
    # not to grade correctness (that still uses the real checker).
    return " ".join(text.lower().split())[:200]

def main():
    import sys
    split_filter = sys.argv[1] if len(sys.argv) > 1 else "dev,pool"
    allowed_splits = split_filter.split(",")
    out_path = f"results/self_consistency_{split_filter.replace(',', '_')}.json"
    tasks = json.load(open(TASKS_PATH, encoding="utf-8"))

    import sys

    split_filter = sys.argv[1] if len(sys.argv) > 1 else "dev,pool"
    allowed_splits = split_filter.split(",")
    eligible = [t for t in tasks if not t.get("invalid") and t.get("split") in allowed_splits]
    print(f"Collecting self-consistency for {len(eligible)} tasks, {N_SAMPLES} samples each "
          f"({len(eligible) * N_SAMPLES} total local calls)...")

    model = LocalModelSampled()  # override temp for this run

    results = []
    done_ids = set()
    if Path(out_path).exists():
        prev = json.load(open(out_path, encoding="utf-8"))
        results = prev
        done_ids = {r["id"] for r in prev}
        print(f"Resuming: {len(done_ids)} tasks already have samples.")

    for i, t in enumerate(eligible, 1):
        if t["id"] in done_ids:
            continue
        replies = []
        for s in range(N_SAMPLES):
            try:
                r = model(t["prompt"])
                replies.append(r["text"])
            except Exception as e:
                replies.append(None)
        normed = [normalize_for_agreement(r) for r in replies if r is not None]
        agree_count = Counter(normed).most_common(1)[0][1] if normed else 0
        agreement_ratio = agree_count / N_SAMPLES
        correct_each = [check(t, r) if r is not None else False for r in replies]
        results.append({
            "id": t["id"],
            "replies": replies,
            "agreement_ratio": agreement_ratio,
            "correct_each": correct_each,
            "majority_correct": sum(correct_each) > N_SAMPLES / 2,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(eligible)} done, agreement={agreement_ratio:.2f}")
            json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} records to {out_path}")

if __name__ == "__main__":
    main()