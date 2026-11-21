import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.ask import LocalModel, is_grounded, extract_candidate_answer, N_SAMPLES

prompt = "A bank account starts at $50. A withdrawal of $80 is made, then a deposit of $25, then another withdrawal of $10. What is the final balance?"

for attempt in range(5):
    print(f"=== Attempt {attempt+1} ===")
    local = LocalModel()
    replies = [local(prompt)["text"] for _ in range(N_SAMPLES)]
    grounded_flags = [is_grounded(prompt, r) for r in replies]
    grounded_replies = [r for r, g in zip(replies, grounded_flags) if g]
    candidates = [extract_candidate_answer(r) for r in grounded_replies]
    candidates = [c for c in candidates if c is not None]
    agree = len(candidates) >= 2 and len(set(candidates)) == 1
    print(f"grounded_flags: {grounded_flags}")
    print(f"num_grounded: {len(grounded_replies)}")
    print(f"candidates: {candidates}")
    print(f"would_agree: {agree}")
    print()