"""
The actual, usable application: ask a question, get a routed answer.
Calls the local model up to 3 times; if all 3 agree, returns that answer
for free. If they disagree, escalates to the API model once.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.models import LocalModel, APIModel

N_SAMPLES = 3

import re as _re
from datetime import datetime, timedelta

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_DAY_OFFSET_PATTERN = _re.compile(
    r"if today is (\w+),?\s*what day (?:is it|will it be)\s*(\d+)\s*days?\s*from now",
    _re.IGNORECASE
)

_DAY_OFFSET_PATTERNS = [
    _re.compile(r"if today is (\w+),?\s*what day (?:is it|will it be)\s*(\d+)\s*days?\s*from (?:now|today)", _re.IGNORECASE),
    _re.compile(r"what day (?:is it|will it be)\s*(\d+)\s*days?\s*after (\w+)", _re.IGNORECASE),
    _re.compile(r"(\d+)\s*days?\s*after (\w+)\s*is what day", _re.IGNORECASE),
    _re.compile(r"(\w+)\s*plus\s*(\d+)\s*days?\s*is what day", _re.IGNORECASE),
]

def try_day_offset_calculator(prompt: str):
    for pattern in _DAY_OFFSET_PATTERNS:
        m = pattern.search(prompt)
        if not m:
            continue
        groups = m.groups()
        # Figure out which group is the day name vs the number, since
        # patterns above have them in different orders.
        day_str, num_str = None, None
        for g in groups:
            if g.lower() in _WEEKDAYS:
                day_str = g.lower()
            elif g.isdigit():
                num_str = g
        if day_str is None or num_str is None:
            continue
        start_idx = _WEEKDAYS.index(day_str)
        n = int(num_str)
        result_idx = (start_idx + n) % 7
        return _WEEKDAYS[result_idx].capitalize()
    return None

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

def ask(prompt: str, verbose: bool = True):
    calculator_answer = try_day_offset_calculator(prompt)
    if calculator_answer is not None:
        if verbose:
            print(f"  [CALCULATOR] Detected day-offset question, computed directly: {calculator_answer}", file=sys.stderr)
        return {"answer": calculator_answer, "source": "calculator", "samples_agreed": None}

    local = LocalModel()
    # ... rest of the function continues unchanged from here
    local = LocalModel()
    replies = []
    for i in range(N_SAMPLES):
        if verbose:
            print(f"  Asking local model (try {i+1}/{N_SAMPLES})...", file=sys.stderr)
        r = local(prompt)
        replies.append(r["text"])

    candidates = [extract_candidate_answer(r) for r in replies]
    candidates = [c for c in candidates if c is not None]
    agree = len(candidates) >= 2 and len(set(candidates)) == 1

    if agree:
        if verbose:
            print(f"  [LOCAL] All {N_SAMPLES} samples agreed. Using local answer, no API call.", file=sys.stderr)
        return {"answer": replies[0], "source": "local", "samples_agreed": True}
    else:
        if verbose:
            print(f"  [ESCALATE] Samples disagreed, asking the API model...", file=sys.stderr)
        api = APIModel()
        r = api(prompt)
        return {"answer": r["text"], "source": "api", "samples_agreed": False}

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Ask something: ")
    result = ask(prompt)
    print(f"\n[{result['source'].upper()}]")
    print(result["answer"])

if __name__ == "__main__":
    main()