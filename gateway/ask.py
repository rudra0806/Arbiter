"""
The actual, usable application: ask a question, get a routed answer.
Calls the local model up to 3 times; if all 3 agree, returns that answer
for free. If they disagree, escalates to the API model once.
"""
import sys
import re
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gateway.models import LocalModel, APIModel, _post_with_retry

N_SAMPLES = 3


class LocalModelSampled(LocalModel):
    """
    Subclass of LocalModel that overrides temperature to 0.7 for
    genuine self-consistency sampling across multiple calls.
    Keeps temperature=0 in LocalModel untouched for benchmark reproducibility.
    """
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
        t0 = time.perf_counter()
        try:
            resp = _post_with_retry(
                self.OLLAMA_URL,
                headers={"Content-Type": "application/json"},
                payload=payload,
                timeout=self.TIMEOUT,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"LocalModel ({self.model}) error: {exc}") from exc

        latency = time.perf_counter() - t0
        data = resp.json()

        done_reason = data.get("done_reason")
        truncated = (done_reason == "length")

        return {
            "text": data["message"]["content"],
            "latency_s": round(latency, 3),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "truncated": truncated,
        }


from datetime import datetime, timedelta

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_DAY_OFFSET_PATTERNS = [
    re.compile(r"if today is (\w+),?\s*what day (?:is it|will it be)\s*(\d+)\s*days?\s*from (?:now|today)", re.IGNORECASE),
    re.compile(r"what day (?:is it|will it be)\s*(\d+)\s*days?\s*after (\w+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*days?\s*after (\w+)\s*is what day", re.IGNORECASE),
    re.compile(r"(\w+)\s*plus\s*(\d+)\s*days?\s*is what day", re.IGNORECASE),
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


# ---------------------------------------------------------------------------
# FAILURE MODE 1 FIX: Deterministic ranking calculator
# LLMs often fail multi-entity ordering (e.g. 3/3 confident wrong answers)
# despite correct intermediate comparisons. When a prompt presents a clear
# list of entity-number pairs and a ranking/comparison goal, compute it
# directly in Python rather than risking model agreement on an unsorted answer.
# ---------------------------------------------------------------------------

_RANKING_STOPWORDS = {
    'In', 'On', 'At', 'For', 'To', 'From', 'With', 'By', 'After', 'Before',
    'If', 'When', 'Then', 'The', 'A', 'An', 'And', 'Or', 'But', 'Who', 'What',
    'Which', 'Where', 'Why', 'How', 'Rank', 'List', 'Order', 'Sort', 'Find',
    'Four', 'Three', 'Two', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
    'First', 'Second', 'Third', 'Last', 'City', 'Runner', 'Person'
}

_PAIR_PATTERN = re.compile(
    r'\b([A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*)'
    r'(?::\s*|\s+(?:(?i:is|has|had|was|at|with|scored|finished(?:\s+in|\s+at)?))\s*[:=-]?\s*|\s*[:=-]\s*|\s*\(\s*|\s+)'
    r'(-?\$?\d+(?:\.\d+)?)'
    r'(?:\s*((?i:million|billion|thousand|hundred|k|m|b)))?\b'
)

_ASCENDING_PHRASES = [
    r"youngest\s+to\s+oldest",
    r"smallest\s+to\s+largest",
    r"lowest\s+to\s+highest",
    r"least\s+to\s+most",
    r"fastest\s+to\s+slowest",
    r"shortest\s+to\s+tallest",
    r"shortest\s+to\s+longest",
    r"ascending",
]

_DESCENDING_PHRASES = [
    r"oldest\s+to\s+youngest",
    r"largest\s+to\s+smallest",
    r"highest\s+to\s+lowest",
    r"most\s+to\s+least",
    r"slowest\s+to\s+fastest",
    r"tallest\s+to\s+shortest",
    r"longest\s+to\s+shortest",
    r"descending",
]

_ASCENDING_KEYWORDS = [
    r"\bwho\s+finished\s+first\b",
    r"\bfinished\s+first\b",
    r"\bwho\s+is\s+first\b",
    r"\bfirst\b",
    r"\bfastest\b",
    r"\blowest\b",
    r"\byoungest\b",
    r"\bsmallest\b",
    r"\bleast\b",
    r"\bminimum\b",
    r"\bmin\b",
]

_DESCENDING_KEYWORDS = [
    r"\bwho\s+finished\s+last\b",
    r"\bfinished\s+last\b",
    r"\bwho\s+is\s+last\b",
    r"\blast\b",
    r"\bslowest\b",
    r"\boldest\b",
    r"\blargest\b",
    r"\bmost\b",
    r"\bhighest\b",
    r"\bmaximum\b",
    r"\bmax\b",
    r"\bbiggest\b",
    r"\btallest\b",
]

def try_ranking_calculator(prompt: str):
    # WHY: Failure Mode 1 showed local models confidently agreeing on wrong entity
    # rankings/winners (e.g. Dana over Beth despite Beth having a lower race time).
    # Extracting and sorting in Python bypasses the LLM for deterministic ranking tasks.
    pairs = []
    seen_names = set()
    for m in _PAIR_PATTERN.finditer(prompt):
        name = m.group(1).strip()
        if name in _RANKING_STOPWORDS or name.split()[0] in _RANKING_STOPWORDS:
            continue
        if name in seen_names:
            continue
        num_str = m.group(2).replace("$", "")
        unit = (m.group(3) or "").lower()
        try:
            val = float(num_str)
        except ValueError:
            continue
        if unit in ("million", "m"):
            val *= 1_000_000
        elif unit in ("billion", "b"):
            val *= 1_000_000_000
        elif unit in ("thousand", "k"):
            val *= 1_000
        elif unit == "hundred":
            val *= 100
        seen_names.add(name)
        pairs.append((name, val))

    if len(pairs) < 2:
        return None

    # Detect ascending vs descending direction
    is_ascending = None
    for p in _ASCENDING_PHRASES:
        if re.search(p, prompt, re.IGNORECASE):
            is_ascending = True
            break
    if is_ascending is None:
        for p in _DESCENDING_PHRASES:
            if re.search(p, prompt, re.IGNORECASE):
                is_ascending = False
                break
    if is_ascending is None:
        has_asc = any(re.search(kw, prompt, re.IGNORECASE) for kw in _ASCENDING_KEYWORDS)
        has_desc = any(re.search(kw, prompt, re.IGNORECASE) for kw in _DESCENDING_KEYWORDS)
        if has_asc and not has_desc:
            is_ascending = True
        elif has_desc and not has_asc:
            is_ascending = False

    if is_ascending is None:
        return None

    # Sort pairs according to direction
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=(not is_ascending))
    sorted_names = [p[0] for p in sorted_pairs]

    # Distinguish single top/bottom entity questions ("Who finished first?")
    # from full list requests ("List these four people...", "Rank these cities...")
    is_single = (
        bool(re.search(r"\b(who|which\s+(?:person|runner|city|one|team)?)\b", prompt, re.IGNORECASE))
        and not bool(re.search(r"\b(list|rank|order|sort|all)\b", prompt, re.IGNORECASE))
        and not bool(re.search(r"\b\w+\s+to\s+\w+\b", prompt, re.IGNORECASE))
    )

    if is_single:
        return sorted_names[0]
    else:
        return ", ".join(sorted_names)


# ---------------------------------------------------------------------------
# Answer matching and grounding logic (moved to shared module gateway.answer_matching)
# ---------------------------------------------------------------------------
from gateway.answer_matching import (
    extract_numbers,
    is_grounded,
    _CONCLUDING_PATTERNS,
    clean_text_full,
    extract_candidate_answer,
)


def ask(prompt: str, verbose: bool = True):
    # Calculator 1: Day-offset
    calculator_answer = try_day_offset_calculator(prompt)
    if calculator_answer is not None:
        if verbose:
            print(f"  [CALCULATOR] Detected day-offset question, computed directly: {calculator_answer}", file=sys.stderr)
        return {"answer": calculator_answer, "source": "calculator", "samples_agreed": None}

    # Calculator 2: Ranking / comparison (Failure Mode 1)
    ranking_answer = try_ranking_calculator(prompt)
    if ranking_answer is not None:
        if verbose:
            print(f"  [CALCULATOR] Detected ranking question, computed directly: {ranking_answer}", file=sys.stderr)
        return {"answer": ranking_answer, "source": "calculator", "samples_agreed": None}

    local = LocalModelSampled()
    replies = []
    for i in range(N_SAMPLES):
        if verbose:
            print(f"  Asking local model (try {i+1}/{N_SAMPLES})...", file=sys.stderr)
        r = local(prompt)
        replies.append(r["text"])

    # WHY (Failure Mode 2): Discard ungrounded replies that hallucinate numbers not in the prompt
    grounded_replies = [r for r in replies if is_grounded(prompt, r)]

    candidates = [extract_candidate_answer(r) for r in grounded_replies]
    candidates = [c for c in candidates if c is not None]
    agree = len(candidates) >= 2 and len(set(candidates)) == 1

    if agree:
        if verbose:
            print(f"  [LOCAL] Grounded samples agreed. Using local answer, no API call.", file=sys.stderr)
        return {"answer": grounded_replies[0], "source": "local", "samples_agreed": True}
    else:
        if verbose:
            print(f"  [ESCALATE] Samples disagreed or ungrounded, asking the API model...", file=sys.stderr)
        api = APIModel()
        try:
            r = api(prompt)
        except RuntimeError as exc:
            return {"answer": f"Couldn't reach the API model: {exc}", "source": "error", "samples_agreed": False}
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
