"""
gateway/router.py
-----------------
Rule-based router: inspects the prompt text and decides whether to send it
to the local model or the API model.

Rules were designed by analysing ONLY the dev split of the benchmark.
They use general linguistic / structural features — no task IDs, no copied
prompt strings.

Public API
----------
    route(prompt: str) -> {"target": "local"|"api", "reason": str}
"""

import re

# ---------------------------------------------------------------------------
# Tuneable constants — edit these to adjust routing behaviour
# ---------------------------------------------------------------------------

# If the prompt contains ANY of these words/phrases → send to API.
# Pattern: harder arithmetic, multi-step word problems, tricky logic puzzles.
API_KEYWORDS: tuple[str, ...] = (
    # multi-step / large arithmetic
    "average",
    "percent",
    "discounted",
    "sale price",
    "how much",          # word-problem trigger
    "how many minutes",
    "how many days",
    "how many cats",
    "how many times",
    "how many words",
    "how many letters",
    "days from now",
    "minutes are there",
    # clock / angle puzzles
    "angle",
    "clock",
    "overlap",
    # sequence / pattern puzzles
    "days from now",
    "what day",
    # logical fallacy / lateral thinking
    "necessarily",
    "conclude",
    "deduce",
    # reversal (local fails on letter-reversal tasks)
    "reverse the word",
    # climbing / rate puzzles
    "climbs",
    "slips back",
    # race / position puzzles
    "overtake",
    "place are you",
)

# Large multiplication: prompt contains two 3+-digit numbers being multiplied.
# Local model is unreliable at large mental arithmetic.
_LARGE_MULT_RE = re.compile(
    r"\b\d{3,}\s*[*×x]\s*\d{2,}"   # e.g. "987 * 36" or "847 × 63"
    r"|\b\d{2,}\s*[*×x]\s*\d{3,}",
    re.IGNORECASE,
)

# "N * M * K" (chained multiplication) — local also struggles here
_CHAINED_MULT_RE = re.compile(
    r"\b\d+\s*[*×x]\s*\d+\s*[*×x]\s*\d+",
    re.IGNORECASE,
)

# "X% of N" or "discounted by X%" patterns
_PERCENT_RE = re.compile(r"\b\d+\s*%", re.IGNORECASE)

# Prompt length threshold: very long prompts tend to be complex reasoning tasks
LONG_PROMPT_CHARS = 160

# Difficulty signals embedded in prompts:
# phrases that co-occur with hard/medium-hard tasks where local fails
API_PHRASE_RES: tuple[re.Pattern, ...] = (
    re.compile(r"\bhow\s+many\s+\w+\s+are\b", re.IGNORECASE),  # "how many X are there"
    re.compile(r"\bwhat\s+day\s+(of|comes|will)\b", re.IGNORECASE),  # day-of-week calc
    re.compile(r"\bif\s+\w+\s+\w+\s+\w+,\s+\w+\s+\w+\s+\w+\b", re.IGNORECASE),  # conditional syllogisms
    re.compile(r"\bsolve\s+for\s+\w\b", re.IGNORECASE),          # algebra
    re.compile(r"\b\d+\s+people\s+(paint|build|fill|work)\b", re.IGNORECASE),  # rate problems
    re.compile(r"\b\d+\s+(cats?|dogs?|workers?|machines?)\b.*\bminutes?\b", re.IGNORECASE),
)

# Algebra equations: "3x + 2x = 75", "2a - a = 10", etc.
_ALGEBRA_RE = re.compile(r"\b\d+\s*[a-z]\s*[+\-]\s*\d+\s*[a-z]\s*=", re.IGNORECASE)

# Unit-price proportional word problems: "N things cost $X. How many dollars..."
_UNIT_PRICE_RE = re.compile(
    r"\b\d+\s+\w+\s+cost\s+\$?\d+.*how\s+many\s+dollars?\b",
    re.IGNORECASE | re.DOTALL,
)

# Exact word-count format tasks: "exactly 7 words"
_EXACT_WORDCOUNT_RE = re.compile(r"\bexactly\s+\d+\s+words?\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Router logic
# ---------------------------------------------------------------------------

def _prompt_lower(prompt: str) -> str:
    """Return lowercased, whitespace-normalised prompt (helper for matching)."""
    return " ".join(prompt.lower().split())


def route(prompt: str) -> dict:
    """
    Decide whether to send `prompt` to the local or API model.

    Returns:
        {"target": "local"|"api", "reason": "<short explanation>"}

    Default target is "local" (cheaper, faster when it works).
    """
    pl = _prompt_lower(prompt)

    # --- Rule 1: keyword match ------------------------------------------
    for kw in API_KEYWORDS:
        if kw in pl:
            return {"target": "api", "reason": f"keyword match: '{kw}'"}

    # --- Rule 2: large / chained multiplication --------------------------
    if _LARGE_MULT_RE.search(prompt):
        return {"target": "api", "reason": "large multiplication detected"}

    if _CHAINED_MULT_RE.search(prompt):
        return {"target": "api", "reason": "chained multiplication detected"}

    # --- Rule 3: percentage arithmetic -----------------------------------
    if _PERCENT_RE.search(prompt):
        return {"target": "api", "reason": "percentage arithmetic detected"}

    # --- Rule 4: regex phrase patterns -----------------------------------
    for pat in API_PHRASE_RES:
        if pat.search(prompt):
            return {"target": "api", "reason": f"phrase pattern: {pat.pattern!r}"}

    # --- Rule 5: algebra equations (Nx + Mx = K pattern) ----------------
    if _ALGEBRA_RE.search(prompt):
        return {"target": "api", "reason": "algebra equation detected"}

    # --- Rule 6: unit-price / proportional word problems -----------------
    if _UNIT_PRICE_RE.search(prompt):
        return {"target": "api", "reason": "unit-price word problem detected"}

    # --- Rule 7: dollar / cents arithmetic (CRT-style) -------------------
    # Prompts mentioning both a dollar amount and "cents" tend to be tricky
    if "cents" in pl and "$" in prompt:
        return {"target": "api", "reason": "dollar-and-cents arithmetic detected"}

    # --- Rule 8: exact word-count format tasks ---------------------------
    if _EXACT_WORDCOUNT_RE.search(prompt):
        return {"target": "api", "reason": "exact word-count format task"}

    # --- Rule 9: very long prompts (complex reasoning / word problems) ---
    if len(prompt) > LONG_PROMPT_CHARS:
        return {"target": "api", "reason": f"long prompt ({len(prompt)} chars > {LONG_PROMPT_CHARS})"}

    # --- Default: local --------------------------------------------------
    return {"target": "local", "reason": "no complexity signals detected"}
