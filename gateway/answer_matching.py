"""
gateway/answer_matching.py
--------------------------
Shared logic for extracting candidate answers, checking numeric grounding,
and normalizing text for self-consistency agreement checks.

Single source of truth used by gateway/ask.py and offline evaluation scripts
(scripts/eval_verifier.py, scripts/results_summary.py).
"""

import re


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values with currency and comma stripping for float comparison."""
    if not text:
        return []
    cleaned = text.replace("$", "")
    raw_nums = re.findall(r"-?\b\d[\d,]*(?:\.\d+)?\b", cleaned)
    nums = []
    for n in raw_nums:
        try:
            val = float(n.replace(",", ""))
            nums.append(val)
        except ValueError:
            continue
    return nums


def is_grounded(prompt: str, reply: str) -> bool:
    # WHY: Failure Mode 2 showed the local model fabricating entirely different
    # input numbers (e.g. 1000, 2000) instead of engaging with the numbers the
    # PROMPT actually gave (e.g. $50, $80, $25, $10), then confidently agreeing
    # with itself on an answer built from those invented inputs.
    prompt_numbers = extract_numbers(prompt)
    if not prompt_numbers:
        return True  # nothing numeric to ground against (e.g. a logic question)

    reply_numbers = extract_numbers(reply)
    if not reply_numbers:
        return False  # prompt had numbers, reply mentions none at all

    # Single-number replies (e.g. 'The answer is 391.' or '\boxed{391}') simply state
    # the candidate answer without fabricating multiple fake input parameters.
    if len(reply_numbers) == 1:
        return True

    matched = sum(
        1 for pn in prompt_numbers
        if any(abs(pn - rn) < 1e-4 for rn in reply_numbers)
    )
    # Require that most of the prompt's given numbers show up in the reasoning,
    # or that the reply does not introduce 2 or more ungrounded values.
    if matched >= max(1, (len(prompt_numbers) + 1) // 2):
        return True

    ungrounded_values = []
    for rn in reply_numbers:
        if not any(abs(rn - pn) < 1e-4 for pn in prompt_numbers):
            if not any(abs(rn - uv) < 1e-4 for uv in ungrounded_values):
                ungrounded_values.append(rn)
    return len(ungrounded_values) < 2


# ---------------------------------------------------------------------------
# FAILURE MODE 3 FIX: Grounded candidate extraction
# Bare trailing numbers or sign-off words were falsely matched across completely
# unrelated lists (matching the final list number '3') or emails (matching 'regards').
# We only extract short candidates near explicit concluding phrases or markers
# (including words like 'the answer is', 'therefore', 'boxed', and bare '=' signs);
# otherwise the full cleaned text is compared so open-ended generations escalate properly.
# ---------------------------------------------------------------------------

_CONCLUDING_PATTERNS = [
    re.compile(r"\bboxed\b", re.IGNORECASE),
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\btherefore\b", re.IGNORECASE),
    re.compile(r"\bequals\b", re.IGNORECASE),
    re.compile(r"=", re.IGNORECASE),  # bare '=' concluding marker for math results like '17 * 23 = 391'
    re.compile(r"\btotal is\b", re.IGNORECASE),
    re.compile(r"\bfinal answer\b", re.IGNORECASE),
    re.compile(r"\bso,", re.IGNORECASE),
    re.compile(r"\bthus\b", re.IGNORECASE),
]


def clean_text_full(text: str) -> str:
    cleaned = text.replace("**", "").replace("*", "").replace("`", "")
    cleaned = re.sub(r"\\[\(\)\[\]]", "", cleaned)
    cleaned = re.sub(r"\{,\}", ",", cleaned)
    return " ".join(cleaned.split()).strip().lower()


def extract_candidate_answer(text: str):
    # WHY: Failure Mode 3 showed open-ended answers (emails, numbered gift ideas)
    # falsely agreeing because extract_candidate_answer grabbed a meaningless trailing
    # number (e.g. list item '3') or sign-off word. Only trust short candidates within
    # ~15 words of an explicit concluding phrase; otherwise use full cleaned text.
    if not text:
        return None

    cleaned = text.replace("**", "").replace("*", "").replace("`", "")
    cleaned = re.sub(r"\\[\(\)\[\]]", "", cleaned)
    cleaned = re.sub(r"\{,\}", ",", cleaned)

    # Check for candidate trailing number or short word first
    number_matches = list(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", cleaned))
    if number_matches:
        last_m = number_matches[-1]
        cand = last_m.group(0).replace(",", "")
        start_idx = last_m.start()
        end_idx = last_m.end()
    else:
        word_matches = list(re.finditer(r"\b[A-Za-z]+\b", cleaned))
        short_matches = [m for m in word_matches if len(m.group(0)) <= 12]
        if short_matches:
            last_m = short_matches[-1]
            cand = last_m.group(0).lower()
            start_idx = last_m.start()
            end_idx = last_m.end()
        else:
            return clean_text_full(text)

    # Check whether the candidate appears within roughly 15 words of an explicit concluding phrase.
    # Running this first allows sequential step-by-step math reasoning formatted as numbered steps
    # to still extract its final answer (e.g. "\boxed{-15}") successfully.
    words_before = cleaned[:start_idx].split()[-15:]
    window_before = " ".join(words_before)
    words_after = cleaned[end_idx:].split()[:15]
    window_after = " ".join(words_after)
    window = window_before + " " + window_after

    has_concluding = any(p.search(window) for p in _CONCLUDING_PATTERNS)
    if has_concluding:
        return cand

    # If no concluding phrase was found near the candidate, check for numbered lists
    # (parallel item lists like gift suggestions where trailing numbers shouldn't match)
    # or general text lacking a concluding marker, falling back to full-text comparison.
    lines = text.strip().splitlines()
    numbered_lines = sum(1 for line in lines if re.match(r"^\s*\d+[\.\)]\s+", line))
    if numbered_lines >= 2:
        return clean_text_full(text)

    return clean_text_full(text)
