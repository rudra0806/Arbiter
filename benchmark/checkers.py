"""
benchmark/checkers.py
---------------------
One grading function per check type, plus a dispatcher.

All public functions follow the signature:
    check_<type>(reply: str, answers) -> bool

The dispatcher:
    check(task: dict, reply: str) -> bool
"""

import re
import json
import subprocess
import sys
import textwrap
import tempfile
import os


# ---------------------------------------------------------------------------
# Text normalization helper
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text before comparison in 'contains', 'word', and 'exact' checkers:
      1. Replace curly/smart quotes (' ' " ") with straight ASCII quotes (' and ").
      2. Replace en-dash, em-dash, and non-breaking-hyphen variants with '-'.
      3. Strip LaTeX math delimiters \\( \\) \\[ \\], and convert LaTeX digit-grouping
         like "8{,}217" into plain "8,217".
      4. Strip markdown emphasis markers (** and *) and backticks (`).
      5. Collapse runs of whitespace into a single space, and strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Curly/smart quotes to straight ASCII quotes
    for q in ("‘", "’", "‚", "‛"):
        text = text.replace(q, "'")
    for q in ("“", "”", "„", "‟"):
        text = text.replace(q, '"')

    # 2. En-dash, em-dash, and hyphen variants to '-'
    for d in ("–", "—", "―", "‑", "‒", "−", "\u00ad"):
        text = text.replace(d, "-")

    # 3. LaTeX digit-grouping '8{,}217' -> '8,217', strip \( \) \[ \]
    text = re.sub(r"\{\s*,\s*\}", ",", text)
    for delim in (r"\(", r"\)", r"\[", r"\]"):
        text = text.replace(delim, "")

    # 4. Strip markdown emphasis markers (** and *) and backticks (`)
    text = text.replace("*", "").replace("`", "")

    # 5. Collapse runs of whitespace and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------

def check_contains(reply: str, answers: list) -> bool:
    """True if ANY answer string appears anywhere in the reply (case-insensitive)."""
    norm_reply = normalize_text(reply).lower()
    return any(normalize_text(a).lower() in norm_reply for a in answers)


def check_word(reply: str, answers: list) -> bool:
    """
    True if ANY answer appears as a whole word in the reply (case-insensitive).
    Uses \\b word boundaries so "7" does NOT match inside "17" or "70".
    """
    norm_reply = normalize_text(reply).lower()
    for a in answers:
        norm_a = normalize_text(a).lower()
        # re.escape handles answers that contain regex-special chars (e.g. "7.5")
        pattern = r"\b" + re.escape(norm_a) + r"\b"
        if re.search(pattern, norm_reply):
            return True
    return False


def _normalise_exact(text: str) -> str:
    """Strip surrounding whitespace, trailing period, leading/trailing quotes, lowercase."""
    t = normalize_text(text).lower()
    t = t.rstrip(".")
    t = t.strip("\"'")
    return t


def check_exact(reply: str, answers: list) -> bool:
    """True if the normalised reply equals ANY normalised answer."""
    normalised_reply = _normalise_exact(reply)
    return any(_normalise_exact(a) == normalised_reply for a in answers)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences so the inner content can be parsed."""
    # Remove optional language tag after opening fence
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def check_json(reply: str, answers: dict) -> bool:
    """
    Strip code fences, parse the reply as JSON, then check that every
    key-value pair in `answers` matches the parsed object.
    Values are compared with loose equality (int 21 == str "21" is NOT accepted;
    types must match because the task JSON preserves them).
    """
    cleaned = _strip_code_fences(reply)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    for key, expected in answers.items():
        if parsed.get(key) != expected:
            return False
    return True


def check_wordcount(reply: str, answers: list) -> bool:
    """True if the number of whitespace-delimited words in the reply equals answers[0]."""
    expected = int(answers[0])
    actual = len(reply.split())
    return actual == expected


def check_comma_count(reply: str, answers: list) -> bool:
    """True if the number of comma-separated items (stripped) equals answers[0]."""
    expected = int(answers[0])
    items = [item.strip() for item in reply.split(",") if item.strip()]
    return len(items) == expected


# ---------------------------------------------------------------------------
# code_exec checker
# ---------------------------------------------------------------------------

def _extract_python_code(reply: str) -> str:
    """
    Pull Python code from the reply.
    Priority:
      1. Content inside a ```python ... ``` fence
      2. Content inside any ``` ... ``` fence
      3. The whole reply (stripped), as a fallback
    """
    # Try ```python fence first
    m = re.search(r"```python\s*\n(.*?)```", reply, re.DOTALL)
    if m:
        return m.group(1)
    # Any fence
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", reply, re.DOTALL)
    if m:
        return m.group(1)
    return reply.strip()


def check_code_exec(reply: str, answers: list) -> bool:
    """
    Extract Python code from the reply, write it to a temp file, then run
    each {"call": ..., "expect": ...} test case in a subprocess (never exec()
    in-process).  All test cases must pass for the check to return True.

    The subprocess is killed after 5 seconds to prevent infinite loops.
    sys.executable is used so it works on Windows without PATH issues.
    """
    code = _extract_python_code(reply)

    # Build a small test harness that appends to code and prints PASS/FAIL
    test_lines = []
    for i, case in enumerate(answers):
        call = case["call"]
        expect = repr(case["expect"])   # e.g. True, False, 42, "hello"
        test_lines.append(
            f"_result_{i} = {call}\n"
            f"print('CASE_{i}', 'PASS' if _result_{i} == {expect} else 'FAIL')"
        )
    harness = code + "\n" + "\n".join(test_lines)

    # Write to a temporary file (auto-deleted after use)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5,          # kill infinite loops
        )
        output = result.stdout
        # Every test case must print PASS
        passes = [f"CASE_{i} PASS" in output for i in range(len(answers))]
        return all(passes)
    except subprocess.TimeoutExpired:
        return False          # treat timeout (infinite loop, etc.) as failure
    except Exception:
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CHECKER_MAP = {
    "contains":    check_contains,
    "word":        check_word,
    "exact":       check_exact,
    "json":        check_json,
    "wordcount":   check_wordcount,
    "comma_count": check_comma_count,
    "code_exec":   check_code_exec,
}


def check(task: dict, reply: str) -> bool:
    """
    Dispatch to the right checker based on task["check"].
    Returns False (not an error) if the check type is unknown.
    """
    check_type = task.get("check", "")
    checker = _CHECKER_MAP.get(check_type)
    if checker is None:
        raise ValueError(f"Unknown check type: '{check_type}'")
    return checker(reply, task["answers"])
