"""
tests/test_router.py
--------------------
Small pytest tests for gateway/router.py.

Run with:
    pytest tests/test_router.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gateway.router import route

VALID_TARGETS = {"local", "api"}


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

class TestRouterContract:
    def test_returns_dict(self):
        result = route("What is the capital of France?")
        assert isinstance(result, dict)

    def test_has_target_key(self):
        result = route("What is the capital of France?")
        assert "target" in result

    def test_has_reason_key(self):
        result = route("What is the capital of France?")
        assert "reason" in result

    def test_target_is_valid(self):
        result = route("What is the capital of France?")
        assert result["target"] in VALID_TARGETS

    def test_reason_is_nonempty_string(self):
        result = route("What is the capital of France?")
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_deterministic(self):
        """Calling route twice with the same prompt must return the same result."""
        p = "What is 23 * 456?"
        assert route(p) == route(p)

    def test_deterministic_local(self):
        p = "Who wrote Romeo and Juliet?"
        assert route(p) == route(p)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_prompt_returns_local(self):
        """Empty prompt has no complexity signals → default local."""
        result = route("")
        assert result["target"] == "local"

    def test_whitespace_only_prompt(self):
        result = route("   \t\n  ")
        assert result["target"] in VALID_TARGETS
        assert result["reason"]

    def test_very_long_prompt(self):
        """A very long prompt (>160 chars) should be sent to api."""
        long_p = "A " * 100   # 200 chars of filler
        result = route(long_p)
        assert result["target"] == "api"
        assert result["reason"]

    def test_single_word_prompt(self):
        result = route("Hello")
        assert result["target"] in VALID_TARGETS

    def test_prompt_with_only_numbers(self):
        result = route("42")
        assert result["target"] in VALID_TARGETS


# ---------------------------------------------------------------------------
# Rule-specific smoke tests (verify correct direction, not exact reason)
# ---------------------------------------------------------------------------

class TestRuleRouting:
    def test_simple_fact_goes_local(self):
        result = route("What is the capital of Australia?")
        assert result["target"] == "local"

    def test_average_keyword_goes_api(self):
        result = route("What is the average of 10, 20, and 30?")
        assert result["target"] == "api"

    def test_percent_goes_api(self):
        result = route("What is 15% of 240?")
        assert result["target"] == "api"

    def test_large_multiplication_goes_api(self):
        result = route("What is 987 * 36? Answer with only the number.")
        assert result["target"] == "api"

    def test_chained_multiplication_goes_api(self):
        result = route("What is 17 * 23 * 2? Answer with only the number.")
        assert result["target"] == "api"

    def test_algebra_equation_goes_api(self):
        result = route("If 3x + 2x = 75, what is x?")
        assert result["target"] == "api"

    def test_clock_angle_goes_api(self):
        result = route("What is the angle between the clock hands at 3:15?")
        assert result["target"] == "api"

    def test_reverse_word_goes_api(self):
        result = route("Reverse the word 'stream'. Output only the reversed word.")
        assert result["target"] == "api"

    def test_exact_wordcount_format_goes_api(self):
        result = route("Write a sentence about rain that has exactly 5 words.")
        assert result["target"] == "api"

    def test_dollar_cents_goes_api(self):
        result = route("A bat and a ball cost $1.10. The ball costs how many cents?")
        assert result["target"] == "api"

    def test_days_from_now_goes_api(self):
        result = route("What day will it be 10 days from now?")
        assert result["target"] == "api"

    def test_code_easy_goes_local(self):
        result = route("What function returns the length of a list in Python?")
        assert result["target"] == "local"

    def test_sentiment_extract_goes_local(self):
        result = route("Is the sentiment positive or negative? 'I loved it!'")
        assert result["target"] == "local"
