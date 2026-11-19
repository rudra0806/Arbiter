"""
tests/test_checkers.py
----------------------
pytest tests for every checker in benchmark/checkers.py.

Run with:
    pytest tests/ -v
"""

import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.checkers import (
    check_contains,
    check_word,
    check_exact,
    check_json,
    check_wordcount,
    check_comma_count,
    check_code_exec,
    check,
    normalize_text,
)


# ---------------------------------------------------------------------------
# check_contains
# ---------------------------------------------------------------------------

class TestCheckContains:
    def test_simple_match(self):
        assert check_contains("The capital is Canberra.", ["canberra"]) is True

    def test_case_insensitive(self):
        assert check_contains("CANBERRA is the answer", ["canberra"]) is True

    def test_multiple_answers_first_matches(self):
        assert check_contains("carbon dioxide is used", ["carbon dioxide", "co2"]) is True

    def test_multiple_answers_second_matches(self):
        assert check_contains("plants absorb CO2", ["carbon dioxide", "co2"]) is True

    def test_no_match(self):
        assert check_contains("Paris is the capital of France", ["canberra"]) is False

    def test_partial_word_still_matches(self):
        # contains does NOT use word boundary — substring is enough
        assert check_contains("The answer is 17", ["7"]) is True

    def test_empty_reply(self):
        assert check_contains("", ["canberra"]) is False


# ---------------------------------------------------------------------------
# check_word
# ---------------------------------------------------------------------------

class TestCheckWord:
    def test_simple_whole_word(self):
        assert check_word("The answer is 7.", ["7"]) is True

    def test_no_match_inside_larger_number(self):
        # "7" must NOT match inside "17"
        assert check_word("The answer is 17.", ["7"]) is False

    def test_no_match_inside_larger_number_70(self):
        assert check_word("70 is the answer", ["7"]) is False

    def test_case_insensitive(self):
        assert check_word("Au is the symbol", ["au"]) is True

    def test_multiple_answers_match(self):
        assert check_word("seven continents", ["7", "seven"]) is True

    def test_decimal_answer(self):
        # "7.5" as a whole word
        assert check_word("The angle is 7.5 degrees.", ["7.5"]) is True

    def test_decimal_no_false_match(self):
        # "7.5" should not match if the reply says "17.5"
        assert check_word("The angle is 17.5 degrees.", ["7.5"]) is False

    def test_no_match(self):
        assert check_word("The answer is 45", ["7"]) is False

    def test_word_at_end_of_line(self):
        assert check_word("Raj\n", ["raj"]) is True


# ---------------------------------------------------------------------------
# check_exact
# ---------------------------------------------------------------------------

class TestCheckExact:
    def test_simple_exact(self):
        assert check_exact("yes", ["yes"]) is True

    def test_exact_with_trailing_period(self):
        assert check_exact("yes.", ["yes"]) is True

    def test_exact_with_quotes(self):
        assert check_exact('"yes"', ["yes"]) is True

    def test_exact_case_insensitive(self):
        assert check_exact("No", ["no"]) is True

    def test_no_match_extra_word(self):
        assert check_exact("yes please", ["yes"]) is False

    def test_exact_multiple_answers(self):
        assert check_exact("HELLO WORLD", ["hello world"]) is True

    def test_exact_mismatch(self):
        assert check_exact("maerts ", ["maerts"]) is True  # trailing space stripped


# ---------------------------------------------------------------------------
# check_json
# ---------------------------------------------------------------------------

class TestCheckJson:
    def test_plain_json(self):
        reply = '{"name": "Asha", "age": 21}'
        assert check_json(reply, {"name": "Asha", "age": 21}) is True

    def test_json_inside_code_fence(self):
        reply = '```json\n{"name": "Asha", "age": 21}\n```'
        assert check_json(reply, {"name": "Asha", "age": 21}) is True

    def test_json_inside_plain_fence(self):
        reply = '```\n{"name": "Asha", "age": 21}\n```'
        assert check_json(reply, {"name": "Asha", "age": 21}) is True

    def test_wrong_value(self):
        reply = '{"name": "Asha", "age": 22}'
        assert check_json(reply, {"name": "Asha", "age": 21}) is False

    def test_missing_key(self):
        reply = '{"name": "Asha"}'
        assert check_json(reply, {"name": "Asha", "age": 21}) is False

    def test_invalid_json(self):
        assert check_json("not json at all", {"name": "Asha"}) is False

    def test_type_mismatch_string_vs_int(self):
        # age "21" (string) != 21 (int)
        reply = '{"name": "Asha", "age": "21"}'
        assert check_json(reply, {"name": "Asha", "age": 21}) is False

    def test_extra_keys_ok(self):
        # Extra keys in reply are allowed; we only check required keys
        reply = '{"name": "Asha", "age": 21, "city": "Mumbai"}'
        assert check_json(reply, {"name": "Asha", "age": 21}) is True


# ---------------------------------------------------------------------------
# check_wordcount
# ---------------------------------------------------------------------------

class TestCheckWordCount:
    def test_exact_five_words(self):
        assert check_wordcount("Rain falls softly every night", [5]) is True

    def test_too_many_words(self):
        assert check_wordcount("Rain falls softly every single night", [5]) is False

    def test_too_few_words(self):
        assert check_wordcount("Rain falls.", [5]) is False

    def test_single_word(self):
        assert check_wordcount("cold", [1]) is True


# ---------------------------------------------------------------------------
# check_comma_count
# ---------------------------------------------------------------------------

class TestCheckCommaCount:
    def test_three_items(self):
        assert check_comma_count("apple, banana, cherry", [3]) is True

    def test_too_few_items(self):
        assert check_comma_count("apple, banana", [3]) is False

    def test_trailing_comma_ignored(self):
        # trailing comma creates an empty item which we strip
        assert check_comma_count("apple, banana, cherry,", [3]) is True

    def test_exact_one_item(self):
        # No comma → 1 item
        assert check_comma_count("apple", [1]) is True


# ---------------------------------------------------------------------------
# check_code_exec
# ---------------------------------------------------------------------------

class TestCheckCodeExec:
    def test_simple_function(self):
        reply = "def is_even(n):\n    return n % 2 == 0\n"
        answers = [
            {"call": "is_even(4)", "expect": True},
            {"call": "is_even(7)", "expect": False},
        ]
        assert check_code_exec(reply, answers) is True

    def test_code_in_fence(self):
        reply = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        answers = [{"call": "is_even(2)", "expect": True}]
        assert check_code_exec(reply, answers) is True

    def test_wrong_implementation(self):
        reply = "def is_even(n):\n    return n % 2 == 1\n"  # inverted logic
        answers = [{"call": "is_even(4)", "expect": True}]
        assert check_code_exec(reply, answers) is False

    def test_syntax_error_in_code(self):
        reply = "def is_even(n)\n    return n % 2 == 0\n"  # missing colon
        answers = [{"call": "is_even(4)", "expect": True}]
        assert check_code_exec(reply, answers) is False

    def test_infinite_loop_times_out(self):
        reply = "def is_even(n):\n    while True: pass\n    return n % 2 == 0\n"
        answers = [{"call": "is_even(4)", "expect": True}]
        # Should return False within ~5 seconds
        assert check_code_exec(reply, answers) is False

    def test_multiple_cases_all_must_pass(self):
        # Second case will fail
        reply = "def is_even(n):\n    return n == 4\n"
        answers = [
            {"call": "is_even(4)", "expect": True},
            {"call": "is_even(2)", "expect": True},  # returns False for 2
        ]
        assert check_code_exec(reply, answers) is False


# ---------------------------------------------------------------------------
# Dispatcher  check()
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_dispatches_contains(self):
        task = {"check": "contains", "answers": ["canberra"]}
        assert check(task, "The capital is Canberra.") is True

    def test_dispatches_word(self):
        task = {"check": "word", "answers": ["7"]}
        assert check(task, "The answer is 7.") is True

    def test_dispatches_exact(self):
        task = {"check": "exact", "answers": ["yes"]}
        assert check(task, "yes") is True

    def test_dispatches_json(self):
        task = {"check": "json", "answers": {"name": "Asha", "age": 21}}
        assert check(task, '{"name": "Asha", "age": 21}') is True

    def test_dispatches_wordcount(self):
        task = {"check": "wordcount", "answers": [5]}
        assert check(task, "Rain falls softly every night") is True

    def test_dispatches_comma_count(self):
        task = {"check": "comma_count", "answers": [3]}
        assert check(task, "apple, banana, cherry") is True

    def test_dispatches_code_exec(self):
        task = {
            "check": "code_exec",
            "answers": [{"call": "is_even(4)", "expect": True}],
        }
        reply = "def is_even(n):\n    return n % 2 == 0"
        assert check(task, reply) is True

    def test_unknown_check_type_raises(self):
        import pytest
        task = {"check": "nonexistent_type", "answers": []}
        with pytest.raises(ValueError, match="Unknown check type"):
            check(task, "some reply")


# ---------------------------------------------------------------------------
# normalize_text and normalized matching
# ---------------------------------------------------------------------------

class TestTextNormalization:
    def test_curly_quotes_match_plain_ascii(self):
        """A reply containing curly quotes still matches a plain-ASCII answer."""
        assert check_exact("“yes”", ['"yes"']) is True
        assert check_exact("‘yes’", ["'yes'"]) is True
        assert check_word("She said “hello” today.", ["hello"]) is True
        assert check_contains("It’s a ‘great’ day", ["it's a 'great' day"]) is True

    def test_latex_grouped_number_matches_plain(self):
        """A LaTeX '{,}'-grouped number (e.g. '8{,}217') matches its plain form ('8217' or '8,217')."""
        assert check_word(r"The answer is \(8{,}217\).", ["8,217"]) is True
        assert check_exact(r"\[8{,}217\]", ["8,217"]) is True
        assert check_contains(r"Total: 8{,}217 units", ["8,217"]) is True
        assert check_word(r"\(8{,}217\)", ["8217", "8,217"]) is True
        assert check_contains(r"Value is \(8{,}217\)", ["8217", "8,217"]) is True

    def test_markdown_bold_wrapping_matches(self):
        """Markdown-bold wrapping around a correct answer doesn't block the match."""
        assert check_exact("**42**", ["42"]) is True
        assert check_word("The answer is **Paris**.", ["paris"]) is True
        assert check_contains("The answer is **Australia**.", ["australia"]) is True
        assert check_exact("`hello`", ["hello"]) is True

    def test_em_dash_matches_plain_hyphen_or_no_hyphen(self):
        """An em-dash in the reply doesn't block a match against a plain-hyphen or no-hyphen answer."""
        # Matches plain-hyphen answer
        assert check_exact("state—of—the—art", ["state-of-the-art"]) is True
        assert check_contains("We offer state—of—the—art models.", ["state-of-the-art"]) is True

        # Matches no-hyphen answer
        assert check_word("The result is — 42.", ["42"]) is True
        assert check_contains("A win—win situation.", ["win"]) is True
