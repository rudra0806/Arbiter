"""
tests/test_ask.py
-----------------
Tests for the three failure mode fixes in gateway/ask.py:
  1. Multi-entity ranking calculator (try_ranking_calculator)
  2. Numeric hallucination grounding check (is_grounded & ask escalation)
  3. False agreement on open-ended non-numeric answers (extract_candidate_answer)
"""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gateway.ask import (
    ask,
    try_ranking_calculator,
    is_grounded,
    extract_numbers,
    extract_candidate_answer,
    LocalModelSampled,
)


# ---------------------------------------------------------------------------
# FAILURE MODE 1: Multi-entity ranking / comparison calculator
# ---------------------------------------------------------------------------

class TestRankingCalculator:
    def test_reproduction_1_age_ranking(self):
        """Sam (34), Priya (29), Tom (41), Lily (37) from youngest to oldest -> Priya, Sam, Lily, Tom."""
        prompt = "Sam is 34 years old, Priya is 29, Tom is 41, and Lily is 37. List these four people from youngest to oldest."
        result = try_ranking_calculator(prompt)
        assert result == "Priya, Sam, Lily, Tom"

    def test_reproduction_2_race_winner(self):
        """Alex 12.4s, Beth 11.9s, Carl 13.1s, Dana 12.0s. Who finished first? -> Beth (lowest time)."""
        prompt = "Four runners finished a race with times: Alex 12.4 seconds, Beth 11.9 seconds, Carl 13.1 seconds, Dana 12.0 seconds. Who finished first?"
        result = try_ranking_calculator(prompt)
        assert result == "Beth"

    def test_reproduction_3_city_population(self):
        """CityA (2.3M), CityB (1.8M), CityC (3.1M) from largest to smallest -> CityC, CityA, CityB."""
        prompt = "Rank these cities by population from largest to smallest: CityA has 2.3 million, CityB has 1.8 million, CityC has 3.1 million."
        result = try_ranking_calculator(prompt)
        assert result == "CityC, CityA, CityB"

    def test_returns_none_for_non_ranking_prompt(self):
        """Prompts without ranking intent or ambiguous direction return None to let normal flow handle them."""
        prompt = "Alice has 5 apples and Bob has 3 apples. How many apples do they have in total?"
        assert try_ranking_calculator(prompt) is None

    def test_returns_none_for_single_entity(self):
        """Prompts with fewer than 2 entities return None."""
        prompt = "Sam is 34 years old. Is he older than 30?"
        assert try_ranking_calculator(prompt) is None

    def test_ask_integration_bypasses_llm(self):
        """When ranking calculator matches, ask() returns calculator source without calling local/API models."""
        prompt = "Sam is 34 years old, Priya is 29, Tom is 41, and Lily is 37. List these four people from youngest to oldest."
        res = ask(prompt, verbose=False)
        assert res["source"] == "calculator"
        assert res["answer"] == "Priya, Sam, Lily, Tom"
        assert res["samples_agreed"] is None


# ---------------------------------------------------------------------------
# FAILURE MODE 2: Numeric hallucination grounding check
# ---------------------------------------------------------------------------

class TestNumericGrounding:
    def test_bank_account_reproduction_is_ungrounded(self):
        """Replies introducing 1000, 2000 when prompt only has $50, $80, $25, $10 are marked ungrounded."""
        prompt = "A bank account starts at $50. A withdrawal of $80 is made, then a deposit of $25, then another withdrawal of $10. What is the final balance?"
        reply = (
            "Suppose the bank account starts with 1000. A withdrawal of 2000 is made, "
            "then another deposit, ending with 1000. The balance is unchanged."
        )
        assert is_grounded(prompt, reply) is False

    def test_single_derived_answer_remains_grounded(self):
        """A normal calculation producing 1 new final answer number is grounded."""
        prompt = "What is 17 times 23?"
        reply = "17 * 23 = 391. Therefore the answer is 391."
        assert is_grounded(prompt, reply) is True

    def test_bank_account_escalates_to_api(self):
        """When local samples hallucinate ungrounded numbers, ask() forces escalation to API."""
        prompt = "A bank account starts at $50. A withdrawal of $80 is made, then a deposit of $25, then another withdrawal of $10. What is the final balance?"
        hallucinated_text = "Assume balance is 1000. After 2000 withdrawn and 1000 deposited, balance is unchanged."

        with patch("gateway.ask.LocalModelSampled") as mock_local_cls, patch("gateway.ask.APIModel") as mock_api_cls:
            mock_local = MagicMock()
            mock_local.side_effect = lambda p: {"text": hallucinated_text}
            mock_local_cls.return_value = mock_local

            mock_api = MagicMock()
            mock_api.side_effect = lambda p: {"text": "-$15"}
            mock_api_cls.return_value = mock_api

            res = ask(prompt, verbose=False)

            # Must escalate to API because all 3 local samples are ungrounded
            assert res["source"] == "api"
            assert res["samples_agreed"] is False
            assert res["answer"] == "-$15"
            mock_api.assert_called_once_with(prompt)


# ---------------------------------------------------------------------------
# FAILURE MODE 3: Open-ended non-numeric answers & candidate extraction
# ---------------------------------------------------------------------------

class TestCandidateExtractionAndAgreement:
    def test_numbered_list_does_not_extract_trailing_number(self):
        """A numbered list does not extract its own trailing index (e.g. '3')."""
        reply = "Here are 3 ideas:\n1. Dinosaur excavation kit\n2. T-rex action figure\n3. Prehistoric puzzle"
        cand = extract_candidate_answer(reply)
        assert cand != "3"
        # Should return full normalized text
        assert "dinosaur excavation kit" in cand

    def test_numbered_sequential_math_extracts_candidate(self):
        r"""Sequential numbered math steps ending in a concluding phrase extract the final answer."""
        reply = (
            "1. Start with $50.\n"
            "2. Withdraw $80, balance -$30.\n"
            "3. Deposit $25, balance -$5.\n"
            r"4. Withdraw $10. Therefore \boxed{-15}."
        )
        assert extract_candidate_answer(reply) == "-15"

    def test_open_ended_email_does_not_extract_signoff_word(self):
        """An email without concluding phrase does not extract the sign-off word."""
        reply = "Hi Mark,\nCould we reschedule our meeting to Friday afternoon?\nThanks,\nAlice"
        cand = extract_candidate_answer(reply)
        assert cand != "alice"
        assert cand != "thanks"
        assert "could we reschedule" in cand

    def test_verified_math_boxed_still_matches(self):
        r"""Mathematical answers with \boxed{...} extract the boxed candidate."""
        reply = r"After solving, we get \boxed{44}."
        assert extract_candidate_answer(reply) == "44"

    def test_verified_math_equals_still_matches(self):
        """Mathematical answers with '=' extract the answer after equals."""
        reply = "17 * 23 = 391"
        assert extract_candidate_answer(reply) == "391"

    def test_verified_math_concluding_phrases_match(self):
        """Explicit concluding phrases ('the answer is', 'therefore', etc.) extract the answer."""
        assert extract_candidate_answer("Therefore, the answer is 44.") == "44"
        assert extract_candidate_answer("The total is 391.") == "391"
        assert extract_candidate_answer("So, 44.") == "44"
        assert extract_candidate_answer("Thus 44.") == "44"

    def test_dinosaur_gift_ideas_escalate_to_api(self):
        """Different open-ended list responses disagree on full text and escalate to API."""
        prompt = "Give me three ideas for a birthday gift for a 10-year-old who likes dinosaurs."
        replies = [
            {"text": "1. Dinosaur Lego\n2. T-Rex model\n3. Fossil dig kit"},
            {"text": "1. Dinosaur book\n2. Museum tickets\n3. Dino pajamas"},
            {"text": "1. Remote control raptor\n2. Dinosaur puzzle\n3. Wall poster"},
        ]

        with patch("gateway.ask.LocalModelSampled") as mock_local_cls, patch("gateway.ask.APIModel") as mock_api_cls:
            mock_local = MagicMock()
            mock_local.side_effect = replies
            mock_local_cls.return_value = mock_local

            mock_api = MagicMock()
            mock_api.side_effect = lambda p: {"text": "API gift suggestions"}
            mock_api_cls.return_value = mock_api

            res = ask(prompt, verbose=False)

            assert res["source"] == "api"
            assert res["samples_agreed"] is False
            assert res["answer"] == "API gift suggestions"

    def test_reschedule_email_escalates_to_api(self):
        """Different open-ended emails disagree on full text and escalate to API."""
        prompt = "Write a short, friendly email asking a colleague to reschedule a meeting."
        replies = [
            {"text": "Hi Bob,\nCould we move our chat to Friday?\nBest, Alice"},
            {"text": "Hello Bob,\nWould it be possible to reschedule our meeting?\nThanks, Alice"},
            {"text": "Hey Bob,\nLet's postpone our sync to next week.\nRegards, Alice"},
        ]

        with patch("gateway.ask.LocalModelSampled") as mock_local_cls, patch("gateway.ask.APIModel") as mock_api_cls:
            mock_local = MagicMock()
            mock_local.side_effect = replies
            mock_local_cls.return_value = mock_local

            mock_api = MagicMock()
            mock_api.side_effect = lambda p: {"text": "API written email"}
            mock_api_cls.return_value = mock_api

            res = ask(prompt, verbose=False)

            assert res["source"] == "api"
            assert res["samples_agreed"] is False
            assert res["answer"] == "API written email"

    def test_identical_math_samples_still_agree_locally(self):
        """Identical short math replies still agree and return free local answer."""
        prompt = "What is 17 * 23?"
        replies = [
            {"text": "17 * 23 = 391"},
            {"text": "The answer is 391."},
            {"text": r"Therefore \boxed{391}"},
        ]

        with patch("gateway.ask.LocalModelSampled") as mock_local_cls, patch("gateway.ask.APIModel") as mock_api_cls:
            mock_local = MagicMock()
            mock_local.side_effect = replies
            mock_local_cls.return_value = mock_local

            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api

            res = ask(prompt, verbose=False)

            assert res["source"] == "local"
            assert res["samples_agreed"] is True
            mock_api.assert_not_called()


# ---------------------------------------------------------------------------
# TEMPERATURE OVERRIDE: LocalModelSampled testing
# ---------------------------------------------------------------------------

class TestLocalModelSampled:
    def test_local_model_sampled_temperature_point_seven(self):
        """Verifies that LocalModelSampled sends temperature: 0.7 in request options."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Sampled reply"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 15,
            "eval_count": 42,
        }

        with patch("gateway.ask._post_with_retry", return_value=mock_response) as mock_post:
            model = LocalModelSampled()
            result = model("test prompt")

        assert result["text"] == "Sampled reply"
        assert result["tokens_in"] == 15
        assert result["tokens_out"] == 42
        assert result["truncated"] is False

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        call_args = mock_post.call_args.args
        payload = call_kwargs.get("payload", call_args[2] if len(call_args) > 2 else None)
        assert payload is not None
        assert payload["options"]["temperature"] == 0.7


class TestAnswerMatchingSharedModule:
    def test_direct_imports_and_single_source_of_truth(self):
        """Confirm answer matching functions can be directly imported from gateway.answer_matching."""
        from gateway.answer_matching import (
            extract_candidate_answer as ec,
            is_grounded as ig,
            extract_numbers as en,
        )
        import gateway.ask as ga
        assert ec is ga.extract_candidate_answer
        assert ig is ga.is_grounded
        assert en is ga.extract_numbers

