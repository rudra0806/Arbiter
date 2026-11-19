"""
tests/test_models.py
--------------------
Tests for LocalModel and APIModel with mocked HTTP responses:
  - Gemini multi-part reply (joining parts, skipping thought parts)
  - Gemini MAX_TOKENS finish reason (raises 'reply truncated')
  - Gemini token counting (candidatesTokenCount + thoughtsTokenCount)
  - Ollama done_reason of 'length' (marks truncated=True)
"""

from unittest.mock import MagicMock, patch
import pytest

from gateway.models import LocalModel, APIModel


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Ensure standard environment variables for models during testing."""
    monkeypatch.setenv("LOCAL_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("LOCAL_NUM_PREDICT", "512")
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("API_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("API_KEY", "test-secret-key-12345")
    monkeypatch.setenv("API_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setattr("time.sleep", lambda s: None)


class TestGeminiModel:
    def test_gemini_multipart_reply(self):
        """Builds reply by joining all non-thought parts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I should think about this...", "thought": True},
                            {"text": "Part 1: Hello, "},
                            {"text": "Part 2: World!"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8,
                "thoughtsTokenCount": 20,
            },
        }

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            result = model("test prompt")

        assert result["text"] == "Part 1: Hello, Part 2: World!"
        assert result["truncated"] is False

    def test_gemini_max_tokens_finish_reason(self):
        """Raises 'reply truncated' if finishReason is MAX_TOKENS."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Partial answer cut off"},
                        ]
                    },
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2048,
            },
        }

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            with pytest.raises(RuntimeError, match="reply truncated"):
                model("test prompt")

    def test_gemini_empty_text_raises_truncated(self):
        """Raises 'reply truncated' if the non-thought text is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Only thought part", "thought": True},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 50,
            },
        }

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            with pytest.raises(RuntimeError, match="reply truncated"):
                model("test prompt")

    def test_gemini_token_counting(self):
        """tokens_out includes candidatesTokenCount PLUS thoughtsTokenCount."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Answer text"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 25,
                "thoughtsTokenCount": 100,
            },
        }

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            result = model("test prompt")

        assert result["tokens_in"] == 15
        assert result["tokens_out"] == 125  # 25 + 100


class TestOllamaModel:
    def test_ollama_done_reason_length(self):
        """Sets truncated=True when done_reason is 'length'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Truncated reply..."},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 20,
            "eval_count": 512,
        }

        with patch("requests.post", return_value=mock_response):
            model = LocalModel()
            result = model("test prompt")

        assert result["truncated"] is True
        assert result["text"] == "Truncated reply..."
        assert result["tokens_in"] == 20
        assert result["tokens_out"] == 512

    def test_ollama_normal_completion(self):
        """Sets truncated=False when done_reason is normal (e.g. 'stop')."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Full reply."},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 50,
        }

        with patch("requests.post", return_value=mock_response):
            model = LocalModel()
            result = model("test prompt")

        assert result["truncated"] is False


class TestRunEvalSummary:
    def test_summary_includes_truncated_count(self, capsys):
        """Verifies that _print_summary prints the truncated tasks count."""
        from benchmark.run_eval import _print_summary

        records = [
            {"correct": True, "truncated": False, "latency_s": 1.0, "est_cost_usd": 0.0},
            {"correct": False, "truncated": True, "latency_s": 2.0, "est_cost_usd": 0.0},
        ]
        _print_summary(records, "test-model")
        captured = capsys.readouterr().out
        assert "Truncated : 1" in captured

    def test_sanitize_filename_unsafe_characters(self):
        """Verifies that model names with slashes and unsafe characters are sanitized for filenames."""
        from benchmark.run_eval import _sanitize_filename

        assert _sanitize_filename("openai/gpt-oss-120b") == "openai-gpt-oss-120b"
        assert _sanitize_filename("meta-llama/Llama-3.2-3B-Instruct") == "meta-llama-Llama-3.2-3B-Instruct"
        assert _sanitize_filename('bad:model*name?with"chars<and>pipes|') == "bad-model-name-with-chars-and-pipes-"

    def test_summary_includes_skipped_invalid_count(self, capsys):
        """Verifies that _print_summary prints skipped invalid tasks and excludes them from total."""
        from benchmark.run_eval import _print_summary

        records = [
            {"id": 1, "correct": True, "truncated": False, "latency_s": 1.0, "est_cost_usd": 0.0},
            {"id": 207, "correct": False, "truncated": False, "latency_s": 2.0, "est_cost_usd": 0.0},
        ]
        _print_summary(records, "test-model")
        captured = capsys.readouterr().out
        assert "Skipped (invalid): 1" in captured
        assert "Tasks run : 1" in captured


class TestRetryLogic:
    def test_retry_captures_status_and_body_snippet(self):
        """Verifies that failure after retries includes status, body, and not None."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.text = '{"error": "Quota exceeded for resource"}'

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            with pytest.raises(RuntimeError) as exc_info:
                model("test prompt")

        err = str(exc_info.value)
        assert "Request failed after 3 attempts" in err
        assert "None" not in err.split("attempts:")[-1]
        assert "429" in err
        assert "Quota exceeded for resource" in err

    def test_retry_redacts_api_key(self, monkeypatch):
        """Verifies that API keys never leak into the retry error message."""
        secret_key = "very-secret-gemini-token"
        monkeypatch.setenv("API_KEY", secret_key)

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_response.text = f'{{"error": "Invalid API key: {secret_key}"}}'

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            with pytest.raises(RuntimeError) as exc_info:
                model("test prompt")

        err = str(exc_info.value)
        assert secret_key not in err
        assert "<REDACTED>" in err

    def test_exponential_backoff_sleep_calls(self, monkeypatch):
        """Verifies that exponential backoff pauses (2s, 4s) occur between retries."""
        sleep_durations = []
        monkeypatch.setattr("time.sleep", lambda s: sleep_durations.append(s))

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.text = '{"error": "Rate limit"}'

        with patch("requests.post", return_value=mock_response):
            model = APIModel()
            with pytest.raises(RuntimeError):
                model("test prompt")

        # Initial call has inter-call delay (1.5s), then 2 retries with backoff 2s and 4s
        assert 2 in sleep_durations
        assert 4 in sleep_durations
