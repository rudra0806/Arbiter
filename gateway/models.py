"""
gateway/models.py
-----------------
Two model wrappers that share a common return shape:
  {"text": str, "latency_s": float, "tokens_in": int, "tokens_out": int, "truncated": bool}

LocalModel  → Ollama HTTP API  (http://localhost:11434/api/chat)
APIModel    → Gemini or Groq HTTP API  (provider chosen via PROVIDER env var)
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _redact(text: str, key: str | None = None) -> str:
    """Replace any occurrence of the API key in `text` with <REDACTED>."""
    api_key = key if key is not None else os.getenv("API_KEY", "")
    if api_key and api_key in text:
        return text.replace(api_key, "<REDACTED>")
    return text


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff: 2s, 4s, 8s … capped at 8 seconds."""
    time.sleep(min(2 ** (attempt + 1), 8))


MAX_RETRIES = 2


def _post_with_retry(url: str, headers: dict, payload: dict,
                     timeout: int, max_retries: int = MAX_RETRIES,
                     rate_limit_codes: tuple = (429,)) -> requests.Response:
    """
    POST with up to `max_retries` retries on network errors or rate-limit
    responses.  Raises on final failure with full error details.
    """
    last_exc = None
    key_candidates = [os.getenv("API_KEY", "")]
    if headers:
        for hk, hv in headers.items():
            if hk.lower() == "x-goog-api-key" and isinstance(hv, str):
                key_candidates.append(hv)
            elif hk.lower() == "authorization" and isinstance(hv, str) and hv.startswith("Bearer "):
                key_candidates.append(hv[7:].strip())
    if "?key=" in url:
        key_candidates.append(url.split("?key=")[-1].split("&")[0])

    for attempt in range(max_retries + 1):
        resp = None
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=timeout)
            if resp.status_code in rate_limit_codes:
                if attempt < max_retries:
                    # Respect Retry-After header if present, else use backoff (2s, 4s, 8s)
                    retry_after = None
                    if "Retry-After" in resp.headers:
                        try:
                            retry_after = int(resp.headers["Retry-After"])
                        except (ValueError, TypeError):
                            retry_after = None
                    if retry_after is not None and retry_after > 0:
                        time.sleep(retry_after)
                    else:
                        _backoff_sleep(attempt)
                    continue  # retry
                # Final attempt reached rate limit: raise HTTPError to capture details
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} Client Error: Rate limit exceeded",
                    response=resp,
                )

            if resp.status_code >= 400:
                resp.raise_for_status()
                # Safety fallback if a mock response object didn't raise
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} Error",
                    response=resp,
                )

            return resp
        except requests.exceptions.RequestException as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            err_resp = getattr(exc, "response", None) or resp
            status_code = getattr(err_resp, "status_code", None) if err_resp is not None else None
            try:
                raw_text = err_resp.text if err_resp is not None else ""
            except Exception:
                raw_text = ""
            snippet = raw_text.strip()[:300] if raw_text else ""

            err_str = f"{exc_type}: {exc_msg}"
            if status_code is not None:
                err_str += f" [status {status_code}]"
            if snippet:
                err_str += f" - body: {snippet}"

            for k in key_candidates:
                if k:
                    err_str = _redact(err_str, k)
            last_exc = err_str

            if attempt < max_retries:
                _backoff_sleep(attempt)

    raise RuntimeError(f"Request failed after {max_retries + 1} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# LocalModel – Ollama
# ---------------------------------------------------------------------------

class LocalModel:
    """
    Calls the Ollama /api/chat endpoint in non-streaming mode.

    Config (via .env):
        LOCAL_MODEL        – Ollama model tag, e.g. qwen2.5:1.5b (default)
        LOCAL_NUM_PREDICT  – max tokens to generate (default 512)

    Returns a dict with text, latency_s, tokens_in, tokens_out, truncated.
    """

    OLLAMA_URL = "http://localhost:11434/api/chat"
    TIMEOUT = 120          # CPU inference can be slow
    NUM_PREDICT = 512      # default max tokens to generate (keeps answers short)
    NUM_CTX = 2048         # context window

    def __init__(self):
        self.model = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
        self.num_predict = int(os.getenv("LOCAL_NUM_PREDICT", 512))

    def __call__(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
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

        # Ollama wraps usage in top-level "prompt_eval_count" / "eval_count"
        return {
            "text": data["message"]["content"],
            "latency_s": round(latency, 3),
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "truncated": truncated,
        }


# ---------------------------------------------------------------------------
# APIModel – Gemini or Groq
# ---------------------------------------------------------------------------

# Cost per million tokens (input / output) for common free-tier models.
# Override via .env: COST_PER_M_IN and COST_PER_M_OUT (USD).
_DEFAULT_COSTS = {
    # provider  : (input $/M, output $/M)
    "gemini": (0.0, 0.0),   # Gemini 1.5 Flash free tier
    "groq":   (0.0, 0.0),   # Groq free tier
}

# Fixed delay between API calls to avoid bursting per-minute rate limits
_API_INTER_CALL_DELAY_S = 2


class APIModel:
    """
    Calls a hosted LLM API (Gemini or Groq) using plain HTTP requests.

    Config (via .env):
        PROVIDER               – "gemini" | "groq"
        API_MODEL              – model name, e.g. "gemini-1.5-flash" or "llama3-8b-8192"
        API_KEY                – provider API key  (never logged)
        API_MAX_OUTPUT_TOKENS  – optional, max output tokens (default 2048)
        COST_PER_M_IN          – optional, USD per million input tokens
        COST_PER_M_OUT         – optional, USD per million output tokens

    Returns a dict with text, latency_s, tokens_in, tokens_out, est_cost_usd, truncated.
    """

    TIMEOUT = 20
    MAX_RETRIES = 2
    NUM_CTX = 2048
    MAX_OUTPUT_TOKENS = 2048

    def __init__(self):
        self.provider = os.getenv("PROVIDER", "gemini").lower()
        self.model = os.getenv("API_MODEL", "")
        # Key is loaded but intentionally never printed/logged
        self._api_key = os.getenv("API_KEY", "")
        self.max_output_tokens = int(os.getenv("API_MAX_OUTPUT_TOKENS", self.MAX_OUTPUT_TOKENS))

        default_in, default_out = _DEFAULT_COSTS.get(self.provider, (0.0, 0.0))
        self.cost_per_m_in = float(os.getenv("COST_PER_M_IN", default_in))
        self.cost_per_m_out = float(os.getenv("COST_PER_M_OUT", default_out))

        if not self.model:
            raise ValueError("API_MODEL must be set in .env")
        if not self._api_key:
            raise ValueError("API_KEY must be set in .env")
        if self.provider not in ("gemini", "groq"):
            raise ValueError(f"Unsupported PROVIDER '{self.provider}'. Use gemini or groq.")

    # --- provider-specific helpers ----------------------------------------

    def _redact(self, text: str) -> str:
        """Replace any occurrence of the API key in `text` with <REDACTED>."""
        return _redact(text, self._api_key)

    def _gemini_call(self, prompt: str) -> tuple[str, int, int]:
        """
        Calls the Gemini generateContent REST endpoint.
        API key is sent in the x-goog-api-key header, never in the URL.
        Returns (text, tokens_in, tokens_out).
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        resp = _post_with_retry(
            url,
            headers={
                "Content-Type": "application/json",
                # Key in a request header; never appears in the URL or logs
                "x-goog-api-key": self._api_key,
            },
            payload=payload,
            timeout=self.TIMEOUT,
            max_retries=self.MAX_RETRIES,
        )
        data = resp.json()
        candidates = data.get("candidates") or [{}]
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        parts = candidate.get("content", {}).get("parts", []) if candidate.get("content") else []
        text = "".join(
            part.get("text", "")
            for part in parts
            if not part.get("thought")
        )

        if finish_reason == "MAX_TOKENS" or not text.strip():
            raise RuntimeError("reply truncated")

        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0) or 0
        candidates_tokens = usage.get("candidatesTokenCount", 0) or 0
        thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0
        tokens_out = candidates_tokens + thoughts_tokens
        return text, tokens_in, tokens_out

    def _groq_call(self, prompt: str) -> tuple[str, int, int]:
        """
        Calls the Groq chat completions endpoint (OpenAI-compatible).
        Returns (text, tokens_in, tokens_out).
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        resp = _post_with_retry(
            url,
            headers={
                "Content-Type": "application/json",
                # API key in Authorization header, never echoed to output
                "Authorization": f"Bearer {self._api_key}",
            },
            payload=payload,
            timeout=self.TIMEOUT,
            max_retries=self.MAX_RETRIES,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        return text, tokens_in, tokens_out

    # --- public interface --------------------------------------------------

    def __call__(self, prompt: str) -> dict:
        # Fixed delay right before each call to avoid bursting per-minute rate limits
        time.sleep(2)

        t0 = time.perf_counter()
        try:
            if self.provider == "gemini":
                text, tokens_in, tokens_out = self._gemini_call(prompt)
            else:
                text, tokens_in, tokens_out = self._groq_call(prompt)
        except RuntimeError as exc:
            # Redact the key before the message propagates to any log or traceback
            safe_msg = self._redact(str(exc))
            raise RuntimeError(f"APIModel ({self.provider}/{self.model}) error: {safe_msg}") from None
        latency = time.perf_counter() - t0

        # Estimate cost (may be 0 for free tiers)
        est_cost = (tokens_in / 1_000_000 * self.cost_per_m_in +
                    tokens_out / 1_000_000 * self.cost_per_m_out)

        return {
            "text": text,
            "latency_s": round(latency, 3),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "est_cost_usd": round(est_cost, 8),
            "truncated": False,
        }
