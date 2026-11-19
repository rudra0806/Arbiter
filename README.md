# Arbiter

A local/API routing layer that answers checkable questions with a small local
model (Ollama, `qwen2.5:1.5b`) when it's confident, and escalates to a
stronger API model (Groq, `openai/gpt-oss-120b`) only when it isn't — using
the local model's own self-consistency as the signal, not a trained
classifier guessing from the question's wording. Named for what it actually
does: it doesn't guess in advance which questions are hard, it waits for the
local model to show disagreement with itself, then arbitrates by calling in
the stronger model only on the cases that need it.

**Who this is for:** scripts and applications that call an LLM repeatedly and
are bounded by a provider's free-tier rate limits (requests per day/minute).
It is *not* meant to replace a normal chatbot for a single, one-off question
— the savings only show up over many automated calls, not one typed by a
human. The included GUI is a demo/inspection tool for seeing the routing
decision happen, not the primary way this is meant to be used.

---

## Scope — read this first

This project is validated for **checkable-answer tasks only**: math, facts,
code output, extraction, and classification — anything with one verifiable
correct answer, because that's what let us build an automated benchmark at
all.

It is **not validated** for open-ended chat, creative writing, or document
generation (tested live: asking it to draft a PRD returns a wall of plain
text, since no file-generation capability exists in the gateway itself).

---

## Final results

Run with `python scripts/results_summary.py`. These numbers come straight
from that script, reading the saved benchmark result files — nothing here is
hand-typed.

| Split | Tasks | Always-Local | Always-API | **Verifier** | % Routed Local | API Cost |
|---|---|---|---|---|---|---|
| dev   | 100 | 72.0% | 100.0% | **91.0%** | 70.0% | $0.00808 |
| test  | 50  | 86.0% | 100.0% | **92.0%** | 76.0% | $0.00316 |
| **final** | **50** | **92.0%** | **100.0%** | **98.0%** | **46.0%** | **$0.00889** |

**`final` is the headline number.** It's the one-time held-out split —
never looked at while designing or tuning anything — evaluated exactly once,
with no retries or adjustments afterward. The verifier recovers almost all
of the gap between always-local and always-API, while still keeping under
half the calls on the free, zero-cost local model.

`%Local` varies by split (46–76%) because it adapts to how often the local
model actually agrees with itself on that particular batch of questions —
it isn't a fixed dial, it's a live measurement of task difficulty.

---

## How it works

1. A question comes in.
2. **Deterministic calculator check first.** A small set of question patterns
   (currently: "day X days from [weekday]", in several phrasings) are
   answered directly in plain Python, bypassing the LLM entirely, because
   they're 100% computable and the local model has a known blind spot here
   (see "Known limitation" below).
3. **If no calculator matches:** the local model answers the same question
   **3 times** at temperature 0.7 (not 0 — it needs room to vary).
4. **If all 3 answers agree** (after stripping markdown/LaTeX noise and
   comparing the extracted final answer), that answer is trusted. No API
   call, $0 cost.
5. **If they disagree,** the question is escalated to the API model once,
   and that answer is returned.

No trained model is part of the final design — see "Why no classifier"
below for why that was a deliberate choice, not an oversight.

---

## The journey (why the design is this simple)

Several more complex approaches were tried and honestly measured before
landing on the plain self-consistency verifier above. This table is here on
purpose — it's the evidence that the final design was *earned*, not a first
guess.

| Version | Approach | Validation | Result | Verdict |
|---|---|---|---|---|
| v1 | 9 hand-written keyword/regex rules on the prompt text | Held-out 50-task split, never seen while writing rules | 94.0% accuracy vs. 94.1% for **random routing** with the same local/API split | **Failed.** No measurable skill — overfit to one narrow phrasing style used while writing the rules. |
| v2/v3 | TF-IDF text features → logistic regression, later combined with a (leaked, then fixed) self-consistency feature | Grouped 5-fold CV by phrasing template, to catch the same overfitting v1 had | Text-only: 0.800 acc / 0.654 AUC. With a correctly-computed self-consistency feature added: 0.757 AUC | **Modest, genuine signal** — beat a naive "route by difficulty label" baseline, but never beat the plain verifier below. |
| Hybrid | Let the trained classifier override the verifier's "escalate" decision in disagreement cases | Same splits as the plain verifier | In every disagreement case tested, escalating to the API was correct 100% of the time; the classifier's overrides were right less than half the time it tried to override | **Worse than plain verifier.** Discarded. |
| **Verifier (final)** | 3-sample self-consistency, no training, no classifier | dev / test / **final** (one-time) | See results table above | **Kept.** Simpler, no training/maintenance burden, and empirically the strongest of everything tried. |

**One important caveat about the self-consistency feature itself:** the
first version of it was computed by checking each sampled reply against the
*true answer key* — which is not information a live router has access to
for a new question. That version's AUC (0.935) was discarded as leakage.
The version actually used compares the 3 samples **to each other only**,
with no access to the correct answer — that's what produces the real,
reportable 0.757 AUC and the verifier's final numbers above.

---

## Known limitation: day-of-week arithmetic (found via live testing, now fixed)

The benchmark's automated grading didn't catch this — it was found by
manually asking the live CLI a battery of fresh questions after the
benchmark was already "done."

The local model answers questions like *"If today is Wednesday, what day is
it 12 days from now?"* **confidently wrong, with full 3/3 self-agreement,**
roughly half the time — because for small day-counts it tends to skip
straight to an answer instead of showing the modular-arithmetic ("12 ÷ 7 =
1 remainder 5") reasoning that, when shown, was reliably correct. Since the
error is systematic (the same wrong guess every time) rather than random,
**self-consistency cannot detect it** — that's a structural limit of the
technique, not a bug in this implementation.

Tested: 10 day-offset questions, 5 wrong, all 5 with full agreement and zero
escalation.

**Fix:** a plain-Python calculator (`try_day_offset_calculator` in
`gateway/ask.py`) detects several phrasings of this exact question type and
computes the answer directly — no model call, no training needed, 100%
correct by construction.

**Broader stress test after the fix:** 17 additional live questions across
7 other categories (multi-step money math, geometry, pure logic, unit
conversion, probability, number sequences, leap years, clock duration) —
**17/17 correct**, with appropriate escalation on the genuinely harder ones.
This suggests the day-of-week issue is narrow rather than representative of
a wider problem, though it was never formally benchmarked at the same scale
as the main 481-task set — only live-spot-checked.

---

## Why no classifier made the final cut

It's tempting to assume "more machine learning = better," but the evidence
pointed the other way here: the plain verifier (no training at all) scored
higher on every held-out split than the trained classifier did on its own,
and adding the classifier on top of the verifier (the hybrid) made results
*worse*, because every time it overrode "escalate" it was overriding a
decision that was correct 100% of the time in testing. The honest
conclusion: for this problem, checking the model's own answer behavior
directly was more informative than trying to predict that behavior from the
question's text.

---

## Reliability features

Built while debugging real free-tier API problems during development, not
designed in the abstract:

- **Automatic retry with exponential backoff** (2s → 4s → 8s) on failed API
  calls, so a single rate-limit hit or timeout doesn't stop a run.
- **Clear, redacted error messages** — API keys are never printed in logs or
  error text, even inside exception chains, but the real failure reason
  (429 quota exceeded, 503 server overloaded, read timeout) is always
  visible, so failures are diagnosable instead of showing a bare `None`.
- **Switched providers mid-project** (Gemini → Groq) after discovering
  Gemini's free tier in this project's Google Cloud project allowed only
  20 requests/day for the model in use — Groq's `openai/gpt-oss-120b` offers
  1,000 requests/day on the free tier, which the full 482-task benchmark
  completed in one clean pass with zero errors.
- **Caching**: every model reply is cached by (model, prompt), so interrupted
  or re-run benchmarks never repeat a call that already succeeded.

---

## Benchmark corrections log

Found and fixed during development, before any final evaluation was run on
the affected data — kept here for transparency rather than quietly
corrected:

- **Task 161** had a wrong answer key (620 instead of the correct 625) due
  to an integer-division bug in the task generator.
- **Comma-formatted numbers** (e.g. a model answering "2,688" where the key
  only listed "2688") were initially graded wrong; the checker now accepts
  both forms for large numbers.
- **LaTeX and markdown noise** (`\(...\)`, `{,}` digit grouping, `**bold**`,
  curly quotes, em-dashes) in model replies were breaking exact-match
  checking; a normalization pass was added to all checkers.
- **Task 207** was retired (`"invalid": true`) for genuinely ambiguous
  wording — "the cap was $19" reads as a spending limit, but its answer key
  assumed it meant an additional charge. The model's disagreement with the
  key was correct; the task itself was flawed.
- **One API truncation bug**: `maxOutputTokens` was initially set too low
  for a "thinking" model, cutting a reply off at a single `{` character.
  Fixed by raising the limit and treating a `MAX_TOKENS` finish reason as an
  error rather than a gradable (and cacheable) answer.

---

## Setup

### 1. Local model (Ollama)
```
# install from ollama.com, then:
ollama pull qwen2.5:1.5b
```

### 2. API model (Groq)
Get a free key at console.groq.com, then create `.env` from `.env.example`:
```
PROVIDER=groq
API_MODEL=openai/gpt-oss-120b
API_KEY=<your key>
LOCAL_MODEL=qwen2.5:1.5b
LOCAL_NUM_PREDICT=512
```

### 3. Install dependencies
```
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
pip install fastapi uvicorn
```

### 4. Use it

**Command line, one question at a time:**
```
python gateway/ask.py "What is 17 times 23?"
```

**As a local HTTP service:**
```
python -m uvicorn gateway.server:app --port 8000
```
Then `POST http://localhost:8000/ask` with `{"prompt": "..."}`.

**GUI (demo/inspection tool):** with the server above running, open
`gui/index.html` directly in Chrome or Edge. Includes optional voice input
via the browser's built-in speech recognition — note that specific feature
sends audio to a cloud transcription service, unlike every other part of
this tool, which runs fully on your own machine or against your own API key.

### 5. Reproduce the benchmark
```
python -m benchmark.run_eval --model local
python -m benchmark.run_eval --model api
python scripts/collect_self_consistency.py        # dev + pool
python scripts/collect_self_consistency.py test
python scripts/collect_self_consistency.py final
python scripts/eval_verifier.py --split final --n-samples 3   # run once
python scripts/results_summary.py
```

---

## Checker types reference

For anyone adding tasks to `benchmark/tasks.json`:

| Type | Passes when … |
|---|---|
| `contains` | Any answer substring appears anywhere in the reply |
| `word` | Any answer appears as a whole word (word-boundary match) |
| `exact` | Reply (trimmed, lowercased, trailing period/quote stripped) equals an answer |
| `json` | Reply parses as JSON with all required key-value pairs |
| `wordcount` | Reply's word count equals `answers[0]` |
| `comma_count` | Number of comma-separated items equals `answers[0]` |
| `code_exec` | Extracted Python code passes all `{call, expect}` test cases, run in a subprocess |

All checkers normalize text first (strip markdown bold, LaTeX delimiters,
curly quotes, and digit-grouping) before comparing — see the benchmark
corrections log above for why.

---

## Repository structure

```
gateway/
  ask.py              # the actual application: CLI + verifier + calculator
  server.py           # FastAPI wrapper around ask.py
  models.py           # LocalModel / APIModel, retries, redaction
  router.py           # v1 rule-based router (retired, kept for history)
  router_model*.pkl    # v2/v3 trained classifier (not used in final design)
benchmark/
  tasks.json          # 482 tasks across 4 splits: dev, pool, test, final
  checkers.py         # grading logic, with text-normalization fixes
  run_eval.py         # benchmark runner
  simulate_router.py  # offline router strategy comparison
gui/
  index.html          # local demo/inspection page
scripts/
  collect_self_consistency.py
  eval_verifier.py
  eval_hybrid.py
  train_router.py / train_router_v3.py   # ML experiments (history)
  results_summary.py
results/              # saved model outputs, one file per run
```

---

## Honest limitations, in one place

- Validated only for checkable-answer tasks, not general chat or document
  generation.
- Self-consistency cannot catch systematic (non-random) reasoning errors —
  one instance found and patched with a deterministic calculator; others may
  exist in categories not yet stress-tested.
- The `final` split (50 tasks) is small; its 98.0% figure, while honestly
  measured, carries real sampling noise at that size.
- Voice input depends on a cloud speech-recognition service, unlike the rest
  of the local pipeline.
- Latency for a locally-resolved answer is roughly 3× a single model call
  (three samples), typically 10–25 seconds on modest CPU hardware — a
  deliberate trade of speed for the free, unlimited local tier's
  reliability.