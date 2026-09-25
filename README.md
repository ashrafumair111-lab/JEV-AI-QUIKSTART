# JEV AI Quickstart

**A production-shaped Python starter kit that gets you from `git clone` to a working Jev-powered AI agent in under five minutes.**

Jev is the "System One" decision model from [TypeSafe AI](https://typesafe.ai): you send it a *state* plus typed questions, and it returns **typed decisions with calibrated confidence** instead of prose. This repository wires Jev into a small, complete agent - triage, confidence gating, LLM drafting, guardrails - in code you can read in one sitting.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Jev](https://img.shields.io/badge/Jev-jev--latest-8b5cf6)
![Tests](https://img.shields.io/badge/tests-29_passing-brightgreen)
![Runs offline](https://img.shields.io/badge/runs-offline_with_no_keys-informational)

> **Disclaimer:** this is an independent, community quickstart. It is **not affiliated with or endorsed by TypeSafe AI**. For the authoritative source of truth always read <https://docs.typesafe.ai>.

---

## Table of contents

- [What is JEV AI?](#what-is-jev-ai)
- [What this repository does](#what-this-repository-does)
- [Prerequisites](#prerequisites)
- [Installation and setup](#installation-and-setup)
- [Quick start usage](#quick-start-usage)
- [Project structure](#project-structure)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Running the tests](#running-the-tests)
- [Next steps](#next-steps)
- [License](#license)

---

## What is JEV AI?

Generative LLMs are great at writing and bad at being trusted with a *decision*. Jev flips that around: you define the question and the allowed answers **before** the call, and Jev returns one of your options, a probability for every option, and a `confidence` score your code can branch on.

| | Jev (System One) | Generative LLM |
| --- | --- | --- |
| **Returns** | Typed decisions + probabilities | Free-form text |
| **Allowed answers** | Defined by you, up front | Open-ended |
| **Great at** | Classify, route, score, gate, verify | Draft, explain, summarise, plan |
| **Needs output parsing?** | No - it is already typed | Almost always |
| **Can still be wrong?** | Yes - which is why confidence exists | Yes |

Jev has three question types (the *primitives*):

| Primitive | You ask | You get back |
| --- | --- | --- |
| **Choice** | "Which option fits?" with a `criteria` dict | `choice`, `probabilities` per option, `confidence` |
| **Score** | "How much, on this ordered scale?" | `score`, `legend`, `probabilities`, `confidence` |
| **Noul** | "Is this statement true?" | `noul` - a single probability between 0 and 1 |

A typical Jev call sends **many independent questions in one request** and your code decides what to do with each answer. That is both cheaper and faster than one call per question.

**Where Jev belongs in an agent**

```
customer message
      |
      v
[ Jev: what is this? ]  -->  typed decision + confidence      <- you are here
      |
      +-- confidence too low? --> your code escalates to a human
      |
      v
[ LLM: write the reply ]  -->  prose
      |
      v
[ Jev: is this reply safe? ]  -->  yes/no probabilities       <- and here
      |
      v
your code decides whether to send, hold, or drop it
```

The model never gets to decide its own authority: permissions, thresholds and side effects stay in **your** code, where they are testable.

---

## What this repository does

It processes one support ticket end to end and prints a full report:

1. **Triage fan-out** - one Jev request answers three questions: which queue owns the ticket (`Choice`), how frustrated the customer is (`Score`), and whether the message is time-sensitive (`Noul`).
2. **Confidence gate** - plain Python code turns that confidence into an action: `auto_draft`, `draft_with_review`, or `escalate_to_human`. An escalated ticket costs one cheap Jev call and **zero** LLM calls.
3. **Drafting** - the reply is written by an OpenAI chat model through LangChain (LCEL), guided by Jev's decision. No `OPENAI_API_KEY`? A deterministic template writer takes over so the demo still runs.
4. **Guardrails** - a second Jev call checks the draft for invented policy, leaked personal data and tone before anything is released.

It ships with two engines and two writers behind protocols, so you can swap any layer for your own:

| Layer | Live implementation | Offline implementation |
| --- | --- | --- |
| Decisions | `JevDecisionEngine` (official `typesafe-sdk`) | `OfflineDecisionEngine` (deterministic keyword rules) |
| Drafting | `LangChainDraftWriter` (OpenAI via LangChain) | `TemplateDraftWriter` (fixed, reviewed sentences) |

`--offline` uses the right-hand column: **no API keys, no network, no spend** - which is the fastest way to see whether the architecture suits you before you sign up for anything.

---

## Prerequisites

| Requirement | Why you need it |
| --- | --- |
| **Python 3.10 or newer** | `typesafe-sdk` and LangChain both require 3.10+. Check yours with `python --version`. |
| **A Jev (TypeSafe) API key** | Needed for *live* mode only. Create one at <https://console.typesafe.ai/keys>. `--offline` runs need no key at all. |
| **An OpenAI API key** | **Optional.** Used only to draft the reply text. Create one at <https://platform.openai.com/api-keys>. Without it the demo falls back to a deterministic template writer. |
| **git** | To clone the repository. |

---

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/ashrafumair111-lab/JEV-AI-QUIKSTART.git
cd JEV-AI-QUIKSTART
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Create your `.env` file

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

### 5. Fill in your keys

```dotenv
# .env
JEV_AI_API_KEY=your-key-from-console.typesafe.ai
OPENAI_API_KEY=your-optional-openai-key
```

`.env` is gitignored. The placeholders that ship in `.env.example` (`your_jev_api_key_here`) are treated as *not configured*, so a fresh clone tells you to set a key instead of firing off a request that is guaranteed to fail.

### 6. Prove it works before spending a single token

```bash
python -m src.main --offline
```

That is the entire setup. Nothing below this line is required for the offline run.

---

## Quick start usage

### Start here: run it with no keys at all

```bash
python -m src.main --offline
```

This is real output from a fresh clone (only trailing whitespace was trimmed):

```text
---------------------- JEV AI Quickstart - OFFLINE mode -----------------------
+------------------------------------------+
| decision engine | offline/canned-rules   |
| draft writer    | template/deterministic |
+------------------------------------------+
-------------------------------- Configuration --------------------------------
+-------------------------------------------------+
| mode               | offline (canned decisions) |
| JEV_AI_API_KEY     | (not set)                  |
| OPENAI_API_KEY     | (not set)                  |
| JEV_AI_MODEL       | jev-latest                 |
| OPENAI_MODEL       | gpt-4o-mini                |
| confidence floor   | 0.50                       |
| auto-act threshold | 0.85                       |
| timeout            | 30s                        |
| log level          | INFO                       |
+-------------------------------------------------+
note: Offline mode: decisions come from the built-in canned engine, not from Jev.
------------------------------ Customer message -------------------------------
Hi, I've been trying to connect my Stripe account for 3 days and the
integration keeps failing. I'm losing sales. Please help ASAP.
-------------------------------- Jev decision ---------------------------------
+---------------------------------------------------------------------------+
| queue           | technical (confidence 0.87)                             |
|   probabilities | technical 0.90 | billing 0.03 | sales 0.03 | other 0.03 |
| frustration     | 0 = Calm and matter-of-fact (confidence 0.40)           |
| urgency         | 0.90 probability the message is time-sensitive          |
| answered by     | offline/canned-rules in 0 ms (offline)                  |
+---------------------------------------------------------------------------+
---------------------- Confidence gate (code, not model) ----------------------
+-----------------------------------------------------------------------------+
| action     | draft_with_review                                              |
| confidence | 0.87                                                           |
| reason     | confidence 0.87 clears the auto-act threshold 0.85, but        |
|            | urgency 0.90 means a human should still look                   |
+-----------------------------------------------------------------------------+
-------------- Drafted reply (template: template/deterministic) ---------------
Hello,

Thank you for the detail in your message, and I am sorry this integration is
not behaving as it should.

I appreciate you taking the time to report it. I have asked an integration
engineer to reproduce the failing call and come back with a fix or a
workaround. Because this is affecting you right now, I am treating it as a
priority.

Kind regards,
The support team
--------------------- Guardrail checks (Jev on the draft) ---------------------
+----------------------------------+
| answers_the_request | 0.92  pass |
| no_personal_data    | 0.97  pass |
| tone_ok             | 0.94  pass |
+----------------------------------+
----------------------------------- Result ------------------------------------
HELD FOR REVIEW: the reply is drafted but a human should approve it
+-------------------------+
| end to end       | 0 ms |
| LLM calls        | 1    |
| Jev input tokens | 0    |
+-------------------------+
```

**What to notice, because this is the whole lesson:**

| Line in the output | Why it matters |
| --- | --- |
| `technical (confidence 0.87)` | Jev returns an option **and** how sure it is - plus the full distribution, so your code can disagree with the winner when two options are close. |
| `urgency 0.90` | "losing sales" and "ASAP" made Jev rate the message time-sensitive, as a probability rather than a yes/no. |
| `draft_with_review` | Confidence cleared the auto-act threshold, but this project's policy says a *very urgent* ticket still gets a human read. That rule lives in `src/pipeline.py`, not in a prompt. |
| `Guardrail checks` | A **second** Jev call inspected the draft - the model that wrote the text is not trusted to approve it. |
| `HELD FOR REVIEW` is not an error | The pipeline is working as designed. Change the gate policy (or the thresholds in `.env`) and the same ticket is released. |
| `LLM calls 1`, `Jev input tokens 0` | The run reports its own cost. Offline tokens are `0` because nothing was billed. |

### Now run it live against Jev

```bash
python -m src.main
```

With a real `JEV_AI_API_KEY` in `.env` the same report appears, but `answered by` shows `jev/jev-latest`, the model route and latency come from the API, `input tokens` is the figure the API reported, and `--verbose` adds the `request_id` you would quote in a support ticket.

### Try the other routing branches

```bash
# A vague message: no queue fits, so it escalates - zero LLM calls, no spend
python -m src.main --offline --ticket "Hey, quick question about something else"

# A very angry customer: high confidence, but the reply is held for a human
python -m src.main --offline --ticket "I was charged twice and need a refund now. This is unacceptable!"

# Your own ticket, live against Jev
python -m src.main --ticket "Your API returns a 500 whenever I POST a customer"

# Machine-readable output for scripts and dashboards
python -m src.main --offline --json
```

### Command-line flags

| Flag | What it does |
| --- | --- |
| `--offline` | Canned decisions and a template writer: no keys, no network, no spend. |
| `--ticket "TEXT"` | Process your own customer message instead of the sample. |
| `--json` | Print the complete run record as JSON (pipe it into `jq`). |
| `--env-file PATH` | Load a different environment file. |
| `-v`, `--verbose` | Debug logging, request ids and token counts. |
| `--version` | Print the version and exit. |
| `-h`, `--help` | Full help text. |

Both launch styles work: `python -m src.main ...` and `python src/main.py ...`.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Ran successfully - *any* routing decision, escalation included, is a success. |
| `2` | Configuration problem: missing or invalid setting. |
| `3` | A required package is not installed. |
| `4` | Jev or the LLM provider returned an error. |
| `130` | Interrupted with `Ctrl+C`. |

A missing key therefore fails cleanly in CI instead of dumping a traceback:

```text
Error: No Jev API key found, so live mode cannot start.

How to fix it:
1. Create a key at https://console.typesafe.ai/keys
2. Copy .env.example to .env and set JEV_AI_API_KEY=<your key>
3. Or run this demo with --offline to see the whole pipeline working with no
credentials and no network calls.
```

---

## Project structure

```text
JEV-AI-QUIKSTART/
├── .env.example              # Template for your keys - copy it to .env
├── .gitignore                # Ignores .env, .venv, caches and OS cruft
├── LICENSE                   # MIT
├── README.md                 # You are here
├── requirements.txt          # Jev SDK, LangChain, OpenAI, dotenv, pydantic, rich
├── src/
│   ├── __init__.py           # Package docstring and version
│   ├── config.py             # Loads .env, validates every setting (Pydantic)
│   ├── errors.py             # User-facing exception hierarchy
│   ├── decisions.py          # Jev layer: Choice/Score/Noul answers + engines
│   ├── writer.py             # Generative layer: LangChain writer + fallback
│   ├── pipeline.py           # Triage -> gate -> draft -> guardrail
│   └── main.py               # CLI: flags, rich report, exit codes
└── tests/
    ├── __init__.py
    └── test_quickstart.py    # 29 tests - all offline, no keys, no network
```

The two files worth reading first:

* **`src/main.py`** - the entry point. Run it, read the report, then read the code beside it.
* **`src/pipeline.py`** (`_gate`) - the confidence gate, i.e. where *your* risk policy belongs.

Dependencies point one way only: `main` -> `pipeline` -> (`decisions`, `writer`) -> `config`. The two middle layers exchange plain dataclasses, so either can be replaced without touching the other.

### Design decisions worth stealing

| Decision | Why |
| --- | --- |
| Protocols instead of base classes | `DecisionEngine` and `DraftWriter` are `Protocol`s, so a ten-line test double satisfies them with no inheritance. |
| SDK imports are lazy | `--offline` and the whole test suite run on a machine without `typesafe-sdk`, LangChain or a network connection. |
| Placeholder keys count as missing | `.env.example` ships fake keys; treating them as unset turns a baffling `401` into "set your key". |
| Constructor arguments are probed | `inspect.signature` filters the SDK and LangChain options, so a renamed keyword in a future release cannot break your clone. |
| One ticket, one record | `TicketOutcome` carries the decision, the gate, the draft and the guardrails - easy to log, assert on, or serialise with `--json`. |
| Cost is printed, not implied | The report shows LLM calls and Jev input tokens, so the value of gating on confidence is visible rather than theoretical. |

---

## How it works

Four moves, only two of which touch a model:

| Step | Model call | Where it lives |
| --- | --- | --- |
| 1. **Triage** | one Jev request, three questions | `src/decisions.py` -> `JevDecisionEngine.triage` |
| 2. **Gate** | none - pure Python | `src/pipeline.py` -> `TriagePipeline._gate` |
| 3. **Draft** | one LLM call, *skipped entirely on escalation* | `src/writer.py` -> `LangChainDraftWriter.draft` |
| 4. **Guardrail** | one Jev request, three nouls | `src/decisions.py` -> `JevDecisionEngine.guard` |

### Step 1 - ask everything at once

```python
response = client.system_one(
    state=ticket,                       # the raw customer message
    questions={
        "department": Choice(
            instructions="Which team should handle this ticket?",
            criteria=DEPARTMENTS,       # the allowed answers, defined by you
        ),
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=FRUSTRATION_LEVELS,
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity..."
        ),
    },
)

answer = response.answers["department"]
print(answer.choice, answer.confidence)       # e.g. "technical 0.87"
print(answer.probabilities)                   # every option, not just the winner
```

Three independent questions travel in one request and are evaluated in parallel, so a fourth question barely changes the latency. That is what makes it reasonable to ask *speculative* questions your code may never read - see the [speculative fan-out pattern](https://docs.typesafe.ai/patterns/fan-out).

### Step 2 - let code, not the model, decide what happens next

```python
if decision.department.choice == "other":       escalate_to_human(...)
elif confidence < FLOOR:                        escalate_to_human(...)
elif decision.frustration.score >= 2:           draft_then_hold_for_review(...)
elif confidence >= AUTO_ACT and urgency < 0.85: draft_and_release(...)
else:                                           draft_then_hold_for_review(...)
```

Confidence says *how sure* Jev is. Policy says *how much that certainty is worth*, and policy is yours: a read-only action at 0.6 may be fine while an irreversible one at 0.95 may not. The escalation branch makes **zero** model calls, which is where the savings come from. See [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing) and [confidence](https://docs.typesafe.ai/confidence).

### Step 3 - the LLM writes, steered by the decision

The prompt receives the queue, the frustration label and the urgency probability, and is instructed to invent nothing:

```text
You are the first-line support agent for the {queue} team.

A decision model has already classified this ticket for you:
- routing queue: {queue}
- customer frustration level: {frustration_label}
- probability the message is urgent: {urgency:.2f}
```

Change the decision and the same prompt produces a different reply. The model writes; it does not decide.

### Step 4 - Jev checks the draft before a customer sees it

```python
state = {"customer_message": ticket, "draft_reply": draft}
questions = {
    "answers_the_request": Noul(instructions="...and invents no policies, prices, refunds or timelines."),
    "no_personal_data":    Noul(instructions="...contains no names, emails, phone numbers, card numbers."),
    "tone_ok":             Noul(instructions="...calm, professional, appropriate for the frustration level."),
}
```

Every check comes back as a probability rather than a verdict, and `GUARDRAIL_MIN_NOUL` in `src/decisions.py` is the single line you move to trade safety against how much a human has to review.

---

## Configuration reference

Everything is optional except a Jev key in live mode. Copy `.env.example` to `.env` and edit it.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JEV_AI_API_KEY` | live mode | - | Your Jev credential. The SDK's own name, `TYPESAFE_API_KEY`, also works; `JEV_AI_API_KEY` wins if both are set. |
| `OPENAI_API_KEY` | no | - | Drafts the reply text. Without it the deterministic template writer takes over. |
| `JEV_AI_MODEL` | no | `jev-latest` | Model route. Other routes include `jev-preview` and `jev-1.13.0`. |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Any chat model your key can access. |
| `JEV_AI_CONFIDENCE_FLOOR` | no | `0.50` | Below this the ticket escalates: no draft, no LLM spend. |
| `JEV_AI_AUTO_ACT_THRESHOLD` | no | `0.85` | At or above this - and subject to the other gate rules - the reply is released without review. |
| `JEV_AI_OFFLINE` | no | `false` | Same effect as the `--offline` flag. |
| `JEV_AI_REQUEST_TIMEOUT_SECONDS` | no | `30` | Per-request timeout for both Jev and OpenAI. |
| `JEV_AI_LOG_LEVEL` | no | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. Also forwarded to the SDK's `TYPESAFE_LOG_LEVEL`. |
| `TYPESAFE_BASE_URL` | no | `https://api.typesafe.ai` | Point the SDK at a gateway, proxy or local mock. |

Three conveniences you did not have to ask for:

* `JEV_AI_*` values are mirrored into the `TYPESAFE_*` names the official SDK reads, so the SDK is correctly configured *before* it is imported.
* Values that are empty, whitespace, or still contain `your_`, `key_here`, `changeme` or `xxxx` count as **not set**.
* `JEV_AI_CONFIDENCE_FLOOR=high` fails at start-up with "must be a number" instead of failing somewhere deep inside a request.

---

## Troubleshooting

| Symptom | What it means | Fix |
| --- | --- | --- |
| `Error: No Jev API key found...` (exit `2`) | `.env` is missing, or the key is still the placeholder | `Copy-Item .env.example .env` and set `JEV_AI_API_KEY`, or run with `--offline` |
| `401 Cannot authenticate with the server` | Jev rejected the credential | Issue a fresh key at <https://console.typesafe.ai/keys> |
| `403` / `TypeSafePermissionDeniedError` | The key is valid but not allowed to use that model route | Try `JEV_AI_MODEL=jev-latest`, or check the key's permissions |
| `429` / `TypeSafeRateLimitError` | Rate limit or quota reached | Wait a few seconds and retry; the SDK honours a retry policy |
| `TypeSafeAPITimeoutError` | The network is slow | Raise `JEV_AI_REQUEST_TIMEOUT_SECONDS` |
| `The official Jev SDK (typesafe-sdk) is not installed` | Dependencies are missing from *this* interpreter | `python -m pip install -r requirements.txt`, and check that the venv is active |
| The report says `template/deterministic` | No `OPENAI_API_KEY`, or you are in offline mode | Set the key, or keep the fallback deliberately |
| No colour or boxes in the output | `rich` is not installed | `python -m pip install rich` (purely cosmetic) |
| `ModuleNotFoundError: No module named 'src'` | You are not in the repository root | `cd` to the root and prefer `python -m src.main` |
| You want to see the raw failure | - | Re-run with `-v` for debug logging including `request_id`s |

Your keys never appear in the output: they are held as `SecretStr` and rendered as `jev_...7890`, and the test suite asserts that they cannot leak into a report.

---

## Running the tests

```bash
# stdlib only - no extra dependency
python -m unittest discover -s tests -t . -v

# or, if you prefer pytest
python -m pytest tests -q
```

All 29 tests run offline - no keys, no network, no spend - and finish in well under a second:

```text
Ran 29 tests in 0.212s

OK
```

What they pin down:

| Area | Examples |
| --- | --- |
| Configuration | placeholder keys are ignored, the `TYPESAFE_API_KEY` alias works, inverted thresholds are rejected, secrets never reach a report |
| Offline engine | routing, unroutable tickets, the urgency noul, guardrail detection of personal data |
| Routing policy | confident -> released, uncertain -> escalated with **zero** LLM calls, angry customer -> held, urgent -> held, failed guardrail -> held |
| CLI | the full report prints, `--json` parses, a custom `--ticket` is honoured, a missing key returns `2` |
| SDK resilience | only the constructor arguments an SDK actually accepts are forwarded |

Because the pipeline depends on protocols rather than concrete classes, the routing tests use a scripted engine and a spy writer - see the test doubles at the top of `tests/test_quickstart.py` if you want to reuse them.

---

## Next steps

Where to go once the demo makes sense:

* **Read the official docs** - <https://docs.typesafe.ai>. Start with [Primitives](https://docs.typesafe.ai/primitives), then [Confidence](https://docs.typesafe.ai/confidence), then the [patterns](https://docs.typesafe.ai/patterns).
* **Get a key and run live** - <https://console.typesafe.ai/keys>.
* **Go async** - `AsyncTypeSafeClient` ships in the same SDK; swap it into `src/decisions.py` to process tickets concurrently.
* **Use richer criteria** - `Choice`, `Score` and `Noul` all accept structured instructions and `NoulCriteria(true=..., false=...)` in place of a one-line sentence, which helps on genuinely hard judgments.
* **Split complex judgments** - several atomic `Score` questions combined with weights you control beat one fuzzy question ("composite scoring").
* **Read the SDK source** - <https://github.com/typesafe-ai/typesafe-sdk-python>.
* **Bring your coding agent along** - TypeSafe publishes an agent skill: `npx skills add typesafe-ai/skills --skill typesafe-ai`.

Ideas that fit this skeleton with a few lines of your own code:

| Idea | The change |
| --- | --- |
| Model routing | Add a `Choice` question listing the models your account can call, with a `Score` for cost and quality, then route on the winner |
| Multi-label tickets | Ask several `Noul` questions ("is this a bug?", "is this about billing?") instead of one `Choice` |
| Retrieval quality | `Score` each retrieved passage for real relevance and drop the ones that only matched on embeddings |
| Human-review queue | Persist every `HELD_FOR_REVIEW` outcome and let reviewers label it - that data is what you tune the thresholds with |
| Second pass | When a guardrail fails, re-draft once with the failed check quoted back to the writer before giving up on it |

---

## License

MIT - see [LICENSE](LICENSE).

This project is an independent community quickstart and is **not affiliated with, endorsed by or supported by TypeSafe AI**. "Jev" and "TypeSafe" belong to their respective owners. The offline decision engine in `src/decisions.py` exists for demos and tests - it is not the Jev model and should never be used as a substitute for it in production.

Built to be read. If you got a Jev agent running in under five minutes, this repository did its job - now go and change `src/pipeline.py` to match your own risk policy.

