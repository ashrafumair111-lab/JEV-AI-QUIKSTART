"""The decision layer: Jev (TypeSafe "System One") behind a small protocol.

Jev does not write prose. You hand it a *state* and a set of *typed questions*
(``Choice``, ``Score``, ``Noul``) and it returns the selected option, a full
probability distribution and a calibrated ``confidence``. This module converts
that response into plain Python dataclasses, so nothing downstream depends on
the shape of the SDK's own models.

Two engines implement :class:`DecisionEngine`:

``JevDecisionEngine``
    The real thing - the official ``typesafe-sdk`` talking to the Jev API.
``OfflineDecisionEngine``
    A deterministic keyword stand-in used by ``--offline`` and the test suite.
    It mimics Jev's *shape* (typed answers plus confidence), never its quality.

Docs: https://docs.typesafe.ai/primitives
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Protocol

from src.config import Settings
from src.errors import MissingDependencyError, UpstreamError

__all__ = [
    "DEPARTMENTS",
    "GUARDRAIL_MIN_NOUL",
    "TRIAGE_QUESTION_KEYS",
    "UNROUTABLE_DEPARTMENT",
    "ChoiceAnswer",
    "DecisionEngine",
    "GuardrailReport",
    "JevDecisionEngine",
    "NoulAnswer",
    "OfflineDecisionEngine",
    "ScoreAnswer",
    "TriageDecision",
    "create_decision_engine",
]

LOGGER: Final[logging.Logger] = logging.getLogger("jevai.decisions")

#: Keys of the three questions asked in the triage call (one request, three
#: judgments - the "speculative fan-out" pattern).
TRIAGE_QUESTION_KEYS: Final[tuple[str, ...]] = ("department", "frustration", "is_urgent")

#: Keys of the questions asked about a drafted reply before it is sent.
GUARDRAIL_QUESTION_KEYS: Final[tuple[str, ...]] = (
    "answers_the_request",
    "no_personal_data",
    "tone_ok",
)

#: A guardrail question passes when its noul - the probability that the
#: statement is true - reaches this value. Tune it against your own data:
#: https://docs.typesafe.ai/confidence
GUARDRAIL_MIN_NOUL: Final[float] = 0.70

#: Option used when a ticket matches no queue this demo can route to.
UNROUTABLE_DEPARTMENT: Final[str] = "other"

#: The queues this demo routes to, sent to Jev as the Choice criteria.
DEPARTMENTS: Final[dict[str, str]] = {
    "billing": "Payment, invoice, refund or subscription problems",
    "technical": "Bugs, integration failures, API errors and outages",
    "sales": "Pricing, plans, licensing and commercial questions",
    UNROUTABLE_DEPARTMENT: "Does not belong to any of the queues above",
}

#: How frustrated the customer sounds, sent to Jev as the Score legend.
FRUSTRATION_LEVELS: Final[tuple[str, ...]] = (
    "Calm and matter-of-fact",
    "Frustrated but civil",
    "Very angry or threatening to leave",
)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """Answer to a ``Choice`` question: one option plus the full distribution.

    Attributes:
        choice: The selected option, guaranteed to be one of the offered keys.
        confidence: Calibrated certainty in ``[0.0, 1.0]``.
        probabilities: Probability for *every* option, so code can second-guess
            the argmax when two options are nearly tied.
    """

    choice: str
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """Answer to a ``Score`` question: a weighted mean over ordered levels."""

    score: float
    confidence: float
    legend: Mapping[str, str] = field(default_factory=dict)
    probabilities: Mapping[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Legend entry for the selected level, or ``""`` when unknown."""
        return self.legend.get(str(int(round(self.score))), "")


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """Answer to a ``Noul`` question: the probability that a statement is true."""

    noul: float


@dataclass(frozen=True, slots=True)
class TriageDecision:
    """Everything Jev told us about one ticket, plus the call's bookkeeping.

    Attributes:
        department: Which queue owns the ticket.
        frustration: How upset the customer sounds.
        is_urgent: Probability that the message is time-sensitive.
        model: Model route that answered, as reported by the API.
        request_id: Provider request id, handy when asking for support.
        input_tokens: Billable input tokens reported by the API.
        source: ``"jev"`` for the live API, ``"offline"`` for the canned engine.
        latency_ms: Wall-clock time of the triage call.
    """

    department: ChoiceAnswer
    frustration: ScoreAnswer
    is_urgent: NoulAnswer
    model: str
    request_id: str | None = None
    input_tokens: int | None = None
    source: str = "jev"
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    """The result of asking Jev to check a drafted reply before it is sent.

    Attributes:
        checks: One noul answer per question in :data:`GUARDRAIL_QUESTION_KEYS`.
        failures: Keys whose noul fell below :data:`GUARDRAIL_MIN_NOUL`.
        model: Model route that answered.
        request_id: Provider request id.
        input_tokens: Billable input tokens reported by the API.
        source: ``"jev"`` for the live API, ``"offline"`` for the canned engine.
        latency_ms: Wall-clock time of the guardrail call.
    """

    checks: Mapping[str, NoulAnswer]
    failures: tuple[str, ...]
    model: str
    request_id: str | None = None
    input_tokens: int | None = None
    source: str = "jev"
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        """True when at least one guardrail question failed."""
        return bool(self.failures)


class DecisionEngine(Protocol):
    """The two judgments the pipeline needs, and nothing more.

    Any object with these two methods can drive the pipeline - the real Jev
    engine, the offline stand-in, or your own mock in a unit test.
    """

    @property
    def name(self) -> str:
        """Short human-readable identifier, e.g. ``"jev/jev-latest"``."""
        ...

    def triage(self, ticket: str) -> TriageDecision:
        """Decide which queue owns ``ticket``, how upset the customer is and
        whether the message is urgent."""
        ...

    def guard(self, ticket: str, draft: str, decision: TriageDecision) -> GuardrailReport:
        """Check ``draft`` against ``ticket`` before a human ever sees it."""
        ...


# ---------------------------------------------------------------------------
# Tolerant parsing helpers.
#
# The SDK returns rich pydantic models today, but the API is versioned and
# forward-compatible: unknown fields are ignored and answer kinds may be added.
# These helpers read attributes *or* dictionary keys so a minor SDK change
# degrades into a clear message instead of an AttributeError deep in the stack.
# ---------------------------------------------------------------------------
def _field(answer: Any, name: str, default: Any = None) -> Any:
    """Read a named field from an SDK model, a ``dict`` or any object."""
    if isinstance(answer, Mapping):
        return answer.get(name, default)
    return getattr(answer, name, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to ``float``, tolerating ``None`` and junk strings."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_probabilities(value: Any) -> dict[str, float]:
    """Normalise a probabilities mapping, dropping anything non-numeric."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _as_float(item) for key, item in value.items()}


def _usage_tokens(usage: Any) -> int | None:
    """Read ``input_tokens`` from the usage object, if the API reported it."""
    tokens = _field(usage, "input_tokens", None)
    if tokens is None:
        return None
    try:
        return int(tokens)
    except (TypeError, ValueError):
        return None


def _parse_choice(key: str, answer: Any) -> ChoiceAnswer:
    """Convert one SDK ``Choice`` answer into a :class:`ChoiceAnswer`."""
    if answer is None:
        raise UpstreamError(
            f"Jev returned no answer for the {key!r} question.",
            hint="The response did not contain this key - check the questions "
            "defined in src/decisions.py against https://docs.typesafe.ai/primitives",
        )
    choice = _field(answer, "choice", None)
    if not isinstance(choice, str) or not choice:
        raise UpstreamError(f"Jev returned an unusable {key!r} answer: {answer!r}")
    return ChoiceAnswer(
        choice=choice,
        confidence=_as_float(_field(answer, "confidence")),
        probabilities=_as_probabilities(_field(answer, "probabilities")),
    )


def _parse_score(key: str, answer: Any) -> ScoreAnswer:
    """Convert one SDK ``Score`` answer into a :class:`ScoreAnswer`."""
    if answer is None:
        raise UpstreamError(
            f"Jev returned no answer for the {key!r} question.",
            hint="The response did not contain this key - check the questions "
            "defined in src/decisions.py against https://docs.typesafe.ai/primitives",
        )
    legend = _field(answer, "legend", {})
    return ScoreAnswer(
        score=_as_float(_field(answer, "score")),
        confidence=_as_float(_field(answer, "confidence")),
        legend={str(k): str(v) for k, v in legend.items()}
        if isinstance(legend, Mapping)
        else {},
        probabilities=_as_probabilities(_field(answer, "probabilities")),
    )


def _parse_noul(key: str, answer: Any) -> NoulAnswer:
    """Convert one SDK ``Noul`` answer into a :class:`NoulAnswer`."""
    if answer is None:
        raise UpstreamError(
            f"Jev returned no answer for the {key!r} question.",
            hint="The response did not contain this key - check the questions "
            "defined in src/decisions.py against https://docs.typesafe.ai/primitives",
        )
    return NoulAnswer(noul=_as_float(_field(answer, "noul")))


def _as_upstream_error(exc: Exception, context: str) -> UpstreamError:
    """Translate an SDK or network failure into an actionable message.

    The SDK exposes pydantic-style exceptions with a ``status_code``; anything
    without one is treated as a transport problem.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()

    if status == 401 or "auth" in lowered or "unauthor" in lowered:
        hint = (
            "The Jev API rejected the credential. Compare JEV_AI_API_KEY in your "
            ".env with the key at https://console.typesafe.ai/keys"
        )
    elif status == 429 or "rate" in lowered or "quota" in lowered:
        hint = "The Jev API rate limit was hit. Wait a few seconds and retry."
    elif status == 403 or "permissiondenied" in lowered:
        hint = (
            "The Jev API key was accepted but is not allowed to use this model. "
            "Check the key at https://console.typesafe.ai/keys"
        )
    elif status in {400, 422} or "badrequest" in lowered or "unprocessable" in lowered:
        hint = (
            "Jev rejected the request body. Check the state and the questions in "
            "src/decisions.py against https://docs.typesafe.ai/primitives"
        )
    elif "timeout" in lowered or "timed out" in lowered:
        hint = (
            "Jev did not answer in time. Raise JEV_AI_REQUEST_TIMEOUT_SECONDS "
            "in .env, or retry."
        )
    elif "connect" in lowered or "dns" in lowered or "ssl" in lowered:
        hint = (
            "The Jev API could not be reached. Check your network, proxy and "
            "TYPESAFE_BASE_URL, or run with --offline."
        )
    else:
        hint = (
            "Retry; if it persists check https://docs.typesafe.ai and the SDK "
            "release notes for a schema change."
        )
    return UpstreamError(f"Jev request failed during {context}: {name}: {text}", hint=hint)


class JevDecisionEngine:
    """Asks the real Jev model, through the official ``typesafe-sdk``.

    The SDK is imported lazily (inside :meth:`_load_sdk`), so ``--offline`` mode
    and the test suite keep working on a machine where ``typesafe-sdk`` is not
    installed and there is no network access.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sdk: Any | None = None
        self._client: Any | None = None

    @property
    def name(self) -> str:
        """Human-readable identifier used in logs and in the CLI header."""
        return f"jev/{self._settings.jev_model}"

    # -- SDK plumbing ---------------------------------------------------------
    def _load_sdk(self) -> Any:
        """Import ``typesafe_sdk`` on first use.

        Raises:
            MissingDependencyError: If the package is not installed.
        """
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("typesafe_sdk")
            except ImportError as exc:  # pragma: no cover - depends on the env
                raise MissingDependencyError(
                    "The official Jev SDK (typesafe-sdk) is not installed.",
                    hint=(
                        "Install the project dependencies:\n"
                        "    python -m pip install -r requirements.txt\n"
                        "or run this demo without credentials using --offline."
                    ),
                ) from exc
        return self._sdk

    def _get_client(self) -> Any:
        """Create (once) and return the Jev client.

        Only constructor arguments this SDK version actually accepts are
        forwarded, so the quickstart survives SDK upgrades that rename or drop a
        keyword.
        """
        if self._client is not None:
            return self._client
        sdk = self._load_sdk()
        kwargs = _supported_client_kwargs(sdk.TypeSafeClient, self._settings)
        try:
            self._client = sdk.TypeSafeClient(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise UpstreamError(
                f"The Jev client could not be created: {type(exc).__name__}: {exc}",
                hint=(
                    "Check JEV_AI_API_KEY, JEV_AI_MODEL and TYPESAFE_BASE_URL in "
                    ".env, then try again."
                ),
            ) from exc
        return self._client

    def _system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        context: str,
    ) -> tuple[Any, float]:
        """Send one request and return ``(response, latency_ms)``.

        Every question travels in a single call: Jev evaluates them in parallel,
        which is cheaper and faster than one request per question.
        """
        client = self._get_client()
        started = time.perf_counter()
        try:
            response = client.system_one(state=state, questions=questions)
        except Exception as exc:  # noqa: BLE001 - translated for the user
            raise _as_upstream_error(exc, context) from exc
        return response, (time.perf_counter() - started) * 1000.0

    # -- the two judgments ----------------------------------------------------
    def triage(self, ticket: str) -> TriageDecision:
        """Ask three questions about one ticket in a single Jev call.

        The questions are deliberately small and independent, so one request
        answers all of them and the code - not the model - decides what to do
        with each answer.
        """
        sdk = self._load_sdk()
        questions = {
            "department": sdk.Choice(
                instructions="Which team should handle this ticket?",
                criteria=dict(DEPARTMENTS),
            ),
            "frustration": sdk.Score(
                instructions="How frustrated does the customer appear?",
                criteria=list(FRUSTRATION_LEVELS),
            ),
            "is_urgent": sdk.Noul(
                instructions=(
                    "The message conveys urgency or time-sensitivity: an explicit "
                    "deadline, blocked work, or revenue being lost right now."
                )
            ),
        }
        response, latency_ms = self._system_one(
            state=ticket, questions=questions, context="triage"
        )
        answers: Mapping[str, Any] = _field(response, "answers", {}) or {}
        return TriageDecision(
            department=_parse_choice("department", answers.get("department")),
            frustration=_parse_score("frustration", answers.get("frustration")),
            is_urgent=_parse_noul("is_urgent", answers.get("is_urgent")),
            model=str(_field(response, "model", self._settings.jev_model)),
            request_id=_field(response, "request_id"),
            input_tokens=_usage_tokens(_field(response, "usage")),
            source="jev",
            latency_ms=latency_ms,
        )

    def guard(self, ticket: str, draft: str, decision: TriageDecision) -> GuardrailReport:
        """Ask whether a drafted reply is safe to send, in one Jev call.

        The state is a JSON object holding the customer message next to the
        draft, so each yes/no question can compare the two. Questions are asked
        together because they are independent - and because a second request that
        could have been folded into the first one is wasted money.
        """
        sdk = self._load_sdk()
        state = {
            "customer_message": ticket,
            "draft_reply": draft,
            "routed_queue": decision.department.choice,
            "customer_frustration": decision.frustration.score,
        }
        questions = {
            "answers_the_request": sdk.Noul(
                instructions=(
                    "The draft reply directly addresses the customer's request and "
                    "invents no policies, prices, refunds or timelines."
                )
            ),
            "no_personal_data": sdk.Noul(
                instructions=(
                    "The draft reply contains no personal data: no names, email "
                    "addresses, phone numbers, card or account numbers."
                )
            ),
            "tone_ok": sdk.Noul(
                instructions=(
                    "The tone is professional, calm and appropriate for the "
                    "customer's level of frustration."
                )
            ),
        }
        response, latency_ms = self._system_one(
            state=state, questions=questions, context="guardrail"
        )
        answers: Mapping[str, Any] = _field(response, "answers", {}) or {}
        checks = {
            key: _parse_noul(key, answers.get(key)) for key in GUARDRAIL_QUESTION_KEYS
        }
        failures = tuple(
            key for key, answer in checks.items() if answer.noul < GUARDRAIL_MIN_NOUL
        )
        return GuardrailReport(
            checks=checks,
            failures=failures,
            model=str(_field(response, "model", self._settings.jev_model)),
            request_id=_field(response, "request_id"),
            input_tokens=_usage_tokens(_field(response, "usage")),
            source="jev",
            latency_ms=latency_ms,
        )


def _supported_client_kwargs(client_cls: Any, settings: Settings) -> dict[str, Any]:
    """Return only the constructor arguments this SDK version accepts.

    ``typesafe-sdk`` gained ``model``, ``timeout`` and ``base_url`` over time.
    Filtering against the live signature keeps this quickstart working on older
    and newer releases alike, instead of failing with a ``TypeError``.
    """
    wanted: dict[str, Any] = {}
    if settings.jev_api_key is not None:
        wanted["api_key"] = settings.jev_api_key.get_secret_value()
    wanted["model"] = settings.jev_model
    wanted["timeout"] = settings.request_timeout_seconds
    if settings.jev_base_url:
        wanted["base_url"] = settings.jev_base_url

    try:
        parameters = inspect.signature(client_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables only
        return wanted

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return wanted
    return {name: value for name, value in wanted.items() if name in parameters}


# ---------------------------------------------------------------------------
# Offline engine - deterministic stand-in, used by --offline and by the tests.
# ---------------------------------------------------------------------------
_KEYWORD_RULES: Final[dict[str, tuple[str, ...]]] = {
    "billing": (
        "charged",
        "charge",
        "invoice",
        "refund",
        "billing",
        "payment",
        "subscription",
        "duplicate",
        "receipt",
    ),
    "technical": (
        "error",
        "bug",
        "failing",
        "fails",
        "broken",
        "crash",
        "integration",
        "api",
        "timeout",
        "connect",
        "500",
    ),
    "sales": ("pricing", "price", "quote", "plan", "upgrade", "enterprise", "license"),
}

_URGENCY_MARKERS: Final[tuple[str, ...]] = (
    "asap",
    "urgent",
    "immediately",
    "today",
    "right now",
    "losing",
    "blocked",
    "emergency",
    "for days",
)

_FRUSTRATION_MARKERS: Final[tuple[str, ...]] = (
    "frustrated",
    "frustrating",
    "angry",
    "unacceptable",
    "ridiculous",
    "terrible",
    "again",
    "still",
    "no one",
)

_PII_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),  # email address
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),  # card-like digit run
    re.compile(r"\+?\d[\d\s().-]{7,}\d"),  # phone-like digit run
)

_RUDE_MARKERS: Final[tuple[str, ...]] = (
    "not our problem",
    "calm down",
    "whatever",
    "as i said",
)


def _confidence_from(probabilities: Mapping[str, float]) -> float:
    """Reproduce the documented confidence formula for a discrete answer.

    For ``n`` options, ``confidence = (n * peak - 1) / (n - 1)`` clamped to
    ``[0, 1]`` - a flat distribution scores 0, a dominant option approaches 1.
    See https://docs.typesafe.ai/confidence
    """
    values = list(probabilities.values())
    if len(values) < 2:
        return 1.0 if values else 0.0
    peak = max(values)
    return max(0.0, min(1.0, (len(values) * peak - 1.0) / (len(values) - 1.0)))


def _spread(winner: str, peak: float, keys: tuple[str, ...]) -> dict[str, float]:
    """Give ``winner`` the probability ``peak`` and share the rest evenly."""
    others = [key for key in keys if key != winner]
    remainder = max(0.0, 1.0 - peak)
    each = remainder / len(others) if others else 0.0
    return {key: (round(peak, 4) if key == winner else round(each, 4)) for key in keys}


class OfflineDecisionEngine:
    """Deterministic, network-free stand-in for the real Jev model.

    It exists so that a reader can clone the repository and watch the whole
    pipeline run before they have any credentials, and so the test suite can
    assert on behaviour that never touches the network.

    It mimics Jev's *shape* - typed answers, a probability distribution and a
    confidence - not its quality. Keywords are not understanding: swap this out
    for :class:`JevDecisionEngine` to see what the real model does.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        """Identifier shown in the CLI header and in logs."""
        return "offline/canned-rules"

    def triage(self, ticket: str) -> TriageDecision:
        """Score the ticket with keyword rules, deterministically."""
        started = time.perf_counter()
        text = ticket.lower()

        department, department_hits = self._pick_department(text)
        peak = min(0.90, 0.55 + 0.15 * department_hits) if department_hits else 0.45
        probabilities = _spread(department, peak, tuple(DEPARTMENTS))

        frustration_index, frustration_hits = self._pick_frustration(text)
        frustration_peak = min(0.90, 0.60 + 0.15 * frustration_hits)
        frustration_probabilities = _spread(
            str(frustration_index),
            frustration_peak,
            tuple(str(index) for index in range(len(FRUSTRATION_LEVELS))),
        )

        urgency_hits = sum(marker in text for marker in _URGENCY_MARKERS)
        urgency = 0.90 if urgency_hits >= 2 else (0.70 if urgency_hits == 1 else 0.12)

        return TriageDecision(
            department=ChoiceAnswer(
                choice=department,
                confidence=_confidence_from(probabilities),
                probabilities=probabilities,
            ),
            frustration=ScoreAnswer(
                score=float(frustration_index),
                confidence=_confidence_from(frustration_probabilities),
                legend={
                    str(index): label for index, label in enumerate(FRUSTRATION_LEVELS)
                },
                probabilities=frustration_probabilities,
            ),
            is_urgent=NoulAnswer(noul=urgency),
            model="offline/canned-rules",
            source="offline",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _pick_department(text: str) -> tuple[str, int]:
        """Return the best matching queue and how many keywords matched."""
        best, best_hits = UNROUTABLE_DEPARTMENT, 0
        for queue, keywords in _KEYWORD_RULES.items():
            hits = sum(keyword in text for keyword in keywords)
            if hits > best_hits:
                best, best_hits = queue, hits
        return best, best_hits

    @staticmethod
    def _pick_frustration(text: str) -> tuple[int, int]:
        """Return a 0-2 frustration level plus the number of signals found."""
        hits = sum(marker in text for marker in _FRUSTRATION_MARKERS)
        hits += text.count("!")
        if hits == 0:
            return 0, 0
        return min(2, hits), hits

    def guard(self, ticket: str, draft: str, decision: TriageDecision) -> GuardrailReport:
        """Check the draft with regexes instead of a model.

        Deliberately crude: it catches the failure modes that matter for a demo
        (leaked personal data, a draft too short to answer anything, a rude
        tone) and nothing subtler.
        """
        started = time.perf_counter()
        lowered = draft.lower()
        words = len(draft.split())
        leak = self._find_personal_data(draft)
        rude = next((marker for marker in _RUDE_MARKERS if marker in lowered), None)

        checks = {
            "answers_the_request": NoulAnswer(noul=0.92 if words >= 25 else 0.55),
            "no_personal_data": NoulAnswer(noul=0.05 if leak else 0.97),
            "tone_ok": NoulAnswer(noul=0.35 if rude else 0.94),
        }
        failures = tuple(
            key for key, answer in checks.items() if answer.noul < GUARDRAIL_MIN_NOUL
        )
        if leak:
            LOGGER.warning("Offline guardrail matched personal data: %s", leak)

        return GuardrailReport(
            checks=checks,
            failures=failures,
            model="offline/canned-rules",
            source="offline",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _find_personal_data(draft: str) -> str | None:
        """Return the first personal-data match in ``draft``, if any."""
        for pattern in _PII_PATTERNS:
            match = pattern.search(draft)
            if match:
                return match.group(0)
        return None


def create_decision_engine(settings: Settings) -> DecisionEngine:
    """Return the offline stand-in or the live Jev engine, based on settings.

    Args:
        settings: Validated configuration; ``settings.offline`` picks the engine.

    Returns:
        An object satisfying the :class:`DecisionEngine` protocol.
    """
    if settings.offline:
        LOGGER.info("Offline mode: canned decisions, no network calls.")
        return OfflineDecisionEngine(settings)
    LOGGER.info("Live mode: asking Jev (%s).", settings.jev_model)
    return JevDecisionEngine(settings)
