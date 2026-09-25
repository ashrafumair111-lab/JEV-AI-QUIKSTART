"""The generative layer: turn a Jev decision into text a customer can read.

Jev decides; an LLM only writes. Keeping those two jobs apart is the entire
point of the architecture this quickstart demonstrates:

* the decision is a small, typed, cheap call with a ``confidence`` attached;
* the draft is the one part that needs a generative model, and it is only
  produced *after* the confidence gate says it is worth spending tokens;
* Jev then checks the draft again (see :mod:`src.decisions`) before a customer
  ever sees it.

Two writers implement :class:`DraftWriter`, mirroring the two decision engines:

``LangChainDraftWriter``
    OpenAI chat model wired up with LangChain (LCEL: prompt | model | parser).
``TemplateDraftWriter``
    Deterministic, no key and no network - used for ``--offline`` runs and when
    ``OPENAI_API_KEY`` is not configured.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Final, Mapping, Protocol

from src.config import Settings
from src.decisions import UNROUTABLE_DEPARTMENT, TriageDecision
from src.errors import MissingDependencyError, UpstreamError

__all__ = [
    "DraftResult",
    "DraftWriter",
    "LangChainDraftWriter",
    "TemplateDraftWriter",
    "create_draft_writer",
]

LOGGER: Final[logging.Logger] = logging.getLogger("jevai.writer")

#: Low temperature: this is a support reply, not a creative writing exercise.
_TEMPERATURE: Final[float] = 0.3

#: Transient provider errors are worth one automatic retry inside LangChain.
_MAX_RETRIES: Final[int] = 2

#: The system prompt. Note what it does *not* let the model do: invent policy.
#: Every fact in the reply has to come from the customer's own message.
SYSTEM_PROMPT: Final[str] = """\
You are the first-line support agent for the {queue} team.

A decision model has already classified this ticket for you:
- routing queue: {queue}
- customer frustration level: {frustration_label}
- probability the message is urgent: {urgency:.2f}

Write the reply body only, at most 120 words, following these rules:
1. Acknowledge the specific problem the customer described, in their terms.
2. Give exactly one concrete next step that a support agent would really take.
3. Never invent policies, prices, refunds, delivery dates or names.
4. Include no personal data of any kind.
5. Match the tone to the frustration level: calm for level 0, apologetic and
   reassuring from level 1 upwards."""

#: The human turn: the raw ticket, quoted so the model cannot mistake it for
#: instructions (a cheap prompt-injection guard for the demo).
HUMAN_PROMPT: Final[str] = """\
Customer message, between the markers:

--- BEGIN CUSTOMER MESSAGE ---
{customer_message}
--- END CUSTOMER MESSAGE ---

Write the reply body now."""


@dataclass(frozen=True, slots=True)
class DraftResult:
    """A drafted reply plus the provenance needed to audit it later.

    Attributes:
        text: The reply body that will be sent if the guardrails pass.
        model: Model that produced it, or the template's identifier.
        source: ``"langchain-openai"`` or ``"template"``.
        latency_ms: Wall-clock time spent producing the draft.
    """

    text: str
    model: str
    source: str
    latency_ms: float = 0.0


class DraftWriter(Protocol):
    """Anything that can turn ``(ticket, decision)`` into a :class:`DraftResult`."""

    @property
    def name(self) -> str:
        """Short human-readable identifier, e.g. ``"langchain-openai/gpt-4o-mini"``."""
        ...

    def draft(self, ticket: str, decision: TriageDecision) -> DraftResult:
        """Write the reply body for ``ticket``, guided by ``decision``."""
        ...


def _label_for(decision: TriageDecision) -> str:
    """Return the human label of the frustration level, else a plain number."""
    return decision.frustration.label or f"level {decision.frustration.score:.1f}"


def _supported_llm_kwargs(llm_cls: Any, settings: Settings) -> dict[str, Any]:
    """Build constructor arguments that this LangChain version accepts.

    LangChain exposes ``model``/``model_name`` and ``timeout``/``request_timeout``
    as aliased fields, and those names have moved between releases. Resolving the
    alias from the model itself keeps the quickstart working across versions.
    """
    fields: Mapping[str, Any] = getattr(llm_cls, "model_fields", {})
    kwargs: dict[str, Any] = {}

    for candidate in ("model", "model_name"):
        field = fields.get(candidate)
        if field is not None:
            kwargs[getattr(field, "alias", None) or candidate] = settings.openai_model
            break

    if "temperature" in fields:
        kwargs["temperature"] = _TEMPERATURE
    if "max_retries" in fields:
        kwargs["max_retries"] = _MAX_RETRIES

    for candidate in ("timeout", "request_timeout"):
        field = fields.get(candidate)
        if field is not None:
            kwargs[getattr(field, "alias", None) or candidate] = (
                settings.request_timeout_seconds
            )
            break

    if settings.openai_api_key is not None:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()
    return kwargs


class LangChainDraftWriter:
    """Drafts the reply with an OpenAI chat model, wired together by LangChain.

    The chain is the plain LCEL shape most people start with::

        prompt | chat_model | string_parser

    Building it once in ``__init__`` means the HTTP client, the retry policy and
    the prompt template are reused for every ticket.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chain: Any = self._build_chain()

    @property
    def name(self) -> str:
        """Identifier shown in the CLI header and in logs."""
        return f"langchain-openai/{self._settings.openai_model}"

    def _build_chain(self) -> Any:
        """Import LangChain lazily and assemble the runnable chain."""
        try:
            prompts = importlib.import_module("langchain_core.prompts")
            parsers = importlib.import_module("langchain_core.output_parsers")
            openai_integration = importlib.import_module("langchain_openai")
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise MissingDependencyError(
                "LangChain (or its OpenAI integration) is not installed.",
                hint=(
                    "Install the project dependencies:\n"
                    "    python -m pip install -r requirements.txt\n"
                    "Without OPENAI_API_KEY the demo uses a deterministic template "
                    "writer, and --offline uses canned decisions instead of Jev."
                ),
            ) from exc

        prompt = prompts.ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        llm_kwargs = _supported_llm_kwargs(openai_integration.ChatOpenAI, self._settings)
        try:
            llm = openai_integration.ChatOpenAI(**llm_kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise UpstreamError(
                f"The OpenAI chat model could not be created: {type(exc).__name__}: {exc}",
                hint="Check OPENAI_MODEL and OPENAI_API_KEY in .env.",
            ) from exc
        return prompt | llm | parsers.StrOutputParser()

    def draft(self, ticket: str, decision: TriageDecision) -> DraftResult:
        """Invoke the chain once and return the reply body.

        Jev's decision is passed into the prompt as context: the model is told
        which queue owns the ticket, how frustrated the customer is and how
        urgent Jev judged the message. The model writes; it does not decide.

        Raises:
            UpstreamError: If the provider call fails (auth, quota, network).
        """
        inputs = {
            "queue": decision.department.choice,
            "frustration_label": _label_for(decision),
            "urgency": decision.is_urgent.noul,
            "customer_message": ticket,
        }
        started = time.perf_counter()
        try:
            text = self._chain.invoke(inputs)
        except Exception as exc:  # noqa: BLE001 - translated for the user
            raise _as_llm_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        body = str(text).strip()
        if not body:
            raise UpstreamError(
                "The chat model returned an empty reply.",
                hint="Retry, or set OPENAI_MODEL to a model your key can access.",
            )
        return DraftResult(
            text=body,
            model=self._settings.openai_model,
            source="langchain-openai",
            latency_ms=latency_ms,
        )


def _as_llm_error(exc: Exception) -> UpstreamError:
    """Translate a provider failure into a message a reader can act on."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    lowered = f"{name} {exc}".lower()

    if status == 401 or "authentication" in lowered or "api key" in lowered:
        hint = (
            "OpenAI rejected the credential. Check OPENAI_API_KEY in .env, or "
            "remove it and let the deterministic template writer run instead."
        )
    elif status == 429 or "rate limit" in lowered or "quota" in lowered:
        hint = (
            "The OpenAI quota or rate limit was hit. Retry later, or unset "
            "OPENAI_API_KEY to use the template writer."
        )
    elif status == 404 or "does not exist" in lowered:
        hint = (
            "OPENAI_MODEL is probably wrong - see "
            "https://platform.openai.com/docs/models"
        )
    elif "timeout" in lowered or "timed out" in lowered:
        hint = "Raise JEV_AI_REQUEST_TIMEOUT_SECONDS in .env, or retry."
    else:
        hint = "Retry; if it persists check https://status.openai.com."
    return UpstreamError(f"OpenAI draft failed: {name}: {exc}", hint=hint)


class TemplateDraftWriter:
    """Deterministic reply writer used when no LLM is available.

    It produces the same shape the prompt asks an LLM for - acknowledge the
    problem, give exactly one next step, invent nothing - so the whole pipeline
    can be demonstrated with no key, no network and no spend.
    """

    _OPENERS: Final[Mapping[str, str]] = {
        "billing": "Thank you for flagging this, and I am sorry the payment side has gone wrong.",
        "technical": "Thank you for the detail in your message, and I am sorry this integration is not behaving as it should.",
        "sales": "Thank you for reaching out about plans and pricing.",
        UNROUTABLE_DEPARTMENT: "Thank you for getting in touch. I want to make sure this reaches the right team.",
    }
    _NEXT_STEPS: Final[Mapping[str, str]] = {
        "billing": "I have asked the billing team to reconcile the account and correct anything that was charged in error.",
        "technical": "I have asked an integration engineer to reproduce the failing call and come back with a fix or a workaround.",
        "sales": "I have asked the sales team to review your requirements and put together a costed proposal.",
        UNROUTABLE_DEPARTMENT: "A support agent will read your message and route it to the team that owns it.",
    }

    @property
    def name(self) -> str:
        """Identifier shown in the CLI header and in logs."""
        return "template/deterministic"

    def draft(self, ticket: str, decision: TriageDecision) -> DraftResult:
        """Compose the reply from fixed, reviewed sentences.

        The decision still shapes the text - the queue picks the opener and the
        next step, frustration picks the apology, urgency picks the closing
        promise - which is why the pipeline behaves differently per ticket even
        with no LLM involved.
        """
        started = time.perf_counter()
        queue = decision.department.choice
        opener = self._OPENERS.get(queue, self._OPENERS[UNROUTABLE_DEPARTMENT])
        next_step = self._NEXT_STEPS.get(queue, self._NEXT_STEPS[UNROUTABLE_DEPARTMENT])
        empathy = (
            "I can see this has been disruptive, and I understand the frustration."
            if decision.frustration.score >= 1
            else "I appreciate you taking the time to report it."
        )
        urgency = (
            "Because this is affecting you right now, I am treating it as a priority."
            if decision.is_urgent.noul >= 0.5
            else "I will keep you updated as it progresses."
        )
        body = (
            f"Hello,\n\n{opener}\n\n{empathy} {next_step} {urgency}\n\n"
            "Kind regards,\nThe support team"
        )
        return DraftResult(
            text=body,
            model="template/deterministic",
            source="template",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def create_draft_writer(settings: Settings) -> DraftWriter:
    """Return an LLM-backed writer when possible, else the deterministic one.

    Falling back instead of failing is deliberate: a reader who has a Jev key but
    no OpenAI key still gets a complete run, they just get a template reply
    instead of a generated one.
    """
    if settings.offline or not settings.has_openai_key:
        reason = "offline mode" if settings.offline else "OPENAI_API_KEY is not set"
        LOGGER.info("Using the deterministic template writer (%s).", reason)
        return TemplateDraftWriter()
    try:
        return LangChainDraftWriter(settings)
    except MissingDependencyError as exc:
        LOGGER.warning("%s Falling back to the template writer.", exc.message)
        return TemplateDraftWriter()
