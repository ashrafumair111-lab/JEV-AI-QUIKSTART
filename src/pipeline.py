"""Orchestration: triage -> confidence gate -> draft -> guardrail.

If you read only one file in this repository, read this one. It shows the
three-layer architecture the Jev documentation recommends:

1. **Jev decides.** A single call returns the queue, the frustration level and
   the urgency probability - each with a calibrated confidence.
2. **Your code owns the rules.** :meth:`TriagePipeline._gate` turns that
   confidence into an action (automatic, review, or human) using thresholds from
   ``.env``. The model is never asked how much authority it should have.
3. **The LLM writes, Jev checks.** A reply is drafted only when the gate says it
   is worth the tokens, and is then re-checked by Jev before it is released.

Cost note: an escalated ticket costs exactly one Jev call and zero LLM calls,
because the expensive model is never woken up. That is the point of gating on
confidence instead of always calling the biggest model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.config import Settings
from src.decisions import (
    UNROUTABLE_DEPARTMENT,
    DecisionEngine,
    GuardrailReport,
    TriageDecision,
)
from src.writer import DraftResult, DraftWriter

__all__ = [
    "GateAction",
    "GateDecision",
    "OutcomeStatus",
    "TicketOutcome",
    "TriagePipeline",
]

LOGGER: Final[logging.Logger] = logging.getLogger("jevai.pipeline")

#: Frustration score (0-2) at or above which even a confident auto-draft is held
#: back for a human read. This is policy, so it lives in code - not in a prompt.
_ESCALATE_FRUSTRATION: Final[float] = 2.0

#: Urgency probability above which an auto-draft is still reviewed by a human.
_REVIEW_URGENCY: Final[float] = 0.85


class GateAction(str, Enum):
    """What the pipeline is allowed to do with a ticket."""

    AUTO_DRAFT = "auto_draft"
    DRAFT_WITH_REVIEW = "draft_with_review"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class OutcomeStatus(str, Enum):
    """How a ticket ended up once the pipeline was finished with it."""

    RELEASED = "released"
    HELD_FOR_REVIEW = "held_for_review"
    ESCALATED_TO_HUMAN = "escalated_to_human"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The routing call *your code* makes from Jev's confidence.

    Attributes:
        action: What to do next.
        confidence: The confidence value the decision was based on.
        reason: Short explanation, printed by the CLI and worth keeping in logs.
    """

    action: GateAction
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class TicketOutcome:
    """Everything that happened to one ticket, ready to print or to audit.

    Attributes:
        ticket: The original customer message.
        decision: Jev's typed judgment.
        gate: The action chosen by code from that judgment.
        draft: The reply body, or ``None`` when the ticket was escalated.
        guardrails: Jev's checks on the draft, or ``None`` when there is no draft.
        status: Final state of the ticket.
        elapsed_ms: End-to-end wall-clock time.
    """

    ticket: str
    decision: TriageDecision
    gate: GateDecision
    draft: DraftResult | None = None
    guardrails: GuardrailReport | None = None
    status: OutcomeStatus = OutcomeStatus.ESCALATED_TO_HUMAN
    elapsed_ms: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        """Billable Jev input tokens across every call made for this ticket."""
        total = self.decision.input_tokens or 0
        if self.guardrails is not None:
            total += self.guardrails.input_tokens or 0
        return total

    @property
    def llm_calls(self) -> int:
        """How many generative-model calls this ticket actually caused."""
        return 1 if self.draft is not None else 0


class TriagePipeline:
    """Runs one ticket through the whole agent, and returns the full record.

    The pipeline depends on the :class:`~src.decisions.DecisionEngine` and
    :class:`~src.writer.DraftWriter` *protocols*, not on concrete classes, so a
    test can pass a scripted double and assert on routing without any network
    access - see ``tests/test_quickstart.py``.
    """

    def __init__(
        self,
        settings: Settings,
        engine: DecisionEngine,
        writer: DraftWriter,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._writer = writer

    @property
    def engine_name(self) -> str:
        """Identifier of the decision engine in use."""
        return self._engine.name

    @property
    def writer_name(self) -> str:
        """Identifier of the draft writer in use."""
        return self._writer.name

    def run(self, ticket: str) -> TicketOutcome:
        """Process one ticket end to end.

        Steps:
            1. Ask Jev for the typed judgment (one request, three questions).
            2. Gate on confidence - code decides, not the model.
            3. Stop here for escalations: no draft, no LLM spend.
            4. Draft the reply, then have Jev check it before release.

        Args:
            ticket: The raw customer message.

        Returns:
            The complete :class:`TicketOutcome` record.
        """
        started = time.perf_counter()
        decision = self._engine.triage(ticket)
        gate = self._gate(decision)
        LOGGER.debug(
            "Triage: queue=%s confidence=%.3f -> %s",
            decision.department.choice,
            decision.department.confidence,
            gate.action.value,
        )

        if gate.action is GateAction.ESCALATE_TO_HUMAN:
            LOGGER.info("Escalating to a human: %s", gate.reason)
            return TicketOutcome(
                ticket=ticket,
                decision=decision,
                gate=gate,
                status=OutcomeStatus.ESCALATED_TO_HUMAN,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        draft = self._writer.draft(ticket, decision)
        guardrails = self._engine.guard(ticket, draft.text, decision)
        if guardrails.blocked:
            LOGGER.info("Guardrails failed: %s", ", ".join(guardrails.failures))
            status = OutcomeStatus.HELD_FOR_REVIEW
        elif gate.action is GateAction.AUTO_DRAFT:
            status = OutcomeStatus.RELEASED
        else:
            status = OutcomeStatus.HELD_FOR_REVIEW

        return TicketOutcome(
            ticket=ticket,
            decision=decision,
            gate=gate,
            draft=draft,
            guardrails=guardrails,
            status=status,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _gate(self, decision: TriageDecision) -> GateDecision:
        """Turn Jev's confidence into an action, using rules you own.

        The order of the checks is the order of the risks:

        1. No queue fits at all - a human must route it.
        2. Below the confidence floor - do not guess; escalate and say so.
        3. Very upset customer, or an urgent-but-not-certain ticket - draft it,
           but a human reads it before it is sent.
        4. At or above the auto-act threshold - draft and release.

        Every branch is deliberately explainable, because the reason string ends
        up in the logs next to the confidence that produced it.
        """
        confidence = decision.department.confidence
        floor = self._settings.confidence_floor
        threshold = self._settings.auto_act_threshold
        urgency = decision.is_urgent.noul

        if decision.department.choice == UNROUTABLE_DEPARTMENT:
            return GateDecision(
                action=GateAction.ESCALATE_TO_HUMAN,
                confidence=confidence,
                reason="Jev found no queue that owns this ticket",
            )
        if confidence < floor:
            return GateDecision(
                action=GateAction.ESCALATE_TO_HUMAN,
                confidence=confidence,
                reason=(
                    f"confidence {confidence:.2f} is below the escalation floor "
                    f"{floor:.2f}"
                ),
            )
        if decision.frustration.score >= _ESCALATE_FRUSTRATION:
            return GateDecision(
                action=GateAction.DRAFT_WITH_REVIEW,
                confidence=confidence,
                reason="customer is very upset: draft the reply, let a human read it",
            )
        if confidence >= threshold:
            if urgency >= _REVIEW_URGENCY:
                return GateDecision(
                    action=GateAction.DRAFT_WITH_REVIEW,
                    confidence=confidence,
                    reason=(
                        f"confidence {confidence:.2f} clears the auto-act threshold "
                        f"{threshold:.2f}, but urgency {urgency:.2f} means a human "
                        "should still look"
                    ),
                )
            return GateDecision(
                action=GateAction.AUTO_DRAFT,
                confidence=confidence,
                reason=(
                    f"confidence {confidence:.2f} clears the auto-act threshold "
                    f"{threshold:.2f}"
                ),
            )
        return GateDecision(
            action=GateAction.DRAFT_WITH_REVIEW,
            confidence=confidence,
            reason=(
                f"confidence {confidence:.2f} sits between the floor {floor:.2f} "
                f"and the auto-act threshold {threshold:.2f}"
            ),
        )
