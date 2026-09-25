"""Tests for the JEV AI Quickstart - no credentials, no network, no spend.

Run them with either runner::

    python -m unittest discover -s tests -v
    python -m pytest tests -q

The suite is deliberately written against the two protocols
(:class:`~src.decisions.DecisionEngine` and :class:`~src.writer.DraftWriter`) so
routing can be asserted with scripted answers instead of live calls.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

# Make `src` importable no matter which runner started the suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402 - path bootstrap must run first
    ConfigurationError,
    build_settings,
    verify_ready,
)
from src.decisions import (  # noqa: E402 - path bootstrap must run first
    GUARDRAIL_QUESTION_KEYS,
    ChoiceAnswer,
    GuardrailReport,
    NoulAnswer,
    OfflineDecisionEngine,
    ScoreAnswer,
    TriageDecision,
    _supported_client_kwargs,
    create_decision_engine,
)
from src.errors import QuickstartError  # noqa: E402 - path bootstrap must run first
from src.main import main  # noqa: E402 - path bootstrap must run first
from src.pipeline import (  # noqa: E402 - path bootstrap must run first
    GateAction,
    OutcomeStatus,
    TicketOutcome,
    TriagePipeline,
)
from src.writer import (  # noqa: E402 - path bootstrap must run first
    DraftResult,
    TemplateDraftWriter,
    create_draft_writer,
)

#: A path that certainly holds no environment file.
MISSING_ENV_FILE = Path(__file__).with_name("no-such-file.env")

SAMPLE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP."
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def make_decision(
    *,
    choice: str = "billing",
    confidence: float = 0.95,
    frustration: float = 0.0,
    urgency: float = 0.1,
) -> TriageDecision:
    """Build a :class:`TriageDecision` without calling any model."""
    return TriageDecision(
        department=ChoiceAnswer(
            choice=choice,
            confidence=confidence,
            probabilities={choice: confidence, "other": 1.0 - confidence},
        ),
        frustration=ScoreAnswer(
            score=frustration,
            confidence=0.9,
            legend={"0": "Calm", "1": "Frustrated", "2": "Very angry"},
        ),
        is_urgent=NoulAnswer(noul=urgency),
        model="scripted",
        source="scripted",
    )


class ScriptedEngine:
    """A ``DecisionEngine`` double that returns answers the test dictates."""

    def __init__(self, decision: TriageDecision, *, failures: tuple[str, ...] = ()) -> None:
        self._decision = decision
        self._failures = failures
        self.guard_calls = 0

    @property
    def name(self) -> str:
        """Identifier of this double."""
        return "scripted"

    def triage(self, ticket: str) -> TriageDecision:
        """Return the scripted decision, ignoring the ticket."""
        return self._decision

    def guard(self, ticket: str, draft: str, decision: TriageDecision) -> GuardrailReport:
        """Return a report that fails exactly the scripted checks."""
        self.guard_calls += 1
        checks = {
            key: NoulAnswer(noul=0.10 if key in self._failures else 0.90)
            for key in GUARDRAIL_QUESTION_KEYS
        }
        return GuardrailReport(
            checks=checks, failures=self._failures, model="scripted", source="scripted"
        )


class RecordingWriter:
    """A ``DraftWriter`` double that counts how often it was asked to write."""

    def __init__(self, text: str = "Thanks for the report. We are on it.") -> None:
        self.calls = 0
        self._text = text

    @property
    def name(self) -> str:
        """Identifier of this double."""
        return "recording"

    def draft(self, ticket: str, decision: TriageDecision) -> DraftResult:
        """Record the call and return the canned text."""
        self.calls += 1
        return DraftResult(text=self._text, model="recording", source="test")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SettingsTests(unittest.TestCase):
    """Configuration is validated once, and failures are readable."""

    def test_defaults_are_usable_without_any_environment(self) -> None:
        settings = build_settings({})
        self.assertFalse(settings.offline)
        self.assertFalse(settings.has_jev_key)
        self.assertEqual(settings.jev_model, "jev-latest")
        self.assertEqual(settings.confidence_floor, 0.50)
        self.assertGreaterEqual(settings.auto_act_threshold, settings.confidence_floor)

    def test_values_are_read_from_the_environment_mapping(self) -> None:
        settings = build_settings(
            {
                "JEV_AI_API_KEY": "jev_live_key_1234567890",
                "OPENAI_API_KEY": "sk-live-key-1234567890",
                "JEV_AI_MODEL": "jev-preview",
                "JEV_AI_OFFLINE": "true",
                "JEV_AI_CONFIDENCE_FLOOR": "0.6",
                "JEV_AI_AUTO_ACT_THRESHOLD": "0.9",
                "JEV_AI_LOG_LEVEL": "debug",
                "TYPESAFE_BASE_URL": "https://api.example.test/",
            }
        )
        self.assertTrue(settings.has_jev_key)
        self.assertTrue(settings.has_openai_key)
        self.assertEqual(settings.jev_model, "jev-preview")
        self.assertTrue(settings.offline)
        self.assertEqual(settings.confidence_floor, 0.6)
        self.assertEqual(settings.auto_act_threshold, 0.9)
        self.assertEqual(settings.log_level.value, "DEBUG")
        self.assertEqual(settings.jev_base_url, "https://api.example.test")

    def test_placeholder_values_count_as_missing(self) -> None:
        settings = build_settings(
            {
                "JEV_AI_API_KEY": "your_jev_api_key_here",
                "OPENAI_API_KEY": "your_openai_api_key_here",
            }
        )
        self.assertFalse(settings.has_jev_key)
        self.assertFalse(settings.has_openai_key)

    def test_typesafe_alias_is_accepted(self) -> None:
        settings = build_settings({"TYPESAFE_API_KEY": "jev_alias_key_1234567890"})
        self.assertTrue(settings.has_jev_key)

    def test_thresholds_may_not_be_inverted(self) -> None:
        with self.assertRaises(ConfigurationError):
            build_settings(
                {
                    "JEV_AI_CONFIDENCE_FLOOR": "0.9",
                    "JEV_AI_AUTO_ACT_THRESHOLD": "0.5",
                }
            )

    def test_non_numeric_threshold_is_reported(self) -> None:
        with self.assertRaises(ConfigurationError) as ctx:
            build_settings({"JEV_AI_CONFIDENCE_FLOOR": "very-high"})
        self.assertIn("JEV_AI_CONFIDENCE_FLOOR", str(ctx.exception))

    def test_live_mode_without_a_key_fails_with_guidance(self) -> None:
        with self.assertRaises(ConfigurationError) as ctx:
            verify_ready(build_settings({}))
        self.assertIn("console.typesafe.ai", ctx.exception.hint or "")

    def test_offline_mode_needs_no_key(self) -> None:
        verify_ready(build_settings({"JEV_AI_OFFLINE": "1"}))

    def test_safe_summary_stays_redacted(self) -> None:
        secret = "jev_super_secret_value_9876543210"
        summary = build_settings({"JEV_AI_API_KEY": secret}).safe_summary()
        self.assertNotIn(secret, json.dumps(summary))
        self.assertIn("jev_...", summary["JEV_AI_API_KEY"])


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------


class OfflineDecisionEngineTests(unittest.TestCase):
    """The canned engine must behave like Jev's *shape*, deterministically."""

    def setUp(self) -> None:
        self.settings = build_settings({"JEV_AI_OFFLINE": "true"})
        self.engine = OfflineDecisionEngine(self.settings)

    def test_triage_routes_a_billing_complaint(self) -> None:
        decision = self.engine.triage(
            "I was charged twice for my subscription and I need a refund"
        )
        self.assertEqual(decision.department.choice, "billing")
        self.assertGreater(decision.department.confidence, self.settings.confidence_floor)
        self.assertAlmostEqual(
            sum(decision.department.probabilities.values()), 1.0, places=2
        )
        self.assertEqual(decision.source, "offline")

    def test_unmatched_ticket_is_unroutable_and_uncertain(self) -> None:
        decision = self.engine.triage("Hello, I have a question about the weather")
        self.assertEqual(decision.department.choice, "other")
        self.assertLess(decision.department.confidence, self.settings.confidence_floor)

    def test_urgency_marker_raises_the_noul(self) -> None:
        self.assertGreater(self.engine.triage(SAMPLE).is_urgent.noul, 0.5)

    def test_guard_flags_personal_data(self) -> None:
        report = self.engine.guard(
            SAMPLE,
            "Please email me at customer@example.com so we can refund the charge.",
            make_decision(),
        )
        self.assertTrue(report.blocked)
        self.assertIn("no_personal_data", report.failures)

    def test_guard_passes_the_template_draft(self) -> None:
        decision = self.engine.triage(SAMPLE)
        draft = TemplateDraftWriter().draft(SAMPLE, decision)
        self.assertFalse(self.engine.guard(SAMPLE, draft.text, decision).blocked)

    def test_factory_picks_the_offline_engine_when_asked(self) -> None:
        self.assertIsInstance(create_decision_engine(self.settings), OfflineDecisionEngine)

    def test_factory_picks_the_template_writer_offline(self) -> None:
        self.assertIsInstance(create_draft_writer(self.settings), TemplateDraftWriter)


class ClientArgumentTests(unittest.TestCase):
    """The SDK constructor is probed, so version drift cannot break a run."""

    def test_only_supported_arguments_are_forwarded(self) -> None:
        class FakeClient:
            def __init__(
                self, *, api_key: str | None = None, model: str | None = None
            ) -> None:
                self.api_key = api_key
                self.model = model

        kwargs = _supported_client_kwargs(
            FakeClient,
            build_settings(
                {
                    "JEV_AI_API_KEY": "jev_key_1234567890",
                    "TYPESAFE_BASE_URL": "https://api.example.test",
                }
            ),
        )
        self.assertEqual(set(kwargs), {"api_key", "model"})


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TriagePipelineTests(unittest.TestCase):
    """Routing, escalation and guardrails - all with scripted answers."""

    def setUp(self) -> None:
        self.settings = build_settings({"JEV_AI_OFFLINE": "true"})

    def _run(
        self,
        decision: TriageDecision,
        *,
        failures: tuple[str, ...] = (),
    ) -> tuple[TicketOutcome, ScriptedEngine, RecordingWriter]:
        """Run the sample ticket through a scripted engine and a spy writer."""
        engine = ScriptedEngine(decision, failures=failures)
        writer = RecordingWriter()
        pipeline = TriagePipeline(self.settings, engine, writer)
        return pipeline.run(SAMPLE), engine, writer

    def test_confident_ticket_is_drafted_and_released(self) -> None:
        outcome, engine, writer = self._run(make_decision(confidence=0.95))
        self.assertEqual(outcome.gate.action, GateAction.AUTO_DRAFT)
        self.assertEqual(outcome.status, OutcomeStatus.RELEASED)
        self.assertEqual(writer.calls, 1)
        self.assertEqual(engine.guard_calls, 1)
        self.assertEqual(outcome.llm_calls, 1)
        self.assertIsNotNone(outcome.draft)

    def test_low_confidence_escalates_and_skips_the_llm(self) -> None:
        outcome, engine, writer = self._run(make_decision(confidence=0.30))
        self.assertEqual(outcome.gate.action, GateAction.ESCALATE_TO_HUMAN)
        self.assertEqual(outcome.status, OutcomeStatus.ESCALATED_TO_HUMAN)
        self.assertIsNone(outcome.draft)
        self.assertIsNone(outcome.guardrails)
        self.assertEqual(writer.calls, 0)
        self.assertEqual(engine.guard_calls, 0)
        self.assertEqual(outcome.llm_calls, 0)

    def test_unroutable_department_escalates_even_when_confident(self) -> None:
        outcome, _, writer = self._run(make_decision(choice="other", confidence=0.99))
        self.assertEqual(outcome.gate.action, GateAction.ESCALATE_TO_HUMAN)
        self.assertEqual(writer.calls, 0)

    def test_very_upset_customer_is_held_for_review(self) -> None:
        outcome, _, _ = self._run(make_decision(confidence=0.95, frustration=2.0))
        self.assertEqual(outcome.gate.action, GateAction.DRAFT_WITH_REVIEW)
        self.assertEqual(outcome.status, OutcomeStatus.HELD_FOR_REVIEW)

    def test_urgent_and_confident_ticket_is_still_reviewed(self) -> None:
        outcome, _, _ = self._run(make_decision(confidence=0.95, urgency=0.90))
        self.assertEqual(outcome.gate.action, GateAction.DRAFT_WITH_REVIEW)

    def test_mid_confidence_drafts_but_holds(self) -> None:
        outcome, _, writer = self._run(make_decision(confidence=0.70))
        self.assertEqual(outcome.gate.action, GateAction.DRAFT_WITH_REVIEW)
        self.assertEqual(writer.calls, 1)
        self.assertEqual(outcome.status, OutcomeStatus.HELD_FOR_REVIEW)

    def test_failed_guardrail_holds_the_reply(self) -> None:
        outcome, _, _ = self._run(
            make_decision(confidence=0.95), failures=("no_personal_data",)
        )
        self.assertIsNotNone(outcome.guardrails)
        assert outcome.guardrails is not None  # narrow the type for the reader
        self.assertTrue(outcome.guardrails.blocked)
        self.assertEqual(outcome.status, OutcomeStatus.HELD_FOR_REVIEW)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    """The CLI is the interface most readers meet first."""

    def _run_cli(self, *args: str) -> tuple[int, str]:
        """Run ``main`` with a guaranteed-absent env file and capture stdout."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([*args, "--env-file", str(MISSING_ENV_FILE)])
        return code, buffer.getvalue()

    def test_offline_run_prints_a_full_report(self) -> None:
        code, output = self._run_cli("--offline")
        self.assertEqual(code, 0)
        for section in (
            "Jev decision",
            "Confidence gate",
            "Drafted reply",
            "Guardrail checks",
            "Result",
        ):
            self.assertIn(section, output)

    def test_json_output_is_machine_readable(self) -> None:
        code, output = self._run_cli("--offline", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn(payload["gate"]["action"], {action.value for action in GateAction})
        self.assertIn(payload["status"], {status.value for status in OutcomeStatus})
        self.assertIsNone(payload["decision"]["request_id"])
        self.assertIn("Stripe", payload["ticket"])
        self.assertTrue(payload["draft"]["text"])

    def test_custom_ticket_is_used(self) -> None:
        code, output = self._run_cli("--offline", "--ticket", "Please refund my invoice")
        self.assertEqual(code, 0)
        self.assertIn("billing", output)

    def test_missing_key_returns_the_config_exit_code(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            code, _ = self._run_cli()
        self.assertEqual(code, 2)

    def test_interactive_style_flags_are_accepted(self) -> None:
        parser = main.__globals__["build_parser"]()
        args = parser.parse_args(["--offline", "--verbose", "--json"])
        self.assertTrue(args.offline)
        self.assertTrue(args.verbose)
        self.assertTrue(args.json)
        self.assertIsNone(args.ticket)


if __name__ == "__main__":  # pragma: no cover - allows `python tests/test_quickstart.py`
    unittest.main(verbosity=2)
