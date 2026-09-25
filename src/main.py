"""Command-line entry point for the JEV AI Quickstart.

Run it either way::

    python -m src.main                     # live mode (needs JEV_AI_API_KEY)
    python -m src.main --offline           # no keys, no network - try this first
    python -m src.main --ticket "..."      # your own customer message
    python -m src.main --json > run.json   # machine-readable output
    python -m src.main --verbose           # extra detail about every call

What it does, in five steps:

1. loads and validates ``.env`` (see ``src/config.py``);
2. sends one Jev request that answers three questions about the ticket;
3. converts Jev's confidence into an action using your thresholds;
4. drafts a reply - with an LLM if ``OPENAI_API_KEY`` is set, otherwise with a
   deterministic template - and skips this step entirely when it escalates;
5. has Jev check the draft before printing the final status.

Exit codes:
    0    ran successfully (whatever the routing decision was)
    2    configuration problem - missing or invalid settings
    3    a required package is not installed
    4    Jev or the LLM provider returned an error
    130  interrupted by the user
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Support both `python -m src.main` and `python src/main.py`.
if __package__ in {None, ""}:  # pragma: no cover - depends on how it is launched
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import __version__
from src.config import (
    DEFAULT_ENV_FILE,
    ConfigurationError,
    LogLevel,
    Settings,
    configure_logging,
    get_settings,
    runtime_warnings,
    verify_ready,
)

from src.decisions import create_decision_engine
from src.errors import MissingDependencyError, QuickstartError, UpstreamError
from src.pipeline import OutcomeStatus, TicketOutcome, TriagePipeline
from src.writer import create_draft_writer

__all__ = ["SAMPLE_TICKET", "build_parser", "main", "run_demo"]

#: The ticket used when ``--ticket`` is not supplied. It is the same message the
#: official Jev quickstart uses, so this output can be compared with that one.
SAMPLE_TICKET: Final[str] = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP."
)

EXIT_OK: Final[int] = 0
EXIT_CONFIG: Final[int] = 2
EXIT_MISSING_DEPENDENCY: Final[int] = 3
EXIT_UPSTREAM: Final[int] = 4
EXIT_INTERRUPTED: Final[int] = 130


def _import_rich() -> Any | None:
    """Return the ``rich`` module with its submodules, or ``None`` if absent.

    ``rich`` is a convenience, never a requirement: the CLI falls back to plain
    ``print`` so the demo still runs on a bare interpreter.

    Note that importing the package alone does not expose ``rich.table`` or
    ``rich.box`` - the submodules have to be imported explicitly.
    """
    try:
        import rich
        import rich.box
        import rich.console
        import rich.table
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return rich


def _format_probabilities(probabilities: Mapping[str, float]) -> str:
    """Render a distribution as ``"technical 0.85 | billing 0.15"``, largest first."""
    if not probabilities:
        return "(not reported)"
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return " | ".join(f"{key} {value:.2f}" for key, value in ordered)


class Reporter:
    """Prints the run report, using ``rich`` when it is available."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._rich = _import_rich()
        self._console = self._rich.get_console() if self._rich is not None else None

    @property
    def verbose(self) -> bool:
        """True when the caller asked for extra detail."""
        return self._verbose

    # -- small primitives used by every section -------------------------------
    def _line(self, text: str = "") -> None:
        """Print one line through rich when present, else plainly."""
        if self._console is not None:
            self._console.print(text)
        else:
            print(text)

    def _rule(self, title: str) -> None:
        """Print a section heading."""
        if self._console is not None:
            self._console.rule(f"[bold]{title}[/bold]")
        else:
            print(f"\n{title}\n" + "-" * max(12, len(title)))

    def _table(self, rows: Sequence[tuple[str, str]]) -> None:
        """Print a two-column key/value table."""
        if self._console is not None:
            table = self._rich.table.Table(box=self._rich.box.ROUNDED, show_header=False)
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column(overflow="fold")
            for key, value in rows:
                table.add_row(key, value)
            self._console.print(table)
            return
        width = max((len(key) for key, _ in rows), default=0)
        for key, value in rows:
            print(f"  {key.ljust(width)}  {value}")

    # -- report sections ------------------------------------------------------
    def header(self, *, settings: Settings, engine_name: str, writer_name: str) -> None:
        """Print the banner and the components this run is using."""
        mode = "OFFLINE" if settings.offline else "LIVE"
        self._rule(f"JEV AI Quickstart - {mode} mode")
        self._table([("decision engine", engine_name), ("draft writer", writer_name)])

    def config(self, settings: Settings) -> None:
        """Print the redacted configuration that is in force."""
        self._rule("Configuration")
        self._table(list(settings.safe_summary().items()))

    def warnings(self, messages: Sequence[str]) -> None:
        """Print non-fatal warnings, if any."""
        for message in messages:
            if self._console is not None:
                self._console.print(f"[yellow]note:[/yellow] {message}")
            else:
                print(f"note: {message}")

    def error(self, message: str, hint: str | None = None) -> None:
        """Print a failure and the instructions that fix it."""
        if self._console is not None:
            self._console.print(f"[bold red]Error:[/bold red] {message}")
            if hint:
                self._console.print(f"[dim]{hint}[/dim]")
            return
        print(f"Error: {message}", file=sys.stderr)
        if hint:
            print(hint, file=sys.stderr)

    def ticket(self, text: str) -> None:
        """Print the customer message under a heading."""
        self._rule("Customer message")
        self._line(text)

    def decision(self, outcome: TicketOutcome) -> None:
        """Print Jev's typed answers - the part that replaces a prompt."""
        decision = outcome.decision
        rows = [
            (
                "queue",
                f"{decision.department.choice} "
                f"(confidence {decision.department.confidence:.2f})",
            ),
            (
                "  probabilities",
                _format_probabilities(decision.department.probabilities),
            ),
            (
                "frustration",
                f"{decision.frustration.score:.0f} = "
                f"{decision.frustration.label or 'not labelled'} "
                f"(confidence {decision.frustration.confidence:.2f})",
            ),
            (
                "urgency",
                f"{decision.is_urgent.noul:.2f} probability the message is "
                "time-sensitive",
            ),
            (
                "answered by",
                f"{decision.model} in {decision.latency_ms:.0f} ms "
                f"({decision.source})",
            ),
        ]
        if self._verbose:
            rows.append(("request id", str(decision.request_id or "(not reported)")))
            rows.append(("input tokens", str(decision.input_tokens or "(not reported)")))
        self._rule("Jev decision")
        self._table(rows)

    def gate(self, outcome: TicketOutcome) -> None:
        """Print the action chosen from that confidence, and the reason why."""
        self._rule("Confidence gate (code, not model)")
        self._table(
            [
                ("action", outcome.gate.action.value),
                ("confidence", f"{outcome.gate.confidence:.2f}"),
                ("reason", outcome.gate.reason),
            ]
        )

    def draft(self, outcome: TicketOutcome) -> None:
        """Print the drafted reply, when the gate produced one."""
        if outcome.draft is None:
            return
        self._rule(f"Drafted reply ({outcome.draft.source}: {outcome.draft.model})")
        self._line(outcome.draft.text)

    def guardrails(self, outcome: TicketOutcome) -> None:
        """Print Jev's verdict on the draft, one row per yes/no question."""
        report = outcome.guardrails
        if report is None:
            return
        rows = [
            (
                key,
                f"{answer.noul:.2f}  "
                f"{'FAIL' if key in report.failures else 'pass'}",
            )
            for key, answer in report.checks.items()
        ]
        if self._verbose:
            rows.append(("answered by", f"{report.model} in {report.latency_ms:.0f} ms"))
        self._rule("Guardrail checks (Jev on the draft)")
        self._table(rows)

    def result(self, outcome: TicketOutcome) -> None:
        """Print the final status plus what the run cost in time and tokens."""
        captions = {
            OutcomeStatus.RELEASED: (
                "[bold green]RELEASED[/bold green]",
                "the reply passed every guardrail",
            ),
            OutcomeStatus.HELD_FOR_REVIEW: (
                "[bold yellow]HELD FOR REVIEW[/bold yellow]",
                "the reply is drafted but a human should approve it",
            ),
            OutcomeStatus.ESCALATED_TO_HUMAN: (
                "[bold magenta]ESCALATED TO HUMAN[/bold magenta]",
                "no draft was written, so no LLM tokens were spent",
            ),
        }
        styled, explanation = captions[outcome.status]
        self._rule("Result")
        if self._console is not None:
            self._console.print(f"{styled}: {explanation}")
        else:
            self._line(f"{outcome.status.value}: {explanation}")
        self._table(
            [
                ("end to end", f"{outcome.elapsed_ms:.0f} ms"),
                ("LLM calls", str(outcome.llm_calls)),
                ("Jev input tokens", str(outcome.total_input_tokens)),
            ]
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="jev-ai-quickstart",
        description="Route one support ticket through a Jev-gated AI agent.",
        epilog=(
            "Docs: https://docs.typesafe.ai   "
            "API keys: https://console.typesafe.ai/keys"
        ),
    )
    parser.add_argument(
        "--ticket",
        metavar="TEXT",
        default=None,
        help="customer message to process (default: the sample ticket)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use canned decisions; no keys and no network calls",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the run record as JSON instead of a human report",
    )
    parser.add_argument(
        "--env-file",
        dest="env_file",
        metavar="PATH",
        default=str(DEFAULT_ENV_FILE),
        help="path of the .env file to load (default: .env beside src/)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every step, and show request ids and token counts",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"jev-ai-quickstart {__version__}",
    )
    return parser


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, enums and mappings into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_demo(settings: Settings, ticket: str | None = None) -> TicketOutcome:
    """Run one ticket through the pipeline and return the full record.

    This is the programmatic entry point - handy in a notebook, a script or a
    test::

        from src.config import get_settings
        from src.main import run_demo

        offline = get_settings().model_copy(update={"offline": True})
        outcome = run_demo(offline)
        print(outcome.status)       # OutcomeStatus.RELEASED
        print(outcome.draft.text)

    Args:
        settings: Validated configuration.
        ticket: Customer message, or ``None`` to use :data:`SAMPLE_TICKET`.

    Returns:
        The :class:`~src.pipeline.TicketOutcome` for that ticket.
    """
    pipeline = TriagePipeline(
        settings,
        create_decision_engine(settings),
        create_draft_writer(settings),
    )
    return pipeline.run(ticket or SAMPLE_TICKET)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``. Tests pass their own.

    Returns:
        A process exit code - see the module docstring for the full table.
    """
    args = build_parser().parse_args(argv)
    configure_logging(LogLevel.DEBUG if args.verbose else LogLevel.WARNING)
    reporter = Reporter(verbose=args.verbose)

    # 1. Configuration. Any problem here is reported, never traced back.
    try:
        settings = get_settings(
            env_file=Path(args.env_file) if args.env_file else None,
            reload=True,
        )
        if args.offline:
            settings = settings.model_copy(update={"offline": True})
        verify_ready(settings)
    except QuickstartError as exc:
        reporter.error(exc.message, exc.hint)
        return EXIT_CONFIG

    # 2. Wiring. Building the engine/writer is where optional packages are needed.
    try:
        pipeline = TriagePipeline(
            settings,
            create_decision_engine(settings),
            create_draft_writer(settings),
        )
    except MissingDependencyError as exc:
        reporter.error(exc.message, exc.hint)
        return EXIT_MISSING_DEPENDENCY
    except QuickstartError as exc:
        reporter.error(exc.message, exc.hint)
        return EXIT_UPSTREAM

    ticket = args.ticket or SAMPLE_TICKET
    if not args.json:
        reporter.header(
            settings=settings,
            engine_name=pipeline.engine_name,
            writer_name=pipeline.writer_name,
        )
        reporter.config(settings)
        reporter.warnings(runtime_warnings(settings))
        reporter.ticket(ticket)

    # 3. The actual work. Provider failures are translated, not dumped.
    try:
        outcome = pipeline.run(ticket)
    except MissingDependencyError as exc:
        reporter.error(exc.message, exc.hint)
        return EXIT_MISSING_DEPENDENCY
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        reporter.error("Interrupted before the run finished.")
        return EXIT_INTERRUPTED
    except QuickstartError as exc:
        reporter.error(exc.message, exc.hint)
        return EXIT_UPSTREAM

    if args.json:
        print(json.dumps(_jsonable(asdict(outcome)), indent=2))
        return EXIT_OK

    reporter.decision(outcome)
    reporter.gate(outcome)
    reporter.draft(outcome)
    reporter.guardrails(outcome)
    reporter.result(outcome)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - process entry point
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED) from None
