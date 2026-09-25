"""JEV AI Quickstart - a confidence-gated AI agent in under five minutes.

The package is split into small layers so each one can be read, tested or
replaced on its own:

``src.config``
    Loads ``.env`` and validates every setting with Pydantic.
``src.errors``
    The user-facing exception hierarchy shared by all layers.
``src.decisions``
    The Jev layer: typed decisions (Choice / Score / Noul) with confidence.
``src.writer``
    The generative layer: turns a decision into prose.
``src.pipeline``
    Orchestration: triage -> confidence gate -> draft -> guardrail.
``src.main``
    CLI entry point that wires the layers together and prints the result.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
