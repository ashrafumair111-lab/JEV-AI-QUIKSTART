"""Environment loading and validation for the JEV AI Quickstart.

Why this module exists
----------------------
A quickstart should fail within five seconds with a message the reader can act
on - not ninety seconds later with an opaque ``401`` from an HTTP client.

* ``.env`` is loaded into ``os.environ`` **before** any SDK is imported, so the
  official ``typesafe-sdk`` picks up ``TYPESAFE_API_KEY``, ``TYPESAFE_BASE_URL``
  and ``TYPESAFE_DEFAULT_MODEL`` by itself.
* Every setting is validated once, at start-up, by Pydantic. Nothing else in
  the application calls ``os.getenv``.
* The placeholder values shipped in ``.env.example`` (``..._key_here``) are
  treated as *absent*. A fresh clone therefore reports "set your key" instead of
  "invalid API key".
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final, Mapping

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from src.errors import ConfigurationError

__all__ = [
    "ConfigurationError",
    "DEFAULT_ENV_FILE",
    "JEV_API_KEY_ALIASES",
    "LogLevel",
    "PROJECT_ROOT",
    "SDK_API_KEY_ENV",
    "Settings",
    "build_settings",
    "configure_logging",
    "get_settings",
    "load_env_file",
    "mask_secret",
    "runtime_warnings",
    "verify_ready",
]

LOGGER: Final[logging.Logger] = logging.getLogger("jevai.config")

#: Repository root - the directory holding ``src/`` and ``.env.example``.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Default location of the local secrets file.
DEFAULT_ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

#: Accepted spellings of the Jev credential, in priority order. The first is the
#: name this project documents; the second is what the SDK reads natively.
JEV_API_KEY_ALIASES: Final[tuple[str, ...]] = ("JEV_AI_API_KEY", "TYPESAFE_API_KEY")

#: The single name the official ``typesafe-sdk`` reads from the environment.
SDK_API_KEY_ENV: Final[str] = "TYPESAFE_API_KEY"

#: Substrings that mark a value as "still the example placeholder".
_PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "your_",
    "your-",
    "key_here",
    "key-here",
    "changeme",
    "placeholder",
    "xxxx",
)

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})


class LogLevel(str, Enum):
    """Allowed values for ``JEV_AI_LOG_LEVEL``."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseModel):
    """Validated, immutable snapshot of the process environment.

    Instances are built by :func:`build_settings`; everything downstream reads
    this object instead of ``os.environ``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- credentials ---------------------------------------------------------
    jev_api_key: SecretStr | None = Field(default=None)
    openai_api_key: SecretStr | None = Field(default=None)

    # --- model selection -----------------------------------------------------
    jev_model: str = Field(default="jev-latest")
    openai_model: str = Field(default="gpt-4o-mini")

    # --- behaviour -----------------------------------------------------------
    offline: bool = Field(default=False)
    confidence_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    auto_act_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    jev_base_url: str | None = Field(default=None)

    # --- validators ----------------------------------------------------------
    @field_validator("jev_model", "openai_model")
    @classmethod
    def _no_blank_model_names(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("jev_base_url")
    @classmethod
    def _strip_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().rstrip("/")
        return cleaned or None

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> "Settings":
        """The auto-act threshold may not sit below the escalation floor."""
        if self.auto_act_threshold < self.confidence_floor:
            raise ValueError(
                "JEV_AI_AUTO_ACT_THRESHOLD "
                f"({self.auto_act_threshold}) must be greater than or equal to "
                f"JEV_AI_CONFIDENCE_FLOOR ({self.confidence_floor})"
            )
        return self

    # --- derived helpers -----------------------------------------------------
    @property
    def has_jev_key(self) -> bool:
        """True when a usable (non-placeholder) Jev credential was found."""
        return self.jev_api_key is not None

    @property
    def has_openai_key(self) -> bool:
        """True when a usable OpenAI credential was found."""
        return self.openai_api_key is not None

    @property
    def is_live(self) -> bool:
        """True when real API calls are expected (i.e. not offline)."""
        return not self.offline

    def safe_summary(self) -> dict[str, str]:
        """Return a redacted view of the configuration, safe to print or log."""
        return {
            "mode": "offline (canned decisions)" if self.offline else "live (Jev API)",
            "JEV_AI_API_KEY": mask_secret(self.jev_api_key),
            "OPENAI_API_KEY": mask_secret(self.openai_api_key),
            "JEV_AI_MODEL": self.jev_model,
            "OPENAI_MODEL": self.openai_model,
            "confidence floor": f"{self.confidence_floor:.2f}",
            "auto-act threshold": f"{self.auto_act_threshold:.2f}",
            "timeout": f"{self.request_timeout_seconds:g}s",
            "log level": self.log_level.value,
        }


def mask_secret(secret: SecretStr | None, *, keep: int = 4) -> str:
    """Return a log-safe rendering of a secret, e.g. ``"jev_...9f3c"``.

    Args:
        secret: The value to mask, or ``None`` when it was not configured.
        keep: Number of leading and trailing characters to leave visible.
    """
    if secret is None:
        return "(not set)"
    raw = secret.get_secret_value()
    if not raw:
        return "(empty)"
    if len(raw) <= keep * 2:
        return "*" * len(raw)
    return f"{raw[:keep]}...{raw[-keep:]}"


def _clean(raw: str | None) -> str | None:
    """Strip whitespace and reject empty or obvious placeholder values."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        LOGGER.debug("Ignoring placeholder value found in the environment.")
        return None
    return cleaned


def _read_secret(env: Mapping[str, str], names: tuple[str, ...]) -> SecretStr | None:
    """Return the first non-placeholder value among ``names``."""
    for name in names:
        value = _clean(env.get(name))
        if value is not None:
            return SecretStr(value)
    return None


def _read_text(env: Mapping[str, str], name: str, *, default: str) -> str:
    """Return a stripped string value, falling back to ``default``."""
    return _clean(env.get(name)) or default


def _read_bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    """Return a boolean flag accepting ``1/true/yes/on`` (case-insensitive)."""
    raw = _clean(env.get(name))
    return default if raw is None else raw.lower() in _TRUTHY


def _read_float(env: Mapping[str, str], name: str, *, default: float) -> float:
    """Return a float value, raising a friendly error when it is not numeric."""
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a number but was {raw!r}.",
            hint="See .env.example for the expected format of every variable.",
        ) from exc


def build_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build a :class:`Settings` object from an environment mapping.

    Passing an explicit mapping keeps this function pure, which is what the test
    suite uses; at runtime it defaults to ``os.environ``.

    Raises:
        ConfigurationError: If a value cannot be parsed or fails validation.
    """
    env = os.environ if environ is None else environ
    try:
        return Settings(
            jev_api_key=_read_secret(env, JEV_API_KEY_ALIASES),
            openai_api_key=_read_secret(env, ("OPENAI_API_KEY",)),
            jev_model=_read_text(env, "JEV_AI_MODEL", default="jev-latest"),
            openai_model=_read_text(env, "OPENAI_MODEL", default="gpt-4o-mini"),
            offline=_read_bool(env, "JEV_AI_OFFLINE", default=False),
            confidence_floor=_read_float(env, "JEV_AI_CONFIDENCE_FLOOR", default=0.50),
            auto_act_threshold=_read_float(env, "JEV_AI_AUTO_ACT_THRESHOLD", default=0.85),
            request_timeout_seconds=_read_float(
                env, "JEV_AI_REQUEST_TIMEOUT_SECONDS", default=30.0
            ),
            log_level=_read_text(env, "JEV_AI_LOG_LEVEL", default="INFO"),
            jev_base_url=_clean(env.get("TYPESAFE_BASE_URL")),
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])} {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration: {problems}",
            hint=(
                "Compare your .env against .env.example - the accepted type and "
                "range of every variable is documented there."
            ),
        ) from exc


def load_env_file(env_file: Path | None = DEFAULT_ENV_FILE) -> Path | None:
    """Load a ``.env`` file into ``os.environ`` and bridge the variable aliases.

    Existing process variables always win (``override=False``), so CI systems and
    containers stay in control. A missing file is not an error: offline mode and
    process-level variables are both perfectly valid setups.

    Args:
        env_file: Path to the file to load, or ``None`` to skip file loading.

    Returns:
        The path that was loaded, or ``None`` when there was nothing to load.
    """
    loaded: Path | None = None
    if env_file is not None:
        candidate = Path(env_file).expanduser()
        if candidate.is_file():
            load_dotenv(dotenv_path=candidate, override=False)
            loaded = candidate
        else:
            LOGGER.debug("No env file at %s - using the process environment.", candidate)
    _bridge_aliases()
    return loaded


def _bridge_aliases() -> None:
    """Expose project-prefixed variables under the names the SDK expects.

    ``typesafe-sdk`` reads ``TYPESAFE_*`` when it is imported, so anything the
    reader configured as ``JEV_AI_*`` is mirrored before that import happens.
    """
    aliases: tuple[tuple[str, str], ...] = (
        ("JEV_AI_API_KEY", SDK_API_KEY_ENV),
        ("JEV_AI_BASE_URL", "TYPESAFE_BASE_URL"),
        ("JEV_AI_MODEL", "TYPESAFE_DEFAULT_MODEL"),
    )
    for source, target in aliases:
        if target in os.environ:
            continue
        value = _clean(os.environ.get(source))
        if value:
            os.environ[target] = value
    if "TYPESAFE_LOG_LEVEL" not in os.environ:
        level = _clean(os.environ.get("JEV_AI_LOG_LEVEL"))
        if level:
            os.environ["TYPESAFE_LOG_LEVEL"] = level.lower()


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return build_settings(os.environ)


def get_settings(
    *,
    env_file: Path | None = DEFAULT_ENV_FILE,
    reload: bool = False,
) -> Settings:
    """Load the environment file, then return the validated settings singleton.

    Args:
        env_file: ``.env`` path to load, or ``None`` to read ``os.environ`` only.
        reload: Drop the cached instance first (the CLI and tests pass ``True``).

    Raises:
        ConfigurationError: If the environment content is invalid.
    """
    if reload:
        _cached_settings.cache_clear()
    load_env_file(env_file)
    return _cached_settings()


def verify_ready(settings: Settings) -> None:
    """Fail fast when the environment cannot run the live pipeline.

    Offline mode needs no credentials at all, so it is always considered ready.

    Raises:
        ConfigurationError: When a required credential is absent.
    """
    if settings.offline:
        return
    if not settings.has_jev_key:
        raise ConfigurationError(
            "No Jev API key found, so live mode cannot start.",
            hint=(
                "1. Create a key at https://console.typesafe.ai/keys\n"
                "2. Copy .env.example to .env and set JEV_AI_API_KEY=<your key>\n"
                "3. Or run this demo with --offline to see the whole pipeline "
                "working with no credentials and no network calls."
            ),
        )


def runtime_warnings(settings: Settings) -> list[str]:
    """Return non-fatal problems worth mentioning before the run starts."""
    warnings: list[str] = []
    if settings.offline:
        warnings.append(
            "Offline mode: decisions come from the built-in canned engine, not from Jev."
        )
    elif not settings.has_openai_key:
        warnings.append(
            "OPENAI_API_KEY is not set: the reply is written by the deterministic "
            "template writer instead of an LLM."
        )
    return warnings


def configure_logging(level: LogLevel = LogLevel.INFO) -> None:
    """Configure root logging once, with a compact format for the terminal."""
    logging.basicConfig(
        level=level.value,
        format="%(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
