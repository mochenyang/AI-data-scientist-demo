"""Runtime settings, read from environment variables (and a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("ANALYST_MODEL", "claude-opus-5")
    effort: str = os.getenv("ANALYST_EFFORT", "high")
    fallbacks: bool = _bool("ANALYST_FALLBACKS", True)
    exec_timeout: float = float(os.getenv("ANALYST_EXEC_TIMEOUT", "300"))
    max_steps: int = int(os.getenv("ANALYST_MAX_STEPS", "30"))
    max_upload_mb: int = int(os.getenv("ANALYST_MAX_UPLOAD_MB", "200"))
    workspace: Path = Path(os.getenv("ANALYST_WORKSPACE", str(PROJECT_ROOT / "workspace")))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    @property
    def has_credentials(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
                    or (Path.home() / ".config" / "anthropic").exists())


settings = Settings()
