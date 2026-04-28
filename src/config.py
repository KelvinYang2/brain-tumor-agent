"""Centralized configuration. Loads from .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings, populated from .env at the repo root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Data paths -----------------------------------------------------
    brats_dir: Path | None = Field(default=None)
    kaist_pred_dir: Path | None = Field(default=None)
    upenn_gbm_dir: Path | None = Field(default=None)
    ucsf_pdgm_dir: Path | None = Field(default=None)

    # --- LLM endpoints --------------------------------------------------
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

    # --- Tracking -------------------------------------------------------
    wandb_api_key: str | None = Field(default=None)
    wandb_project: str = "brain-tumor-agent"
    wandb_entity: str | None = Field(default=None)

    # --- Misc -----------------------------------------------------------
    log_level: str = "INFO"

    def require(self, *fields: str) -> None:
        """Raise if any of the named fields is unset. Use at script entry points."""
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise RuntimeError(
                f"Required settings missing in .env: {missing}. "
                f"Copy .env.example to .env and fill them in."
            )


settings = Settings()
