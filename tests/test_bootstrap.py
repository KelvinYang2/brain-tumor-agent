"""Smoke test: package imports and config loads."""

from __future__ import annotations


def test_package_imports() -> None:
    import src

    assert src.__version__ == "0.1.0"


def test_settings_loads() -> None:
    from src.config import settings

    assert settings.vllm_base_url.startswith("http")
    assert settings.wandb_project == "brain-tumor-agent"


def test_tool_base_imports() -> None:
    from src.tools import Tool, ToolMeta

    assert Tool is not None
    assert ToolMeta is not None
