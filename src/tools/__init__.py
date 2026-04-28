"""Base Tool interface.

Every specialist capability the agent can invoke (segmentation, feature
extraction, molecular query, demographics, history, commit_diagnosis) is a
`Tool` subclass. Each Tool exposes:

  - `meta`: name, description, input schema, cost, latency
  - `execute(**kwargs) -> dict`: run the tool, return a structured observation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolMeta:
    """Static description of a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    cost: float
    expected_latency_s: float = 1.0
    tags: list[str] = field(default_factory=list)


class Tool(ABC):
    """Abstract specialist tool. Subclass and set `meta`."""

    meta: ToolMeta

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the tool. Returns a JSON-serializable observation dict."""
        ...

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.meta.name,
            "description": self.meta.description,
            "input_schema": self.meta.input_schema,
        }

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.meta.name,
                "description": self.meta.description,
                "parameters": self.meta.input_schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.meta.name} cost={self.meta.cost}>"


__all__ = ["Tool", "ToolMeta"]
