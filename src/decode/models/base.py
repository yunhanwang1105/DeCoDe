from __future__ import annotations

from abc import ABC, abstractmethod


class MultimodalLLM(ABC):
    """Minimal interface needed by the DeCoDe evaluator."""

    @abstractmethod
    def score_yes(self, messages: list[dict], max_new_tokens: int) -> float:
        """Return log p("Yes") for a support/query comparison prompt."""

    def score_yes_batch(self, messages_batch: list[list[dict]], max_new_tokens: int) -> list[float]:
        """Return log p("Yes") for a batch of support/query comparison prompts."""
        return [
            self.score_yes(messages=messages, max_new_tokens=max_new_tokens)
            for messages in messages_batch
        ]
