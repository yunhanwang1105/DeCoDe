from __future__ import annotations

from decode.config import ModelConfig
from decode.models.base import MultimodalLLM


def build_model(config: ModelConfig) -> MultimodalLLM:
    if config.backend == "qwen3vl":
        from decode.models.qwen3_vl import Qwen3VLAdapter

        return Qwen3VLAdapter(config)
    raise ValueError(f"Unsupported model backend: {config.backend}")
