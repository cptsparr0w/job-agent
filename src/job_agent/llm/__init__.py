"""LLM clients and routing.

The router is the architecturally interesting bit: it picks Qwen or Claude
per task tier and logs every call to the cost ledger.
"""

from .router import LLMRouter, ModelTier
from .qwen import QwenClient

__all__ = ["LLMRouter", "ModelTier", "ClaudeClient", "QwenClient"]
