"""Token accounting and cost estimates.

With prompt caching, the API reports three kinds of input tokens, each billed differently:
  input_tokens                 - processed normally (after the last cache breakpoint)
  cache_creation_input_tokens  - written to the cache (a bit more expensive than normal input)
  cache_read_input_tokens      - read from the cache (a tenth of the normal price)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    def add(self, api_usage) -> None:
        self.input += getattr(api_usage, "input_tokens", 0) or 0
        self.output += getattr(api_usage, "output_tokens", 0) or 0
        self.cache_write += getattr(api_usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(api_usage, "cache_read_input_tokens", 0) or 0

    @property
    def total_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def cost_usd(self, pricing: dict) -> float:
        return (self.input * pricing["input"] + self.output * pricing["output"]
                + self.cache_write * pricing["cache_write"] + self.cache_read * pricing["cache_read"]) / 1e6

    def uncached_cost_usd(self, pricing: dict) -> float:
        """What the same tokens would have cost without caching."""
        return (self.total_input * pricing["input"] + self.output * pricing["output"]) / 1e6
