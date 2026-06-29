"""BiSpikCLM bootstrap package."""

from .models.bispik_config import BiSpikConfig
from .models.bispik_lm import BiSpikForCausalLM

__all__ = ["BiSpikConfig", "BiSpikForCausalLM"]

