"""
LLM 调用模块 - 基于 LangChain

统一 LLM 调用、提示词管理、结构化输出
"""
from .client import LLMClient, get_llm_client, get_light_client
from .prompts import get_prompt_manager
from .structured import QualityScoreSchema

__all__ = [
    "LLMClient",
    "get_llm_client",
    "get_light_client",
    "get_prompt_manager",
    "QualityScoreSchema",
]
