"""
LLM 客户端 - 基于 LangChain 的统一 LLM 调用接口
"""
import os
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from PIL import Image
import httpx

# LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 项目配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENROUTER_API_KEY, LLM_MODEL, LIGHT_LLM_MODEL


class LLMClient:
    """统一的 LLM 客户端"""

    DEFAULT_MODEL = LLM_MODEL
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    TIMEOUT = 300

    def __init__(
        self,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=httpx.Timeout(self.TIMEOUT, connect=60),
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": "https://nano-banana-milvus.localhost",
                    "X-Title": "Nano Banana Milvus"
                }
            }
        )

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

    def invoke(
        self,
        text: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """调用文本模型"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=text))

        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        response = self._llm.invoke(messages, config=config if config else None)
        return response.content

    def invoke_with_images(
        self,
        text: str,
        images: List[Image.Image],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """调用多模态模型"""
        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})

        from utils.core import image_to_uri
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_uri(img)}
            })

        content.append({"type": "text", "text": text})
        messages = [HumanMessage(content=content)]

        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        response = self._llm.invoke(messages, config=config if config else None)
        return response.content

    def invoke_structured(
        self,
        text: str,
        schema: Type[Any],
        images: List[Image.Image] = None,
        system_prompt: str = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """调用模型并返回结构化输出"""
        structured_llm = self._llm.with_structured_output(schema)

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if images:
            content = []
            from utils.core import image_to_uri
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_uri(img)}
                })
            content.append({"type": "text", "text": text})
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=text))

        config = {}
        if temperature is not None:
            config["temperature"] = temperature

        result = structured_llm.invoke(messages, config=config if config else None)

        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif hasattr(result, 'dict'):
            return result.dict()
        return result

    def switch_model(self, model: str) -> "LLMClient":
        """切换到新模型"""
        return LLMClient(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model})"


# 单例缓存
_client_cache: Dict[str, LLMClient] = {}


def get_llm_client(model: str = None) -> LLMClient:
    """获取 LLM 客户端实例"""
    cache_key = f"{model or 'default'}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = LLMClient(model=model)
    return _client_cache[cache_key]


def get_light_client() -> LLMClient:
    """获取轻量级客户端"""
    return get_llm_client(model=LIGHT_LLM_MODEL)
