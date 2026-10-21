"""
Module quản lý giao tiếp với các mô hình ngôn ngữ lớn (LLM).
"""

from src.llm.client import LLMClient, get_llm_client_from_config

__all__ = ["LLMClient", "get_llm_client_from_config"]
