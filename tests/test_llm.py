"""
Unit tests cho module src/llm/client.py.
Kiểm tra cơ chế Caching đĩa, nhiệt độ 0 và kết nối Ollama/Gemini giả lập.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.llm.client import LLMClient, get_llm_client_from_config


def test_cache_key_generation(tmp_path):
    """Kiểm tra tạo hash key duy nhất cho prompt và tham số."""
    client = LLMClient(provider="ollama", model_name="qwen2.5:3b", cache_dir=tmp_path)
    k1 = client._generate_cache_key("Prompt A", "System 1", False)
    k2 = client._generate_cache_key("Prompt A", "System 1", False)
    k3 = client._generate_cache_key("Prompt B", "System 1", False)

    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64


def test_disk_cache_hit(tmp_path):
    """Kiểm tra đọc phản hồi từ cache đĩa mà không gọi mạng."""
    client = LLMClient(provider="ollama", model_name="qwen2.5:3b", cache_dir=tmp_path)
    key = client._generate_cache_key("Test prompt", "System", False)
    cache_file = tmp_path / f"{key}.json"

    # Ghi sẵn file cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"response": "Phản hồi giả lập từ cache"}, f)

    res = client.generate("Test prompt", "System", json_mode=False, use_cache=True)
    assert res == "Phản hồi giả lập từ cache"


@patch("requests.post")
def test_call_ollama_mock(mock_post, tmp_path):
    """Kiểm tra gọi Ollama REST API với payload chuẩn."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "{\"signal\": \"HOLD\"}"}
    }
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    client = LLMClient(provider="ollama", model_name="qwen2.5:3b", cache_dir=tmp_path)
    output = client.generate("Dự đoán xu hướng", json_mode=True, use_cache=False)

    assert "HOLD" in output
    assert mock_post.called
    payload = mock_post.call_args[1]["json"]
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0.0


def test_get_llm_client_from_config():
    """Kiểm tra khởi tạo client từ file cấu hình config.yaml."""
    client = get_llm_client_from_config("agent")
    assert client is not None
    assert client.temperature == 0.0
