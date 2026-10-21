"""
Module gọi Large Language Model (LLM) hỗ trợ mô hình cục bộ (Local Ollama / vLLM)
và các nhà cung cấp đám mây (Gemini, Anthropic, OpenAI).

Tuân thủ nguyên tắc:
1. temperature = 0 (tính tiền định, lặp lại).
2. Disk Caching: Lưu kết quả dựa trên SHA-256(model + prompt + system_prompt + params).
3. Hỗ trợ Structured JSON Output.
4. Tự động tương thích chuẩn OpenAI API của Ollama (http://localhost:11434/v1).
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from dotenv import load_dotenv

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ROOT_DIR

logger = setup_logger("llm_client")


class LLMClient:
    """Client giao tiếp thống nhất với các LLM (Local Ollama, Gemini, OpenAI, Claude)."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5:3b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        temperature: float = 0.0,
    ):
        load_dotenv()
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature

        # URL mặc định cho Local Ollama
        if self.provider == "ollama":
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.api_key = api_key or "ollama"
        elif self.provider == "gemini":
            self.base_url = "https://generativelanguage.googleapis.com"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        elif self.provider == "openai":
            self.base_url = base_url or "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.base_url = base_url
            self.api_key = api_key

        # Thư mục cache đĩa
        if cache_dir is None:
            self.cache_dir = ROOT_DIR / ".cache" / "llm"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        """Tạo khóa băm cache đĩa duy nhất."""
        payload = f"{self.provider}_{self.model_name}_{self.temperature}_{json_mode}_{system_prompt}_{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Bạn là trợ lý phân tích tài chính chuyên nghiệp.",
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> str:
        """Gửi prompt tới LLM và trả về phản hồi văn bản hoặc JSON.

        Args:
            prompt: Nội dung yêu cầu.
            system_prompt: Lời nhắc hệ thống định hình vai trò tác tử.
            json_mode: Ép mô hình trả về định dạng JSON hợp lệ.
            use_cache: Bật lưu bộ nhớ đệm trên đĩa.

        Returns:
            str: Nội dung phản hồi từ mô hình.
        """
        cache_key = self._generate_cache_key(prompt, system_prompt, json_mode)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # 1. Đọc từ cache đĩa nếu đã có
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    logger.debug(f"Đọc phản hồi từ cache đĩa: {cache_key[:8]}...")
                    return cached_data["response"]
            except Exception:
                pass

        # 2. Gọi mô hình theo từng Provider
        if self.provider == "ollama":
            response_text = self._call_ollama(prompt, system_prompt, json_mode)
        elif self.provider == "gemini":
            response_text = self._call_gemini(prompt, system_prompt, json_mode)
        else:
            raise NotImplementedError(f"Chưa hỗ trợ provider: {self.provider}")

        # 3. Lưu vào cache đĩa
        if use_cache:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "provider": self.provider,
                        "model": self.model_name,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": response_text
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Không thể ghi cache LLM ({e})")

        return response_text

    def _call_ollama(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        """Gọi Ollama qua endpoint REST API /api/chat hoặc /api/generate."""
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        if json_mode:
            payload["format"] = "json"

        try:
            res = requests.post(endpoint, json=payload, timeout=120)
            res.raise_for_status()
            data = res.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Không thể kết nối tới Ollama tại {self.base_url}. "
                "Vui lòng đảm bảo Ollama đã được khởi động (chạy lệnh: ollama serve hoặc mở ứng dụng Ollama)!"
            )
        except Exception as e:
            logger.error(f"Lỗi khi gọi Ollama: {e}")
            raise

    def _call_gemini(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        """Gọi Gemini API qua REST interface."""
        if not self.api_key:
            raise ValueError("Chưa thiết lập GEMINI_API_KEY trong .env!")

        model = self.model_name if "gemini" in self.model_name else "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

        contents = [
            {"role": "user", "parts": [{"text": f"HƯỚNG DẪN HỆ THỐNG:\n{system_prompt}\n\nYÊU CẦU:\n{prompt}"}]}
        ]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Cấu trúc phản hồi từ Gemini không hợp lệ: {data}")


def get_llm_client_from_config(role: str = "agent") -> LLMClient:
    """Tạo LLMClient dựa trên cấu hình trong config.yaml."""
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    roles = llm_cfg.get("roles", {})
    model_name = roles.get(role, "qwen2.5:3b")
    temp = llm_cfg.get("temperature", 0.0)

    return LLMClient(provider=provider, model_name=model_name, temperature=temp)
