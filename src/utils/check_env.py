"""
Script kiểm tra tính đầy đủ của các biến môi trường cần thiết mà không để lộ giá trị key.
Có thể chạy trực tiếp bằng lệnh:
    python -m src.utils.check_env
"""

import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_environment() -> bool:
    """Kiểm tra biến môi trường từ .env và hệ điều hành.

    Returns:
        bool: True nếu đủ các biến bắt buộc, False nếu thiếu.
    """
    load_dotenv()

    # Kiểm tra VNSTOCK_API_KEY
    vnstock_key = os.getenv("VNSTOCK_API_KEY")

    # Kiểm tra ít nhất một LLM provider
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")

    all_passed = True

    print("=" * 60)
    print("KIỂM TRA CẤU HÌNH BIẾN MÔI TRƯỜNG DỰ ÁN (VN-LLM-AGENTS)")
    print("=" * 60)

    # 1. VNSTOCK API KEY
    if vnstock_key and vnstock_key.strip():
        print("[✓] VNSTOCK_API_KEY: Đã thiết lập (ẩn giá trị bảo mật)")
    else:
        print("[✗] VNSTOCK_API_KEY: CHƯA THIẾT LẬP! Cần thêm vào .env")
        all_passed = False

    # 2. LLM PROVIDER KEYS
    has_llm = False
    llm_found = []
    if anthropic_key and anthropic_key.strip():
        llm_found.append("Anthropic")
        has_llm = True
    if gemini_key and gemini_key.strip():
        llm_found.append("Gemini")
        has_llm = True
    if openai_key and openai_key.strip():
        llm_found.append("OpenAI")
        has_llm = True

    if has_llm:
        print(f"[✓] LLM API KEY: Đã thiết lập provider [{', '.join(llm_found)}] (ẩn giá trị bảo mật)")
    else:
        print("[✗] LLM API KEY: CHƯA THIẾT LẬP! Cần ít nhất một key (ANTHROPIC_API_KEY, GEMINI_API_KEY hoặc OPENAI_API_KEY)")
        all_passed = False

    # 3. FIRECRAWL (Tùy chọn)
    if firecrawl_key and firecrawl_key.strip():
        print("[✓] FIRECRAWL_API_KEY: Đã thiết lập (tùy chọn)")
    else:
        print("[i] FIRECRAWL_API_KEY: Chưa thiết lập (không bắt buộc, sẽ dùng BeautifulSoup crawler tự viết)")

    print("-" * 60)
    if all_passed:
        print("KẾT QUẢ: Đầy đủ các biến môi trường bắt buộc để vận hành hệ thống.")
    else:
        print("KẾT QUẢ: Thiếu biến môi trường bắt buộc. Vui lòng kiểm tra file .env!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    passed = check_environment()
    sys.exit(0 if passed else 1)
