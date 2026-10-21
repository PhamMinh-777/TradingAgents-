"""
Thiết lập logging cho toàn hệ thống vn-llm-agents.
"""

import logging
import sys
from typing import Optional
from src.utils.paths import LOGS_DIR


def setup_logger(
    name: Optional[str] = "vn_llm_agents",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Khởi tạo và cấu hình logger.

    Args:
        name: Tên logger.
        log_file: Tên file log tùy chọn trong outputs/logs.
        level: Cấp độ logging.

    Returns:
        logging.Logger đã cấu hình.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh duplicate handlers nếu gọi nhiều lần
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler với mã hóa UTF-8 và fallback ký tự an toàn
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler nếu có yêu cầu
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
