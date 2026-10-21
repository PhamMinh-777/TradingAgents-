"""
Tải và tiền xử lý cấu hình hệ thống từ file YAML.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.utils.paths import CONFIG_PATH


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Tải cấu hình từ config.yaml.

    Args:
        config_path: Đường dẫn tùy chọn tới file cấu hình.

    Returns:
        Dict chứa toàn bộ cấu hình.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
