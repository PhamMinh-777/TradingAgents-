"""
Unit tests cho module src/utils.
"""

import os
import pytest
import pandas as pd
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.assert_no_future import assert_no_future
from src.utils.check_env import check_environment
from src.utils.paths import get_project_root, CONFIG_PATH


def test_project_root():
    """Kiểm tra đường dẫn project root và sự tồn tại của file cấu hình."""
    root = get_project_root()
    assert root.exists()
    assert (root / "config" / "config.yaml").exists()


def test_load_config():
    """Kiểm tra việc tải file cấu hình YAML."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict)
    assert "tickers" in cfg
    assert "state_owned" in cfg["tickers"]
    assert "private" in cfg["tickers"]
    assert "VCB" in cfg["tickers"]["state_owned"]
    assert "dates" in cfg
    assert cfg["dates"]["start_date"] == "2020-01-01"
    assert cfg["dates"]["end_date"] == "2026-09-30"
    assert "trading" in cfg
    assert cfg["trading"]["horizon_h"] == 5
    assert cfg["trading"]["decision_every_n_days"] == 5


def test_setup_logger(tmp_path):
    """Kiểm tra khởi tạo logger ghi ra console và file."""
    logger = setup_logger("test_logger", log_file="test.log")
    assert logger is not None
    logger.info("Test logging message")


def test_set_seed():
    """Kiểm tra hàm cố định seed ngẫu nhiên."""
    set_seed(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"


def test_assert_no_future_valid():
    """Kiểm tra assert_no_future không báo lỗi khi dữ liệu hoàn toàn trong quá khứ."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-06-01", "2024-01-01"],
        "price": [10.0, 11.0, 12.0]
    })
    # Không ném ngoại lệ khi as_of >= mọi mốc trong df
    assert_no_future(df, as_of="2024-01-01")
    assert_no_future(df, as_of="2024-12-31")


def test_assert_no_future_violation():
    """Kiểm tra assert_no_future phát hiện và ném lỗi khi có bản ghi tương lai."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2024-05-01", "2025-01-01"],
        "price": [10.0, 11.0, 12.0]
    })
    with pytest.raises(ValueError, match="Look-ahead violation"):
        assert_no_future(df, as_of="2024-01-01")


def test_assert_no_future_empty():
    """Kiểm tra assert_no_future hoạt động bình thường với DataFrame rỗng."""
    df = pd.DataFrame()
    assert_no_future(df, as_of="2024-01-01")


def test_check_environment():
    """Kiểm tra hàm kiểm tra biến môi trường."""
    res = check_environment()
    assert isinstance(res, bool)
