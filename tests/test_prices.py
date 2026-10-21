"""
Unit tests cho module src/data/prices.py.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from src.data.prices import (
    get_yearly_ranges,
    check_price_data_quality,
    verify_rights_adjustment,
    process_and_save_dataset,
    fetch_ohlcv_chunk,
)


def test_get_yearly_ranges():
    """Kiểm tra việc chia khoảng thời gian theo từng năm."""
    ranges = get_yearly_ranges("2020-01-01", "2022-06-30")
    assert len(ranges) == 3
    assert ranges[0] == ("2020-01-01", "2020-12-31")
    assert ranges[1] == ("2021-01-01", "2021-12-31")
    assert ranges[2] == ("2022-01-01", "2022-06-30")


def test_check_price_data_quality_clean():
    """Kiểm tra chất lượng với dữ liệu chuẩn."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "Open": [50.0] * 10,
        "High": [52.0] * 10,
        "Low": [49.0] * 10,
        "Close": [51.0] * 10,
        "Volume": [1000000] * 10,
    })
    report = check_price_data_quality(df, "VCB")
    assert report["status"] == "PASSED"
    assert report["total_rows"] == 10
    assert report["invalid_price_nonpositive"] == 0
    assert report["invalid_high_low"] == 0
    assert report["invalid_open_close_bound"] == 0
    assert report["extreme_daily_return_gt_8pct"] == 0
    assert report["zero_volume"] == 0


def test_check_price_data_quality_anomalies():
    """Kiểm tra phát hiện dữ liệu lỗi (giá âm, High < Low, biên độ vượt trần)."""
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Open": [50.0, -10.0, 50.0, 50.0],       # Hàng 1: giá âm
        "High": [52.0, 52.0, 48.0, 52.0],        # Hàng 2: High < Low
        "Low": [49.0, 49.0, 50.0, 49.0],
        "Close": [51.0, 51.0, 51.0, 90.0],       # Hàng 3: Tăng vọt > 8%
        "Volume": [100, 100, 0, 100],            # Hàng 2: Volume = 0
    })
    report = check_price_data_quality(df, "TEST")
    assert report["status"] == "WARNING"
    assert report["invalid_price_nonpositive"] > 0
    assert report["invalid_high_low"] > 0
    assert report["zero_volume"] > 0
    assert report["extreme_daily_return_gt_8pct"] > 0


def test_verify_rights_adjustment():
    """Kiểm tra bổ sung cột Adj Close."""
    df = pd.DataFrame({
        "Close": [50.0, 51.0, 52.0]
    })
    df_adj = verify_rights_adjustment(df, "VCB")
    assert "Adj Close" in df_adj.columns
    assert (df_adj["Adj Close"] == df_adj["Close"]).all()


def test_process_and_save_dataset(tmp_path):
    """Kiểm tra tổng hợp và lưu dataset long format kèm VNINDEX."""
    dates = pd.date_range("2023-01-02", periods=5, freq="D")
    df_vcb = pd.DataFrame({
        "date": dates,
        "Open": [50.0] * 5,
        "High": [52.0] * 5,
        "Low": [49.0] * 5,
        "Close": [51.0] * 5,
        "Volume": [1000] * 5,
    })
    df_mbb = pd.DataFrame({
        "date": dates,
        "Open": [20.0] * 5,
        "High": [21.0] * 5,
        "Low": [19.0] * 5,
        "Close": [20.5] * 5,
        "Volume": [2000] * 5,
    })
    df_vni = pd.DataFrame({
        "date": dates,
        "Open": [1100.0] * 5,
        "High": [1110.0] * 5,
        "Low": [1090.0] * 5,
        "Close": [1105.0] * 5,
        "Volume": [500000] * 5,
    })

    stock_dict = {"VCB": df_vcb, "MBB": df_mbb}
    state_owned = ["VCB"]
    private_owned = ["MBB"]

    ohlcv_long, vni_clean, quality_df = process_and_save_dataset(
        stock_dict=stock_dict,
        vni_df=df_vni,
        state_owned=state_owned,
        private_owned=private_owned
    )

    assert len(ohlcv_long) == 10  # 5 rows x 2 tickers
    assert set(ohlcv_long["ticker"].unique()) == {"VCB", "MBB"}
    assert "group" in ohlcv_long.columns
    assert ohlcv_long[ohlcv_long["ticker"] == "VCB"]["group"].iloc[0] == "state_owned"
    assert ohlcv_long[ohlcv_long["ticker"] == "MBB"]["group"].iloc[0] == "private"

    assert len(vni_clean) == 5
    assert "VNI_ret" in vni_clean.columns

    assert len(quality_df) == 3  # VCB, MBB, VNINDEX


def test_fetch_ohlcv_chunk_cache(tmp_path):
    """Kiểm tra fetch_ohlcv_chunk tải từ cache khi file đã tồn tại."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "VCB_kbs_2023-01-01_2023-12-31.parquet"

    fake_df = pd.DataFrame({
        "time": ["2023-01-02"],
        "open": [50.0],
        "high": [52.0],
        "low": [49.0],
        "close": [51.0],
        "volume": [1000]
    })
    fake_df.to_parquet(cache_file, index=False)

    with patch("src.data.prices.RAW_DATA_DIR", cache_dir):
        loaded = fetch_ohlcv_chunk("VCB", "2023-01-01", "2023-12-31", source="kbs")
        assert len(loaded) == 1
        assert loaded["close"].iloc[0] == 51.0
