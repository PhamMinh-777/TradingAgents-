"""
Unit tests cho module src/data/fundamentals.py.
"""

import pandas as pd
import pytest

from src.data.fundamentals import (
    calc_available_date,
    standardize_statements_to_long,
    compute_banking_ratios,
    get_fundamentals,
)


def test_calc_available_date():
    """Kiểm tra logic tính ngày khả dụng BCTC (Point-in-time)."""
    # 1. Quý 1 kết thúc 31/03 + 45 ngày -> 15/05
    d_q1 = calc_available_date("2023Q1", delay_quarter=45)
    assert d_q1 == pd.Timestamp("2023-05-15")

    # 2. Quý 2 kết thúc 30/06 + 45 ngày -> 14/08
    d_q2 = calc_available_date("2023-Q2", delay_quarter=45)
    assert d_q2 == pd.Timestamp("2023-08-14")

    # 3. Quý 4 kết thúc 31/12 + 90 ngày -> 31/03 năm sau
    d_q4 = calc_available_date("2023Q4", delay_annual=90)
    assert d_q4 == pd.Timestamp("2024-03-30") or d_q4 == pd.Timestamp("2024-03-31")

    # 4. Khi có ngày công bố thực tế
    d_ann = calc_available_date("2023Q1", announce_date="2023-04-25")
    assert d_ann == pd.Timestamp("2023-04-25")


def test_standardize_statements_to_long():
    """Kiểm tra chuẩn hóa bảng BCTC thô về định dạng long chuẩn."""
    raw_df = pd.DataFrame({
        "Khoản mục": ["Thu nhập lãi thuần", "Chi phí hoạt động", "Lợi nhuận sau thuế"],
        "2023Q1": [1000.0, -300.0, 500.0],
        "2023Q2": [1100.0, -320.0, 550.0],
    })
    long_df = standardize_statements_to_long("VCB", raw_df, "income")
    assert len(long_df) == 6
    assert set(long_df.columns) == {
        "ticker", "fiscal_period", "item_code", "value", "available_date", "statement_type"
    }
    assert (long_df["ticker"] == "VCB").all()
    assert (long_df["available_date"] > pd.Timestamp("2023-03-31")).all()


def test_compute_banking_ratios():
    """Kiểm tra tính toán các chỉ số đặc thù ngành ngân hàng."""
    records = [
        # Kỳ 1: 2023Q1
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Thu nhập lãi thuần", "value": 12000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "income"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Tổng thu nhập hoạt động", "value": 15000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "income"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Chi phí hoạt động", "value": -4500.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "income"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Lợi nhuận sau thuế", "value": 8000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "income"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Tổng tài sản", "value": 1600000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "balance"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Vốn chủ sở hữu", "value": 140000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "balance"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Cho vay khách hàng", "value": 1100000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "balance"},
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "Tiền gửi của khách hàng", "value": 1200000.0, "available_date": pd.Timestamp("2023-05-15"), "statement_type": "balance"},
        # Kỳ 2: 2023Q2
        {"ticker": "VCB", "fiscal_period": "2023Q2", "item_code": "Cho vay khách hàng", "value": 1150000.0, "available_date": pd.Timestamp("2023-08-15"), "statement_type": "balance"},
        {"ticker": "VCB", "fiscal_period": "2023Q2", "item_code": "Tiền gửi của khách hàng", "value": 1260000.0, "available_date": pd.Timestamp("2023-08-15"), "statement_type": "balance"},
        {"ticker": "VCB", "fiscal_period": "2023Q2", "item_code": "Tổng thu nhập hoạt động", "value": 16000.0, "available_date": pd.Timestamp("2023-08-15"), "statement_type": "income"},
        {"ticker": "VCB", "fiscal_period": "2023Q2", "item_code": "Chi phí hoạt động", "value": -4800.0, "available_date": pd.Timestamp("2023-08-15"), "statement_type": "income"},
    ]
    df = pd.DataFrame(records)
    ratios = compute_banking_ratios(df, "VCB")
    assert not ratios.empty

    items = ratios["item_code"].unique()
    assert "CIR" in items
    cir_val = ratios[ratios["item_code"] == "CIR"]["value"].iloc[0]
    assert 0.29 < cir_val < 0.31  # 4500 / 15000 = 0.30

    assert "credit_growth_qoq" in items
    credit_growth = ratios[ratios["item_code"] == "credit_growth_qoq"]["value"].iloc[0]
    assert credit_growth > 0  # 1150000 vs 1100000


def test_get_fundamentals_point_in_time():
    """Kiểm tra nghiêm ngặt get_fundamentals không rò rỉ dữ liệu tương lai."""
    records = [
        {"ticker": "VCB", "fiscal_period": "2023Q1", "item_code": "ROE", "value": 0.22, "available_date": pd.Timestamp("2023-05-15")},
        {"ticker": "VCB", "fiscal_period": "2023Q2", "item_code": "ROE", "value": 0.24, "available_date": pd.Timestamp("2023-08-15")},
        {"ticker": "VCB", "fiscal_period": "2023Q3", "item_code": "ROE", "value": 0.25, "available_date": pd.Timestamp("2023-11-15")},
    ]
    df = pd.DataFrame(records)

    # 1. Trước ngày công bố Q1 -> Không được thấy kỳ nào
    before_q1 = get_fundamentals("VCB", as_of="2023-05-14", fund_df=df)
    assert len(before_q1) == 0

    # 2. Ngày công bố Q1 -> Thấy Q1
    at_q1 = get_fundamentals("VCB", as_of="2023-05-15", fund_df=df)
    assert len(at_q1) == 1
    assert at_q1["fiscal_period"].iloc[0] == "2023Q1"

    # 3. Giữa Q1 và Q2 -> Chỉ thấy Q1
    mid_q2 = get_fundamentals("VCB", as_of="2023-07-01", fund_df=df)
    assert len(mid_q2) == 1
    assert mid_q2["fiscal_period"].iloc[0] == "2023Q1"

    # 4. Sau Q2 -> Thấy Q1 và Q2, nhưng không thấy Q3
    after_q2 = get_fundamentals("VCB", as_of="2023-08-16", fund_df=df)
    assert len(after_q2) == 2
    assert set(after_q2["fiscal_period"]) == {"2023Q1", "2023Q2"}
