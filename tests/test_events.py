"""
Unit tests cho module src/data/events.py.
"""

import pandas as pd
import pytest

from src.data.events import (
    classify_event_type,
    standardize_corporate_events,
    get_corporate_events,
)


def test_classify_event_type():
    """Kiểm tra phân loại loại sự kiện doanh nghiệp chuẩn xác."""
    assert classify_event_type("Cổ tức", "Trả cổ tức đợt 1 năm 2023 bằng tiền tỷ lệ 12%") == "cash_dividend"
    assert classify_event_type("Cổ tức", "Phát hành cổ phiếu để trả cổ tức tỷ lệ 15%") == "stock_dividend"
    assert classify_event_type("Thưởng", "Thưởng cổ phiếu cho cổ đông hiện hữu") == "stock_dividend"
    assert classify_event_type("Chào bán", "Chào bán quyền mua cổ phiếu cho cổ đông") == "issuance"
    assert classify_event_type("ĐHĐCĐ", "Tổ chức Đại hội đồng cổ đông thường niên 2024") == "agm_egm"
    assert classify_event_type("BCTC", "Công bố kết quả kinh doanh quý 3") == "earnings"
    assert classify_event_type("Thông báo", "Thay đổi địa chỉ trụ sở chính") == "other"


def test_standardize_corporate_events():
    """Kiểm tra chuẩn hóa bảng sự kiện về schema chuẩn."""
    raw_df = pd.DataFrame({
        "sự kiện": ["Trả cổ tức", "Họp ĐHĐCĐ"],
        "ngày công bố": ["2023-04-10", "2024-03-01"],
        "ngày gdkhq": ["2023-04-25", "2024-03-20"],
        "ngày dkcc": ["2023-04-26", "2024-03-21"],
        "tỷ lệ": [12.0, None],
        "nội dung": ["Cổ tức tiền mặt 12%", "ĐHĐCĐ thường niên 2024"]
    })
    events = standardize_corporate_events(raw_df, "VCB")
    assert len(events) == 2
    assert set(events.columns) == {
        "ticker", "event_type", "announce_date", "ex_date", "record_date", "value", "raw_text", "source"
    }
    assert events["event_type"].iloc[0] == "cash_dividend"
    assert events["event_type"].iloc[1] == "agm_egm"
    assert events["value"].iloc[0] == 12.0


def test_get_corporate_events_point_in_time():
    """Kiểm tra get_corporate_events không nhìn trước tương lai."""
    events_df = pd.DataFrame([
        {"ticker": "MBB", "event_type": "cash_dividend", "announce_date": pd.Timestamp("2023-05-10"), "ex_date": pd.Timestamp("2023-05-25"), "value": 10.0, "raw_text": "Cổ tức 10%", "source": "test"},
        {"ticker": "MBB", "event_type": "stock_dividend", "announce_date": pd.Timestamp("2023-09-01"), "ex_date": pd.Timestamp("2023-09-20"), "value": 15.0, "raw_text": "Cổ tức 15%", "source": "test"},
    ])

    # Trước sự kiện 1
    ev_early = get_corporate_events("MBB", as_of="2023-05-01", events_df=events_df)
    assert len(ev_early) == 0

    # Sau sự kiện 1, trước sự kiện 2
    ev_mid = get_corporate_events("MBB", as_of="2023-06-01", events_df=events_df)
    assert len(ev_mid) == 1
    assert ev_mid["event_type"].iloc[0] == "cash_dividend"

    # Sau cả 2 sự kiện
    ev_late = get_corporate_events("MBB", as_of="2023-09-05", events_df=events_df)
    assert len(ev_late) == 2
