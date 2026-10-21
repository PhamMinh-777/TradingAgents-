"""
Module thu thập, chuẩn hóa sự kiện doanh nghiệp (Corporate Events).
Phục vụ phân tích sự kiện (Event Study) và xác minh điều chỉnh giá.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.assert_no_future import assert_no_future
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR

logger = setup_logger("events")


def fetch_raw_events(ticker: str, source: str = "kbs") -> pd.DataFrame:
    """Tải danh sách sự kiện doanh nghiệp từ vnstock Reference API.

    Args:
        ticker: Mã chứng khoán.
        source: Nguồn dữ liệu.

    Returns:
        DataFrame sự kiện thô.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / f"{ticker}_events_raw.parquet"

    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        from vnstock import Reference
        ref = Reference()
        df = ref.company(ticker).events()
        if df is not None and not df.empty:
            df.to_parquet(cache_path, index=False)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"[{ticker}] Không thể tải sự kiện doanh nghiệp từ vnstock ({e})")
        return pd.DataFrame()


def classify_event_type(raw_type: str, raw_text: str) -> str:
    """Phân loại sự kiện thành các nhóm chuẩn:

    - cash_dividend (Cổ tức bằng tiền)
    - stock_dividend (Cổ tức bằng cổ phiếu / thưởng cổ phiếu)
    - issuance (Phát hành thêm cổ phiếu / chào bán / quyền mua)
    - agm_egm (Đại hội đồng cổ đông thường niên / bất thường)
    - earnings (Công bố kết quả kinh doanh / BCTC)
    - other (Sự kiện khác)
    """
    combined = (str(raw_type) + " " + str(raw_text)).lower()

    # 1. Phát hành / quyền mua / chào bán ưu tiên trước
    if (
        "phát hành" in combined
        or "chào bán" in combined
        or "quyền mua" in combined
        or "issuance" in combined
    ) and "trả cổ tức" not in combined and "thưởng cổ phiếu" not in combined:
        return "issuance"

    # 2. Cổ tức tiền
    if "tiền" in combined or "cash" in combined or "trả cổ tức đợt" in combined:
        return "cash_dividend"

    # 3. Cổ tức cổ phiếu / Thưởng cổ phiếu
    if (
        "cổ tức bằng cổ phiếu" in combined
        or "trả cổ tức" in combined
        or "thưởng" in combined
        or "stock dividend" in combined
        or "bonus" in combined
        or "cổ phiếu" in combined
    ):
        return "stock_dividend"

    # 4. ĐHCĐ
    if "đại hội" in combined or "đhđcđ" in combined or "agm" in combined or "egm" in combined:
        return "agm_egm"

    # 5. Kết quả kinh doanh
    if (
        "kết quả kinh doanh" in combined
        or "lợi nhuận" in combined
        or "báo cáo tài chính" in combined
        or "earnings" in combined
    ):
        return "earnings"

    return "other"


def standardize_corporate_events(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu sự kiện thô thành bảng corporate_events với các trường chuẩn:

    corporate_events(ticker, event_type, announce_date, ex_date, record_date, value, raw_text, source)
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=[
            "ticker", "event_type", "announce_date", "ex_date", "record_date", "value", "raw_text", "source"
        ])

    df = df_raw.copy()
    cols = {str(c).lower(): c for c in df.columns}

    # Tìm các cột ngày tháng
    def find_dt_col(patterns: List[str]) -> Optional[str]:
        for c in df.columns:
            for p in patterns:
                if re.search(p, str(c), re.IGNORECASE):
                    return c
        return None

    col_ann = find_dt_col(["công bố", "announce", "public", "ngày thông báo"])
    col_ex = find_dt_col(["không hưởng quyền", "ex_date", "gdkhq", "ngày gdkhq"])
    col_rec = find_dt_col(["đăng ký cuối cùng", "record_date", "dkcc"])
    col_type = find_dt_col(["loại", "type", "event_name", "sự kiện", "event"])
    col_val = find_dt_col(["tỷ lệ", "giá trị", "value", "rate", "ratio"])
    col_desc = find_dt_col(["nội dung", "ghi chú", "desc", "content", "note"])

    records = []
    for _, row in df.iterrows():
        raw_t = str(row[col_type]) if col_type else ""
        raw_d = str(row[col_desc]) if col_desc else ""
        e_type = classify_event_type(raw_t, raw_d)

        # Parse ngày tháng
        def parse_date(col_name: Optional[str]) -> Optional[pd.Timestamp]:
            if col_name and not pd.isna(row[col_name]):
                try:
                    dt = pd.to_datetime(row[col_name]).normalize()
                    if dt.year > 2000:
                        return dt
                except Exception:
                    pass
            return None

        ex_dt = parse_date(col_ex)
        ann_dt = parse_date(col_ann)
        rec_dt = parse_date(col_rec)

        # Nếu thiếu ngày công bố nhưng có ngày GDKHQ, ước tính bảo thủ ngày công bố trước đó 15 ngày
        if ann_dt is None and ex_dt is not None:
            ann_dt = ex_dt - pd.Timedelta(days=15)

        val_num = np.nan
        if col_val and not pd.isna(row[col_val]):
            try:
                val_num = float(row[col_val])
            except Exception:
                pass

        records.append({
            "ticker": ticker,
            "event_type": e_type,
            "announce_date": ann_dt,
            "ex_date": ex_dt,
            "record_date": rec_dt,
            "value": val_num,
            "raw_text": f"{raw_t}: {raw_d}".strip(": "),
            "source": "vnstock_reference"
        })

    return pd.DataFrame(records)


def get_corporate_events(
    ticker: str,
    as_of: Union[str, pd.Timestamp],
    events_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Lấy danh sách các sự kiện đã công bố trước hoặc tại mốc as_of (Point-in-time check).

    Args:
        ticker: Mã cổ phiếu.
        as_of: Mốc thời gian giới hạn.
        events_df: DataFrame sự kiện chuẩn hóa.

    Returns:
        DataFrame các sự kiện có announce_date <= as_of.
    """
    if events_df is None:
        events_path = PROCESSED_DATA_DIR / "corporate_events.parquet"
        if not events_path.exists():
            return pd.DataFrame()
        events_df = pd.read_parquet(events_path)

    as_of_ts = pd.to_datetime(as_of).normalize()
    ticker_events = events_df[events_df["ticker"] == ticker].copy()

    if ticker_events.empty:
        return pd.DataFrame()

    # Chỉ lọc các sự kiện đã có ngày công bố hợp lệ và announce_date <= as_of
    valid_ann = ticker_events["announce_date"].dropna()
    valid_mask = pd.to_datetime(ticker_events["announce_date"]).dt.normalize() <= as_of_ts
    result = ticker_events[valid_mask].copy()

    assert_no_future(result, as_of=as_of_ts, date_col="announce_date")
    return result


def build_and_save_corporate_events() -> pd.DataFrame:
    """Thu thập, chuẩn hóa và lưu trữ bảng corporate_events.parquet."""
    config = load_config()
    tickers = config["tickers"]["state_owned"] + config["tickers"]["private"]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    event_dfs = []
    coverage_rows = []

    for t in tickers:
        logger.info(f"Đang xử lý sự kiện doanh nghiệp cho {t}...")
        df_raw = fetch_raw_events(t)
        df_std = standardize_corporate_events(df_raw, t)
        event_dfs.append(df_std)

        coverage_rows.append({
            "ticker": t,
            "total_events": len(df_std),
            "cash_dividends": int((df_std["event_type"] == "cash_dividend").sum()),
            "stock_dividends": int((df_std["event_type"] == "stock_dividend").sum()),
            "issuance": int((df_std["event_type"] == "issuance").sum()),
            "agm_egm": int((df_std["event_type"] == "agm_egm").sum()),
            "other": int((df_std["event_type"] == "other").sum()),
        })

    if event_dfs:
        corporate_events = pd.concat(event_dfs, ignore_index=True)
    else:
        corporate_events = pd.DataFrame(columns=[
            "ticker", "event_type", "announce_date", "ex_date", "record_date", "value", "raw_text", "source"
        ])

    out_file = PROCESSED_DATA_DIR / "corporate_events.parquet"
    corporate_events.to_parquet(out_file, index=False)

    cov_df = pd.DataFrame(coverage_rows)
    cov_path = TABLES_DIR / "events_coverage.csv"
    cov_df.to_csv(cov_path, index=False, encoding="utf-8-sig")

    logger.info(f"Đã lưu sự kiện doanh nghiệp: {out_file} ({len(corporate_events)} dòng)")
    logger.info(f"Đã lưu báo cáo độ phủ sự kiện: {cov_path}")
    return corporate_events


if __name__ == "__main__":
    build_and_save_corporate_events()
