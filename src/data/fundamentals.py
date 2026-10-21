"""
Module thu thập, chuẩn hóa báo cáo tài chính và tính toán chỉ số đặc thù ngân hàng.
Đảm bảo nguyên tắc Point-in-time: Tuyệt đối không dùng ngày kết thúc kỳ kế toán làm ngày khả dụng.
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

logger = setup_logger("fundamentals")


def calc_available_date(
    fiscal_period: str,
    announce_date: Optional[Union[str, pd.Timestamp]] = None,
    delay_quarter: int = 45,
    delay_annual: int = 90
) -> pd.Timestamp:
    """Xác định ngày thông tin BCTC chính thức khả dụng cho thị trường (Point-in-time).

    Nguyên tắc:
    - Nếu có ngày công bố thực tế (announce_date), sử dụng announce_date.
    - Nếu không có, áp dụng độ trễ bảo thủ theo quy định công bố thông tin:
      + Báo cáo quý: Ngày kết thúc quý + delay_quarter (mặc định 45 ngày).
      + Báo cáo năm (Q4): Ngày kết thúc năm + delay_annual (mặc định 90 ngày).

    Args:
        fiscal_period: Chuỗi định dạng kỳ như '2023Q1', '2023-Q1', '2023Q4', '2023'.
        announce_date: Ngày công bố thực tế nếu có.
        delay_quarter: Số ngày trễ tối thiểu cho quý 1, 2, 3.
        delay_annual: Số ngày trễ tối thiểu cho quý 4 / báo cáo năm.

    Returns:
        pd.Timestamp: Ngày thông tin có hiệu lực trên thị trường.
    """
    if announce_date and not pd.isna(announce_date):
        dt = pd.to_datetime(announce_date).normalize()
        if dt.year > 2000:
            return dt

    period_str = str(fiscal_period).upper().strip()
    match = re.search(r"(\d{4})[-_ ]?Q?([1-4])?", period_str)
    if not match:
        raise ValueError(f"Không thể phân tích kỳ tài chính từ: {fiscal_period}")

    year = int(match.group(1))
    quarter = int(match.group(2)) if match.group(2) else 4

    quarter_ends = {
        1: pd.Timestamp(f"{year}-03-31"),
        2: pd.Timestamp(f"{year}-06-30"),
        3: pd.Timestamp(f"{year}-09-30"),
        4: pd.Timestamp(f"{year}-12-31"),
    }
    period_end = quarter_ends[quarter]
    delay = delay_annual if quarter == 4 else delay_quarter
    return (period_end + pd.Timedelta(days=delay)).normalize()


def fetch_raw_financials(ticker: str, period: str = "quarter") -> Dict[str, pd.DataFrame]:
    """Tải 3 báo cáo tài chính (KQKD, CĐKT, LCTT) từ vnstock với cơ chế cache đĩa.

    Args:
        ticker: Mã chứng khoán ngân hàng (VCB, BID, MBB...).
        period: 'quarter' hoặc 'year'.

    Returns:
        Dict chứa 3 DataFrames: 'income', 'balance', 'cash_flow'.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / f"{ticker}_fund_{period}.parquet"

    # Đọc từ cache nếu có
    if cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            return {"combined": cached_df}
        except Exception:
            pass

    results = {}
    try:
        from vnstock import Fundamental
        f = Fundamental()

        # KQKD
        try:
            results["income"] = f.equity(ticker).income_statement(period=period)
        except Exception as e:
            logger.warning(f"[{ticker}] Không thể tải KQKD ({e})")
            results["income"] = pd.DataFrame()

        # CĐKT
        try:
            results["balance"] = f.equity(ticker).balance_sheet(period=period)
        except Exception as e:
            logger.warning(f"[{ticker}] Không thể tải CĐKT ({e})")
            results["balance"] = pd.DataFrame()

        # LCTT
        try:
            results["cash_flow"] = f.equity(ticker).cash_flow(period=period)
        except Exception as e:
            logger.warning(f"[{ticker}] Không thể tải LCTT ({e})")
            results["cash_flow"] = pd.DataFrame()

    except Exception as e:
        logger.warning(f"[{ticker}] vnstock Fundamental lỗi hoặc giới hạn cấp quyền: {e}")
        results["income"] = pd.DataFrame()
        results["balance"] = pd.DataFrame()
        results["cash_flow"] = pd.DataFrame()

    return results


def standardize_statements_to_long(
    ticker: str,
    df_statement: pd.DataFrame,
    statement_type: str,
    delay_quarter: int = 45,
    delay_annual: int = 90
) -> pd.DataFrame:
    """Chuẩn hóa bảng báo cáo tài chính thành định dạng long chuẩn:

    fund_quarterly(ticker, fiscal_period, item_code, value, available_date)
    """
    if df_statement is None or df_statement.empty:
        return pd.DataFrame(columns=["ticker", "fiscal_period", "item_code", "value", "available_date"])

    df = df_statement.copy()
    cols = [str(c) for c in df.columns]

    # Nhận diện cột kỳ và cột chỉ tiêu
    period_cols = [c for c in cols if re.search(r"\d{4}", c)]
    item_cols = [c for c in cols if not re.search(r"\d{4}", c)]

    item_col = item_cols[0] if item_cols else cols[0]

    records = []
    for _, row in df.iterrows():
        item_code = str(row[item_col]).strip()
        for p_col in period_cols:
            val = row[p_col]
            if pd.isna(val):
                continue
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                continue

            avail_dt = calc_available_date(
                p_col,
                delay_quarter=delay_quarter,
                delay_annual=delay_annual
            )

            records.append({
                "ticker": ticker,
                "fiscal_period": p_col,
                "item_code": item_code,
                "value": numeric_val,
                "available_date": avail_dt,
                "statement_type": statement_type
            })

    return pd.DataFrame(records)


def compute_banking_ratios(long_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Tính toán các chỉ số tài chính đặc thù của ngành ngân hàng:

    1. Tăng trưởng tín dụng (Credit growth YoY)
    2. Tăng trưởng huy động (Deposit growth YoY)
    3. ROE (Lợi nhuận sau thuế / Vốn chủ sở hữu)
    4. ROA (Lợi nhuận sau thuế / Tổng tài sản)
    5. NIM xấp xỉ (Thu nhập lãi thuần / Tổng tài sản)
    6. CIR (Chi phí hoạt động / Tổng thu nhập hoạt động)
    7. Dự phòng / Lợi nhuận trước dự phòng (Provisions / PPOP)
    8. Tỷ trọng thu nhập ngoài lãi (Non-interest income ratio)

    Args:
        long_df: DataFrame fund_quarterly chứa item_code và value.
        ticker: Mã ngân hàng.

    Returns:
        DataFrame chứa các chỉ số đã tính toán kèm available_date.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["ticker", "fiscal_period", "item_code", "value", "available_date"])

    # Pivot để tiện tính toán theo từng kỳ
    piv = long_df.pivot_table(
        index=["fiscal_period", "available_date"],
        columns="item_code",
        values="value",
        aggfunc="first"
    ).reset_index()

    piv = piv.sort_values("available_date").reset_index(drop=True)
    metric_records = []

    def find_col(patterns: List[str]) -> Optional[str]:
        for col in piv.columns:
            for pat in patterns:
                if re.search(pat, str(col), re.IGNORECASE):
                    return col
        return None

    col_loan = find_col(["cho vay khách hàng", "customer loans", "cho vay"])
    col_deposit = find_col(["tiền gửi của khách hàng", "customer deposits", "tiền gửi"])
    col_asset = find_col(["tổng tài sản", "total assets"])
    col_equity = find_col(["vốn chủ sở hữu", "equity", "vốn và các quỹ"])
    col_net_profit = find_col(["lợi nhuận sau thuế", "lợi nhuận thuần sau thuế", "net profit"])
    col_net_interest = find_col(["thu nhập lãi thuần", "net interest income"])
    col_total_income = find_col(["tổng thu nhập hoạt động", "total operating income"])
    col_opex = find_col(["chi phí hoạt động", "operating expenses"])
    col_provision = find_col(["chi phí dự phòng", "provision expense", "dự phòng rủi ro tín dụng"])

    for idx, row in piv.iterrows():
        p = row["fiscal_period"]
        dt = row["available_date"]

        # Helper thêm metric
        def add_ratio(code: str, val: float):
            if not np.isnan(val) and not np.isinf(val):
                metric_records.append({
                    "ticker": ticker,
                    "fiscal_period": p,
                    "item_code": code,
                    "value": float(val),
                    "available_date": dt,
                    "statement_type": "ratio"
                })

        # ROE & ROA
        if col_net_profit and col_equity and row[col_equity] > 0:
            add_ratio("ROE_quarterly", (row[col_net_profit] / row[col_equity]) * 4)  # Năm hóa x4
        if col_net_profit and col_asset and row[col_asset] > 0:
            add_ratio("ROA_quarterly", (row[col_net_profit] / row[col_asset]) * 4)

        # NIM xấp xỉ
        if col_net_interest and col_asset and row[col_asset] > 0:
            add_ratio("NIM_approx", (row[col_net_interest] / row[col_asset]) * 4)

        # CIR (Cost-to-Income Ratio)
        if col_opex and col_total_income and row[col_total_income] > 0:
            add_ratio("CIR", abs(row[col_opex]) / row[col_total_income])

        # Tỷ trọng thu nhập ngoài lãi
        if col_net_interest and col_total_income and row[col_total_income] > 0:
            non_interest = row[col_total_income] - row[col_net_interest]
            add_ratio("non_interest_income_ratio", non_interest / row[col_total_income])

        # Dự phòng / PPOP
        if col_provision and col_total_income and col_opex:
            ppop = row[col_total_income] - abs(row[col_opex])
            if ppop > 0:
                add_ratio("provision_to_ppop", abs(row[col_provision]) / ppop)

        # Tăng trưởng tín dụng & huy động QoQ
        if idx > 0:
            prev_row = piv.iloc[idx - 1]
            if col_loan and prev_row[col_loan] > 0 and row[col_loan] > 0:
                credit_growth_qoq = (row[col_loan] - prev_row[col_loan]) / prev_row[col_loan]
                add_ratio("credit_growth_qoq", credit_growth_qoq)

            if col_deposit and prev_row[col_deposit] > 0 and row[col_deposit] > 0:
                deposit_growth_qoq = (row[col_deposit] - prev_row[col_deposit]) / prev_row[col_deposit]
                add_ratio("deposit_growth_qoq", deposit_growth_qoq)

    return pd.DataFrame(metric_records)


def get_fundamentals(
    ticker: str,
    as_of: Union[str, pd.Timestamp],
    fund_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Truy vấn dữ liệu tài chính khả dụng tại mốc thời gian as_of (Tuân thủ nghiêm ngặt Point-in-time).

    Args:
        ticker: Mã ngân hàng.
        as_of: Mốc thời gian điểm nhìn ('YYYY-MM-DD').
        fund_df: DataFrame fund_quarterly (nếu None sẽ đọc từ processed).

    Returns:
        DataFrame các khoản mục và chỉ số tài chính có available_date <= as_of.
    """
    if fund_df is None:
        fund_path = PROCESSED_DATA_DIR / "fund_quarterly.parquet"
        if not fund_path.exists():
            logger.warning("Chưa có file fund_quarterly.parquet.")
            return pd.DataFrame()
        fund_df = pd.read_parquet(fund_path)

    as_of_ts = pd.to_datetime(as_of).normalize()
    ticker_df = fund_df[fund_df["ticker"] == ticker].copy()

    if ticker_df.empty:
        return pd.DataFrame()

    ticker_df["available_date"] = pd.to_datetime(ticker_df["available_date"]).dt.normalize()
    valid_mask = ticker_df["available_date"] <= as_of_ts
    result = ticker_df[valid_mask].copy()

    # Kiểm tra đảm bảo không có rò rỉ tương lai
    assert_no_future(result, as_of=as_of_ts, date_col="available_date")
    return result


def build_and_save_fundamentals() -> pd.DataFrame:
    """Xây dựng và lưu trữ bộ dữ liệu tài chính chuẩn hóa cho toàn bộ danh sách mã."""
    config = load_config()
    tickers = config["tickers"]["state_owned"] + config["tickers"]["private"]
    delay_q = config["data"].get("fundamentals_delay_days_quarter", 45)
    delay_y = config["data"].get("fundamentals_delay_days_annual", 90)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    coverage_rows = []

    for t in tickers:
        logger.info(f"Đang xử lý BCTC cho {t}...")
        raws = fetch_raw_financials(t, period="quarter")

        df_inc = standardize_statements_to_long(t, raws.get("income"), "income", delay_q, delay_y)
        df_bal = standardize_statements_to_long(t, raws.get("balance"), "balance", delay_q, delay_y)
        df_cf = standardize_statements_to_long(t, raws.get("cash_flow"), "cash_flow", delay_q, delay_y)

        t_base = pd.concat([df_inc, df_bal, df_cf], ignore_index=True)
        if not t_base.empty:
            t_ratios = compute_banking_ratios(t_base, t)
            t_combined = pd.concat([t_base, t_ratios], ignore_index=True)
        else:
            t_combined = pd.DataFrame()

        all_records.append(t_combined)

        # Thống kê độ phủ
        periods_found = t_combined["fiscal_period"].unique().tolist() if not t_combined.empty else []
        coverage_rows.append({
            "ticker": t,
            "total_items": len(t_combined),
            "quarters_count": len(periods_found),
            "earliest_period": min(periods_found) if periods_found else "None",
            "latest_period": max(periods_found) if periods_found else "None"
        })

    if all_records:
        fund_quarterly = pd.concat(all_records, ignore_index=True)
    else:
        fund_quarterly = pd.DataFrame(columns=["ticker", "fiscal_period", "item_code", "value", "available_date"])

    out_file = PROCESSED_DATA_DIR / "fund_quarterly.parquet"
    fund_quarterly.to_parquet(out_file, index=False)

    cov_df = pd.DataFrame(coverage_rows)
    cov_path = TABLES_DIR / "fundamentals_coverage.csv"
    cov_df.to_csv(cov_path, index=False, encoding="utf-8-sig")

    logger.info(f"Đã lưu BCTC chuẩn hóa: {out_file} ({len(fund_quarterly)} dòng)")
    logger.info(f"Đã lưu báo cáo độ phủ BCTC: {cov_path}")
    return fund_quarterly


if __name__ == "__main__":
    build_and_save_fundamentals()
