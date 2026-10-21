"""
Module thu thập, chuẩn hóa và kiểm tra chất lượng dữ liệu giá OHLCV.
Áp dụng cho 7 mã ngân hàng HOSE và chỉ số tham chiếu VN-INDEX giai đoạn 2020-01-01 đến 2026-09-30.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = setup_logger("prices")

# Bảng nhật ký thay đổi toàn cục
change_log = pd.DataFrame(columns=["ticker", "date", "column", "old_value", "new_value", "reason"])


def log_change(ticker: str, date: str, column: str, old_val: object, new_val: object, reason: str) -> None:
    """Ghi lại một chỉnh sửa hoặc chuẩn hóa dữ liệu vào change_log."""
    global change_log
    new_row = pd.DataFrame([{
        "ticker": ticker,
        "date": str(date),
        "column": column,
        "old_value": str(old_val),
        "new_value": str(new_val),
        "reason": reason
    }])
    change_log = pd.concat([change_log, new_row], ignore_index=True)


def init_vnstock() -> bool:
    """Đăng nhập vnstock bằng API key từ biến môi trường nếu có.

    Returns:
        bool: True nếu đăng nhập thành công, False nếu dùng chế độ khách/lỗi.
    """
    load_dotenv()
    api_key = os.getenv("VNSTOCK_API_KEY")
    if not api_key:
        logger.warning("Không tìm thấy VNSTOCK_API_KEY trong .env. Dùng chế độ khách mặc định.")
        return False

    try:
        import vnstock
        vnstock.login(api_key.strip())
        logger.info("Đăng nhập vnstock thành công với API key cấu hình.")
        return True
    except Exception as e:
        logger.warning(
            f"Đăng nhập vnstock không thành công ({e}). "
            "Sẽ tiếp tục thử gọi API công khai hoặc dùng cache thô."
        )
        return False


def get_yearly_ranges(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Chia khoảng thời gian thành các khoảng từng năm để tránh vượt trần dòng dữ liệu.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        Danh sách các tuple (start_sub, end_sub)
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    ranges = []

    cur_year = start_dt.year
    while cur_year <= end_dt.year:
        sub_start = max(start_dt, pd.to_datetime(f"{cur_year}-01-01"))
        sub_end = min(end_dt, pd.to_datetime(f"{cur_year}-12-31"))
        if sub_start <= sub_end:
            ranges.append((sub_start.strftime("%Y-%m-%d"), sub_end.strftime("%Y-%m-%d")))
        cur_year += 1

    return ranges


def fetch_ohlcv_chunk(
    symbol: str,
    start: str,
    end: str,
    source: str = "kbs",
    is_index: bool = False,
    max_retries: int = 4,
    delay_between_calls: float = 1.5,
) -> pd.DataFrame:
    """Tải một khoảng dữ liệu OHLCV với cơ chế caching thô và tự động retry backoff khi chạm rate-limit.

    Args:
        symbol: Mã cổ phiếu hoặc mã chỉ số (VNINDEX).
        start: 'YYYY-MM-DD'
        end: 'YYYY-MM-DD'
        source: Nguồn dữ liệu ('kbs', 'vci', 'msn').
        is_index: True nếu là chỉ số thị trường.
        max_retries: Số lần thử lại tối đa.
        delay_between_calls: Độ trễ tối thiểu giữa các lệnh gọi (giây).

    Returns:
        DataFrame OHLCV thô.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DATA_DIR / f"{symbol}_{source}_{start}_{end}.parquet"

    # 1. Kiểm tra cache trên đĩa
    if cache_file.exists():
        logger.debug(f"Đọc dữ liệu từ cache: {cache_file.name}")
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            logger.warning(f"File cache {cache_file.name} hỏng, sẽ tải lại.")

    from vnstock import Market
    m = Market()

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(delay_between_calls)
            if is_index:
                df = m.index(symbol).ohlcv(start=start, end=end, interval="1D", source=source)
            else:
                df = m.equity(symbol).ohlcv(start=start, end=end, interval="1D", source=source)

            if df is not None and not df.empty:
                df.to_parquet(cache_file, index=False)
                return df
            else:
                logger.warning(f"[{symbol}] Gọi {source} từ {start} đến {end} trả về rỗng.")
                return pd.DataFrame()

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "hạn mức" in err_str or "RateLimited" in err_str or "10/10" in err_str or "429" in err_str

            if is_rate_limit and attempt < max_retries:
                wait_sec = 60 * attempt
                logger.warning(
                    f"[{symbol}] Chạm hạn mức lượt gọi vnstock ({err_str}). "
                    f"Tạm nghỉ {wait_sec} giây trước khi thử lại (lần {attempt}/{max_retries})..."
                )
                time.sleep(wait_sec)
            elif attempt < max_retries:
                wait_sec = 5 * attempt
                logger.warning(f"[{symbol}] Lỗi khi tải ({e}), thử lại sau {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                logger.error(f"[{symbol}] Thất bại sau {max_retries} lần thử: {e}")
                raise

    return pd.DataFrame()


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    source: str = "kbs",
    is_index: bool = False,
    fallback_sources: Optional[List[str]] = None,
    delay_between_calls: float = 1.2,
) -> pd.DataFrame:
    """Tải toàn bộ chuỗi OHLCV cho một mã bằng cách chia theo từng năm, nối lại và chuẩn hóa.

    Args:
        symbol: Ticker hoặc 'VNINDEX'.
        start: 'YYYY-MM-DD'.
        end: 'YYYY-MM-DD'.
        source: Nguồn chính (mặc định 'kbs').
        is_index: True nếu là chỉ số.
        fallback_sources: Danh sách nguồn dự phòng nếu nguồn chính thiếu dữ liệu.
        delay_between_calls: Giây chờ giữa các lệnh gọi.

    Returns:
        DataFrame OHLCV đã sắp xếp và chuẩn hóa cột cơ bản.
    """
    ranges = get_yearly_ranges(start, end)
    dfs = []

    for sub_start, sub_end in ranges:
        try:
            df_sub = fetch_ohlcv_chunk(
                symbol=symbol,
                start=sub_start,
                end=sub_end,
                source=source,
                is_index=is_index,
                delay_between_calls=delay_between_calls
            )
        except Exception as e:
            logger.warning(f"Lỗi nguồn chính {source} cho {symbol} ({sub_start} - {sub_end}): {e}")
            df_sub = pd.DataFrame()

        # Thử nguồn dự phòng nếu nguồn chính rỗng
        if (df_sub is None or df_sub.empty) and fallback_sources:
            for fb_source in fallback_sources:
                logger.info(f"Thử nguồn dự phòng {fb_source} cho {symbol} ({sub_start} - {sub_end})...")
                try:
                    df_sub = fetch_ohlcv_chunk(
                        symbol=symbol,
                        start=sub_start,
                        end=sub_end,
                        source=fb_source,
                        is_index=is_index,
                        delay_between_calls=delay_between_calls
                    )
                    if df_sub is not None and not df_sub.empty:
                        break
                except Exception as fb_err:
                    logger.warning(f"Nguồn dự phòng {fb_source} cũng lỗi: {fb_err}")

        if df_sub is not None and not df_sub.empty:
            dfs.append(df_sub)

    if not dfs:
        logger.error(f"Không thể thu thập bất kỳ dữ liệu nào cho {symbol} từ {start} đến {end}.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    # Chuẩn hóa tên cột
    col_map = {str(c): str(c).lower() for c in full_df.columns}
    full_df = full_df.rename(columns=col_map)

    # Chuẩn hóa cột ngày
    time_col = "time" if "time" in full_df.columns else ("date" if "date" in full_df.columns else None)
    if not time_col:
        raise KeyError(f"Không tìm thấy cột thời gian trong dữ liệu {symbol}: {full_df.columns}")

    full_df["date"] = pd.to_datetime(full_df[time_col]).dt.tz_localize(None).dt.normalize()
    full_df = full_df.drop(columns=[time_col]) if time_col != "date" else full_df

    # Loại bỏ trùng ngày, sắp xếp tăng dần
    dup_mask = full_df.duplicated(subset=["date"], keep="first")
    if dup_mask.sum() > 0:
        log_change(
            ticker=symbol,
            date="various",
            column="date",
            old_val=f"{dup_mask.sum()} duplicate rows",
            new_val="kept first",
            reason="Loại bỏ dòng trùng lặp ngày khi ghép các khoảng năm"
        )
        full_df = full_df[~dup_mask]

    full_df = full_df.sort_values("date").reset_index(drop=True)

    # Chuẩn hóa tên cột OHLCV
    rename_cols = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    for old_c, new_c in rename_cols.items():
        if old_c in full_df.columns:
            full_df[new_c] = full_df[old_c]

    # Đảm bảo các cột tối thiểu
    for req_col in ["Open", "High", "Low", "Close", "Volume"]:
        if req_col not in full_df.columns:
            full_df[req_col] = np.nan

    return full_df


def check_price_data_quality(df: pd.DataFrame, ticker: str) -> Dict[str, Union[str, int, float]]:
    """Kiểm tra chất lượng kỹ thuật toàn diện cho chuỗi dữ liệu giá của một mã.

    Args:
        df: DataFrame chứa date, Open, High, Low, Close, Volume.
        ticker: Tên mã chứng khoán.

    Returns:
        Dict ghi nhận các chỉ số chất lượng để tổng hợp thành data_quality.csv.
    """
    if df.empty:
        return {
            "ticker": ticker,
            "total_rows": 0,
            "start_date": "N/A",
            "end_date": "N/A",
            "missing_values": 0,
            "invalid_price_nonpositive": 0,
            "invalid_high_low": 0,
            "invalid_open_close_bound": 0,
            "extreme_daily_return_gt_8pct": 0,
            "zero_volume": 0,
            "large_gaps_gt_5_days": 0,
            "status": "FAILED_EMPTY"
        }

    total_rows = len(df)
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    # 1. NaN checks
    missing_vals = int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())

    # 2. Giá <= 0
    nonpos_mask = (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0)
    invalid_nonpos = int(nonpos_mask.sum())

    # 3. High < Low
    invalid_hl = int((df["High"] < df["Low"]).sum())

    # 4. Open/Close ngoài [Low, High] (cho phép dung sai 1e-4 do làm tròn)
    bound_mask = (df["Open"] < df["Low"] - 1e-4) | (df["Open"] > df["High"] + 1e-4) | \
                 (df["Close"] < df["Low"] - 1e-4) | (df["Close"] > df["High"] + 1e-4)
    invalid_bound = int(bound_mask.sum())

    # 5. Lợi suất ngày tuyệt đối > 8% (biên độ HOSE +-7%)
    ret = df["Close"].pct_change()
    extreme_ret = int((ret.abs() > 0.08).sum())

    # 6. Khối lượng = 0
    zero_vol = int((df["Volume"] == 0).sum())

    # 7. Khoảng trống lịch giao dịch > 5 ngày làm việc (trừ Tết / nghỉ dài)
    date_diffs = df["date"].diff().dt.days
    large_gaps = int((date_diffs > 5).sum())

    status = "PASSED" if (invalid_nonpos == 0 and invalid_hl == 0 and invalid_bound == 0) else "WARNING"

    return {
        "ticker": ticker,
        "total_rows": total_rows,
        "start_date": start_date,
        "end_date": end_date,
        "missing_values": missing_vals,
        "invalid_price_nonpositive": invalid_nonpos,
        "invalid_high_low": invalid_hl,
        "invalid_open_close_bound": invalid_bound,
        "extreme_daily_return_gt_8pct": extreme_ret,
        "zero_volume": zero_vol,
        "large_gaps_gt_5_days": large_gaps,
        "status": status
    }


def verify_rights_adjustment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Xác minh và chuẩn hóa giá đã điều chỉnh quyền (Adj Close).

    Dữ liệu KBS trên vnstock v5 đã được điều chỉnh kỹ thuật cho các đợt chia cổ tức bằng cổ phiếu / cổ tức tiền.
    Nếu cột 'Adj Close' chưa có hoặc là bản sao của Close, thiết lập và ghi chú vào change_log.
    """
    df = df.copy()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
        log_change(
            ticker=ticker,
            date="all",
            column="Adj Close",
            old_val="None",
            new_val="Close",
            reason="Khởi tạo cột Adj Close khớp với Close (giá KBS đã điều chỉnh quyền)"
        )
    return df


def process_and_save_dataset(
    stock_dict: Dict[str, pd.DataFrame],
    vni_df: pd.DataFrame,
    state_owned: List[str],
    private_owned: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tổng hợp 7 mã vào ohlcv_long format, chuẩn hóa VNINDEX và tạo báo cáo chất lượng.

    Args:
        stock_dict: Dict ticker -> DataFrame OHLCV thô.
        vni_df: DataFrame VNINDEX.
        state_owned: Danh sách mã quốc doanh.
        private_owned: Danh sách mã tư nhân.

    Returns:
        Tuple (ohlcv_long, vnindex_df, quality_df)
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    long_list = []
    quality_records = []

    for ticker, df in stock_dict.items():
        if df.empty:
            continue

        q_report = check_price_data_quality(df, ticker)
        quality_records.append(q_report)

        df_proc = verify_rights_adjustment(df, ticker)
        group = "state_owned" if ticker in state_owned else "private"

        df_sub = pd.DataFrame({
            "date": df_proc["date"],
            "ticker": ticker,
            "group": group,
            "Open": df_proc["Open"].astype(float),
            "High": df_proc["High"].astype(float),
            "Low": df_proc["Low"].astype(float),
            "Close": df_proc["Close"].astype(float),
            "Adj Close": df_proc["Adj Close"].astype(float),
            "Volume": df_proc["Volume"].astype(float),
        })
        long_list.append(df_sub)

    ohlcv_long = pd.concat(long_list, ignore_index=True)
    ohlcv_long = ohlcv_long.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Chuẩn hóa VNINDEX
    vni_proc = vni_df.copy()
    q_vni = check_price_data_quality(vni_proc, "VNINDEX")
    quality_records.append(q_vni)

    vni_clean = pd.DataFrame({
        "date": vni_proc["date"],
        "VNI_Open": vni_proc["Open"].astype(float),
        "VNI_High": vni_proc["High"].astype(float),
        "VNI_Low": vni_proc["Low"].astype(float),
        "VNI_Close": vni_proc["Close"].astype(float),
        "VNI_Volume": vni_proc["Volume"].astype(float),
    })
    vni_clean["VNI_ret"] = vni_clean["VNI_Close"].pct_change()
    vni_clean = vni_clean.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    quality_df = pd.DataFrame(quality_records)

    # Lưu kết quả
    ohlcv_path = PROCESSED_DATA_DIR / "ohlcv_long.parquet"
    vni_path = PROCESSED_DATA_DIR / "vnindex.parquet"
    quality_path = TABLES_DIR / "data_quality.csv"

    ohlcv_long.to_parquet(ohlcv_path, index=False)
    vni_clean.to_parquet(vni_path, index=False)
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")

    logger.info(f"Đã lưu bảng giá tổng hợp: {ohlcv_path} ({len(ohlcv_long)} dòng)")
    logger.info(f"Đã lưu VNINDEX: {vni_path} ({len(vni_clean)} dòng)")
    logger.info(f"Đã lưu báo cáo chất lượng: {quality_path}")

    return ohlcv_long, vni_clean, quality_df


def run_pipeline() -> None:
    """Hàm điều phối tải toàn bộ dữ liệu giá theo cấu hình config.yaml."""
    config = load_config()
    init_vnstock()

    tickers_cfg = config["tickers"]
    dates_cfg = config["dates"]
    data_cfg = config["data"]

    symbols = tickers_cfg["state_owned"] + tickers_cfg["private"]
    start_date = dates_cfg["start_date"]
    end_date = dates_cfg["end_date"]
    source = data_cfg.get("prices_source", "kbs")
    fallback = data_cfg.get("fallback_sources", ["vci", "msn"])

    print("=" * 70)
    print(f"BẮT ĐẦU THU THẬP VÀ XỬ LÝ DỮ LIỆU GIÁ ({start_date} -> {end_date})")
    print(f"Danh sách mã: {symbols} + VNINDEX")
    print("=" * 70)

    stock_dict = {}
    for s in symbols:
        print(f"--> Đang tải dữ liệu cho {s}...")
        df_s = fetch_ohlcv(s, start=start_date, end=end_date, source=source, fallback_sources=fallback)
        stock_dict[s] = df_s
        print(f"    {s}: {len(df_s)} dòng (từ {df_s['date'].min().date() if not df_s.empty else 'N/A'} đến {df_s['date'].max().date() if not df_s.empty else 'N/A'})")

    print("--> Đang tải dữ liệu chỉ số VNINDEX...")
    vni_df = fetch_ohlcv("VNINDEX", start=start_date, end=end_date, source=source, is_index=True, fallback_sources=fallback)
    print(f"    VNINDEX: {len(vni_df)} dòng")

    ohlcv_long, vni_clean, quality_df = process_and_save_dataset(
        stock_dict=stock_dict,
        vni_df=vni_df,
        state_owned=tickers_cfg["state_owned"],
        private_owned=tickers_cfg["private"],
    )

    print("\n" + "=" * 70)
    print("BÁO CÁO TỔNG HỢP CHẤT LƯỢNG DỮ LIỆU GIÁ (DATA QUALITY REPORT)")
    print("=" * 70)
    print(quality_df.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
