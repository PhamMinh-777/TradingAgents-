"""
Kiểm tra và ngăn chặn rò rỉ thông tin tương lai (Look-ahead bias / Point-in-time check).
"""

from typing import Union
import pandas as pd


def assert_no_future(
    df: pd.DataFrame,
    as_of: Union[str, pd.Timestamp],
    date_col: str = "date"
) -> None:
    """Xác nhận không có bản ghi nào trong DataFrame có mốc thời gian sau as_of.

    Args:
        df: DataFrame cần kiểm tra.
        as_of: Mốc thời gian giới hạn điểm nhìn (Point-in-time).
        date_col: Tên cột thời gian (mặc định là 'date').

    Raises:
        ValueError: Nếu phát hiện bản ghi có mốc thời gian lớn hơn as_of.
    """
    if df is None or df.empty:
        return

    as_of_ts = pd.to_datetime(as_of).normalize()

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col]).dt.normalize()
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.normalize()
    else:
        raise KeyError(
            f"Không tìm thấy cột '{date_col}' hoặc DatetimeIndex trong DataFrame để kiểm tra mốc thời gian."
        )

    future_mask = dates > as_of_ts
    future_count = int(future_mask.sum())

    if future_count > 0:
        max_date = dates[future_mask].max()
        violating_sample = df[future_mask].head(3)
        raise ValueError(
            f"Phát hiện rò rỉ thông tin tương lai (Look-ahead violation)! "
            f"Có {future_count} bản ghi vượt mốc as_of={as_of_ts.strftime('%Y-%m-%d')}. "
            f"Thời điểm lớn nhất phát hiện: {max_date.strftime('%Y-%m-%d')}.\n"
            f"Mẫu vi phạm:\n{violating_sample}"
        )
