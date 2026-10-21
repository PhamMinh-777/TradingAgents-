"""
Module thu thập, chuẩn hóa và xử lý dữ liệu cho hệ thống vn-llm-agents:
- prices: Giá OHLCV 7 mã ngân hàng HOSE và chỉ số VN-Index
- fundamentals: Báo cáo tài chính theo quý và các chỉ số ngân hàng
- events: Sự kiện doanh nghiệp và cổ tức
- news: Tin tức tài chính tiếng Việt, khử trùng lặp MinHash
- disclosures: Công bố thông tin từ HOSE và IR ngân hàng
"""

from src.data.prices import (
    fetch_ohlcv,
    check_price_data_quality,
    verify_rights_adjustment,
    process_and_save_dataset,
)
from src.data.fundamentals import (
    calc_available_date,
    standardize_statements_to_long,
    compute_banking_ratios,
    get_fundamentals,
    build_and_save_fundamentals,
)
from src.data.events import (
    classify_event_type,
    standardize_corporate_events,
    get_corporate_events,
    build_and_save_corporate_events,
)
from src.data.news import (
    BaseNewsSource,
    clean_text,
    generate_article_id,
    clean_and_deduplicate_news,
    generate_coverage_report,
)
from src.data.disclosures import HOSEDisclosureSource, save_disclosures

__all__ = [
    "fetch_ohlcv",
    "check_price_data_quality",
    "verify_rights_adjustment",
    "process_and_save_dataset",
    "calc_available_date",
    "standardize_statements_to_long",
    "compute_banking_ratios",
    "get_fundamentals",
    "build_and_save_fundamentals",
    "classify_event_type",
    "standardize_corporate_events",
    "get_corporate_events",
    "build_and_save_corporate_events",
    "BaseNewsSource",
    "clean_text",
    "generate_article_id",
    "clean_and_deduplicate_news",
    "generate_coverage_report",
    "HOSEDisclosureSource",
    "save_disclosures",
]
