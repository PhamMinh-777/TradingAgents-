"""
Module thu thập, chuẩn hóa công bố thông tin (Corporate Disclosures / CBTT).
Nguồn: Cổng công bố thông tin HOSE và chuyên mục Quan hệ cổ đông (IR) của 7 ngân hàng.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import pandas as pd
from bs4 import BeautifulSoup

from src.data.news import (
    BaseNewsSource,
    clean_text,
    generate_article_id,
    parse_vn_datetime,
    clean_and_deduplicate_news,
)
from src.utils.logger import setup_logger
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = setup_logger("disclosures")


class HOSEDisclosureSource(BaseNewsSource):
    """Bộ bóc tách công bố thông tin từ Cổng thông tin Sở GDCK TP.HCM (HOSE)."""

    def __init__(self):
        super().__init__("hose_cbtt", rate_limit_delay=2.0)

    def parse_html(self, html_content: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, "html.parser")

        title_tag = soup.find(class_=re.compile(r"(title|headline|subject)", re.I))
        title = clean_text(title_tag.get_text()) if title_tag else ""

        date_tag = soup.find(class_=re.compile(r"(date|publish|time)", re.I))
        date_raw = date_tag.get_text().strip() if date_tag else ""
        pub_dt = parse_vn_datetime(date_raw)

        body_div = soup.find(class_=re.compile(r"(content|detail|article)", re.I))
        body = clean_text(body_div.get_text()) if body_div else ""

        return {
            "article_id": generate_article_id(url),
            "url": url,
            "source": self.name,
            "title": title,
            "lead": "",
            "body": body,
            "published_at": pub_dt,
            "crawled_at": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh"),
        }


def save_disclosures(df_disclosures: pd.DataFrame) -> Path:
    """Lưu trữ bảng công bố thông tin đã chuẩn hóa vào data/processed/disclosures.parquet."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "disclosures.parquet"

    if df_disclosures is None or df_disclosures.empty:
        cols = ["article_id", "ticker_candidates", "title", "lead", "body", "source", "url", "published_at", "crawled_at"]
        df_empty = pd.DataFrame(columns=cols)
        df_empty.to_parquet(out_path, index=False)
        return out_path

    df_disclosures.to_parquet(out_path, index=False)
    logger.info(f"Đã lưu bảng CBTT: {out_path} ({len(df_disclosures)} bản ghi)")
    return out_path
