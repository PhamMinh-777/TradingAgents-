"""
Module thu thập, làm sạch và chuẩn hóa tin tức tài chính tiếng Việt (News Processing).
Nguồn: CafeF, Vietstock, VnEconomy, Tin nhanh chứng khoán.
Hỗ trợ MinHash gần trùng, lọc Look-ahead qua published_at múi giờ Việt Nam (+07:00).
"""

import hashlib
import os
import re
import sys
import unicodedata
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.utils.assert_no_future import assert_no_future
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = setup_logger("news")


def clean_text(text: str) -> str:
    """Làm sạch văn bản: chuẩn hóa Unicode NFC, loại bỏ ký tự rác và khoảng trắng thừa."""
    if not text:
        return ""
    # Chuẩn hóa Unicode sang dạng dựng sẵn NFC
    text = unicodedata.normalize("NFC", text)
    # Loại bỏ thẻ HTML còn sót
    text = re.sub(r"<[^>]+>", " ", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_article_id(url: str) -> str:
    """Tạo ID duy nhất cho bài viết dựa trên băm SHA-256 của URL."""
    clean_url = url.split("?")[0].strip().lower()
    return hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:16]


def parse_vn_datetime(date_str: str) -> Optional[pd.Timestamp]:
    """Phân tích chuỗi ngày giờ báo chí tiếng Việt về ISO múi giờ Việt Nam (UTC+7).

    Hỗ trợ các định dạng phổ biến:
    - DD-MM-YYYY - HH:MM AM/PM
    - DD/MM/YYYY HH:MM
    - YYYY-MM-DDTHH:MM:SS
    """
    if not date_str:
        return None

    clean_str = re.sub(r"[,\-_]", " ", date_str).strip()
    # Tìm mẫu ngày/tháng/năm
    patterns = [
        r"(\d{1,2})/(\d{1,2})/(\d{4})\s*(\d{1,2}):(\d{2})",
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})\s+(\d{1,2}):(\d{2})",
        r"(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}):(\d{2})",
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
    ]

    for pat in patterns:
        m = re.search(pat, date_str)
        if m:
            try:
                # Dùng pd.to_datetime với dayfirst=True
                dt = pd.to_datetime(m.group(0), dayfirst=True)
                return dt.tz_localize("Asia/Ho_Chi_Minh") if dt.tz is None else dt.tz_convert("Asia/Ho_Chi_Minh")
            except Exception:
                pass

    try:
        dt = pd.to_datetime(date_str, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return dt.tz_localize("Asia/Ho_Chi_Minh") if dt.tz is None else dt.tz_convert("Asia/Ho_Chi_Minh")
    except Exception:
        pass

    return None


def get_shingles(text: str, k: int = 3) -> Set[str]:
    """Tạo tập word k-grams (shingles) cho thuật toán MinHash / Jaccard."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Tính độ tương đồng Jaccard giữa 2 tập shingle."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class BaseNewsSource(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các nguồn thu thập tin tức và công bố thông tin."""

    def __init__(self, name: str, rate_limit_delay: float = 1.5):
        self.name = name
        self.rate_limit_delay = rate_limit_delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Student Research Project)"
        }

    @abstractmethod
    def parse_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Trích xuất tiêu đề, sa-pô (lead), thân bài và ngày đăng từ HTML bài viết."""
        pass


class CafeFSource(BaseNewsSource):
    """Bộ bóc tách tin tức từ CafeF."""

    def __init__(self):
        super().__init__("cafef", rate_limit_delay=1.5)

    def parse_html(self, html_content: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, "html.parser")

        # Tiêu đề
        title_tag = soup.find("h1", class_=re.compile(r"title", re.I))
        title = clean_text(title_tag.get_text()) if title_tag else ""

        # Ngày đăng
        date_tag = soup.find(class_=re.compile(r"(pdate|date|time)", re.I))
        date_raw = date_tag.get_text().strip() if date_tag else ""
        pub_dt = parse_vn_datetime(date_raw)

        # Lead / Sapo
        sapo_tag = soup.find(class_=re.compile(r"(sapo|summary)", re.I))
        lead = clean_text(sapo_tag.get_text()) if sapo_tag else ""

        # Nội dung
        body_div = soup.find(class_=re.compile(r"(contentdetail|detail-content|body)", re.I))
        body = ""
        if body_div:
            # Xóa quảng cáo bên trong
            for ad in body_div.find_all(class_=re.compile(r"(banner|ad|relate)", re.I)):
                ad.decompose()
            body = clean_text(body_div.get_text())

        return {
            "article_id": generate_article_id(url),
            "url": url,
            "source": self.name,
            "title": title,
            "lead": lead,
            "body": body,
            "published_at": pub_dt,
            "crawled_at": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh"),
        }


class VietstockSource(BaseNewsSource):
    """Bộ bóc tách tin tức từ Vietstock."""

    def __init__(self):
        super().__init__("vietstock", rate_limit_delay=1.5)

    def parse_html(self, html_content: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, "html.parser")

        title_tag = soup.find("h1", class_=re.compile(r"(article-title|title)", re.I))
        title = clean_text(title_tag.get_text()) if title_tag else ""

        date_tag = soup.find(class_=re.compile(r"(date|time)", re.I))
        date_raw = date_tag.get_text().strip() if date_tag else ""
        pub_dt = parse_vn_datetime(date_raw)

        sapo_tag = soup.find(class_=re.compile(r"(article-summary|summary|lead)", re.I))
        lead = clean_text(sapo_tag.get_text()) if sapo_tag else ""

        body_div = soup.find(class_=re.compile(r"(article-content|content)", re.I))
        body = ""
        if body_div:
            for ad in body_div.find_all(class_=re.compile(r"(ad|banner|recommend)", re.I)):
                ad.decompose()
            body = clean_text(body_div.get_text())

        return {
            "article_id": generate_article_id(url),
            "url": url,
            "source": self.name,
            "title": title,
            "lead": lead,
            "body": body,
            "published_at": pub_dt,
            "crawled_at": pd.Timestamp.now(tz="Asia/Ho_Chi_Minh"),
        }


def assign_ticker_candidates(article: Dict[str, Any], tickers_map: Dict[str, List[str]]) -> List[str]:
    """Xác định danh sách mã chứng khoán liên quan đến bài viết.

    Args:
        article: Dict bài viết (title, lead, body).
        tickers_map: Dict mã -> danh sách từ khóa (ví dụ: 'VCB': ['VCB', 'Vietcombank', 'Ngoại thương']).

    Returns:
        Danh sách các mã chứng khoán được nhắc đến.
    """
    text_to_search = f"{article.get('title', '')} {article.get('lead', '')} {article.get('body', '')[:500]}".lower()
    matched = []
    for ticker, kws in tickers_map.items():
        for kw in kws:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_to_search):
                matched.append(ticker)
                break
    return matched


def clean_and_deduplicate_news(
    raw_articles: List[Dict[str, Any]],
    tickers_map: Optional[Dict[str, List[str]]] = None,
    similarity_threshold: float = 0.85
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Làm sạch, lọc bài viết không hợp lệ và khử trùng lặp (Exact + MinHash).

    Args:
        raw_articles: Danh sách bài viết thô trích xuất.
        tickers_map: Bản đồ mã và từ khóa để lọc liên quan.
        similarity_threshold: Ngưỡng Jaccard xem là trùng gần.

    Returns:
        Tuple (df_valid, df_rejected)
    """
    if not raw_articles:
        empty_cols = ["article_id", "ticker_candidates", "title", "lead", "body", "source", "url", "published_at", "crawled_at"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols + ["rejection_reason"])

    if tickers_map is None:
        tickers_map = {
            "VCB": ["VCB", "Vietcombank", "Ngoại thương"],
            "BID": ["BID", "BIDV", "Đầu tư và Phát triển"],
            "CTG": ["CTG", "VietinBank", "Công thương"],
            "MBB": ["MBB", "MB Bank", "Ngân hàng Quân đội", "Quân đội"],
            "ACB": ["ACB", "Á Châu"],
            "TCB": ["TCB", "Techcombank", "Kỹ thương"],
            "VPB": ["VPB", "VPBank", "Việt Nam Thịnh Vượng"],
        }

    valid_list = []
    rejected_list = []

    seen_ids = set()
    seen_shingles: List[Tuple[str, Set[str]]] = []  # (article_id, shingles)

    for item in raw_articles:
        art_id = item.get("article_id") or generate_article_id(item.get("url", ""))
        title = clean_text(item.get("title", ""))
        body = clean_text(item.get("body", ""))
        lead = clean_text(item.get("lead", ""))
        pub_at = item.get("published_at")

        # 1. Kiểm tra ngày đăng hợp lệ (Bắt buộc)
        if pub_at is None or pd.isna(pub_at):
            rej = dict(item)
            rej["rejection_reason"] = "MISSING_OR_INVALID_PUBLISHED_AT"
            rejected_list.append(rej)
            continue

        # 2. Kiểm tra độ dài nội dung tối thiểu
        if len(title) < 5 or (len(body) < 20 and len(lead) < 20):
            rej = dict(item)
            rej["rejection_reason"] = "BODY_TOO_SHORT"
            rejected_list.append(rej)
            continue

        # 3. Trùng lặp tuyệt đối (URL / article_id)
        if art_id in seen_ids:
            rej = dict(item)
            rej["rejection_reason"] = "EXACT_DUPLICATE_ID"
            rejected_list.append(rej)
            continue

        # 4. Kiểm tra liên quan đến danh mục ngân hàng
        item_copy = dict(item)
        item_copy["title"] = title
        item_copy["lead"] = lead
        item_copy["body"] = body
        item_copy["article_id"] = art_id
        matched_tickers = assign_ticker_candidates(item_copy, tickers_map)

        if not matched_tickers:
            rej = dict(item)
            rej["rejection_reason"] = "NOT_RELEVANT_TO_BANK_PORTFOLIO"
            rejected_list.append(rej)
            continue

        item_copy["ticker_candidates"] = matched_tickers

        # 5. Khử trùng lặp gần (MinHash / Jaccard similarity trên tiêu đề + lead)
        text_signature = get_shingles(f"{title} {lead}", k=3)
        is_near_dup = False
        for prev_id, prev_shingles in seen_shingles:
            sim = jaccard_similarity(text_signature, prev_shingles)
            if sim >= similarity_threshold:
                is_near_dup = True
                rej = dict(item_copy)
                rej["rejection_reason"] = f"NEAR_DUPLICATE_OF_{prev_id}_SIM_{sim:.2f}"
                rejected_list.append(rej)
                break

        if is_near_dup:
            continue

        # Hợp lệ
        seen_ids.add(art_id)
        seen_shingles.append((art_id, text_signature))
        valid_list.append(item_copy)

    df_valid = pd.DataFrame(valid_list)
    df_rejected = pd.DataFrame(rejected_list)

    total_in = len(raw_articles)
    total_valid = len(df_valid)
    rej_rate = ((total_in - total_valid) / total_in * 100) if total_in > 0 else 0.0

    logger.info(
        f"Làm sạch tin tức hoàn tất: {total_valid}/{total_in} bài hợp lệ. "
        f"Tỷ lệ bị loại: {rej_rate:.1f}% ({len(df_rejected)} bài)."
    )

    return df_valid, df_rejected


def generate_coverage_report(
    news_df: pd.DataFrame,
    tickers: List[str],
    output_csv: Path,
    output_png: Optional[Path] = None,
) -> pd.DataFrame:
    """Tạo ma trận số lượng bài viết theo mã và theo tháng (Heatmap Coverage Report).

    Args:
        news_df: DataFrame tin tức hợp lệ (chứa ticker_candidates, published_at).
        tickers: Danh sách mã cần theo dõi.
        output_csv: Đường dẫn lưu CSV.
        output_png: Đường dẫn lưu heatmap PNG.

    Returns:
        DataFrame ma trận độ phủ (Ticker x Tháng).
    """
    if news_df.empty:
        df_empty = pd.DataFrame(index=tickers)
        df_empty.to_csv(output_csv)
        return df_empty

    # Bung mã từ list ticker_candidates
    rows = []
    for _, r in news_df.iterrows():
        dt = pd.to_datetime(r["published_at"])
        month_str = dt.strftime("%Y-%m")
        cand = r["ticker_candidates"]
        if isinstance(cand, list):
            for t in cand:
                if t in tickers:
                    rows.append({"ticker": t, "month": month_str, "article_id": r["article_id"]})
        elif isinstance(cand, str) and cand in tickers:
            rows.append({"ticker": cand, "month": month_str, "article_id": r["article_id"]})

    if not rows:
        matrix = pd.DataFrame(0, index=tickers, columns=["2020-01"])
    else:
        df_flat = pd.DataFrame(rows)
        matrix = df_flat.pivot_table(
            index="ticker",
            columns="month",
            values="article_id",
            aggfunc="count",
            fill_value=0
        )
        matrix = matrix.reindex(tickers).fillna(0).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, encoding="utf-8-sig")
    logger.info(f"Đã lưu báo cáo độ phủ tin tức: {output_csv}")

    # Vẽ heatmap nếu có đường dẫn ảnh
    if output_png:
        try:
            import matplotlib.pyplot as plt
            output_png.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(16, 6))
            plt.imshow(matrix.values, cmap="YlGnBu", aspect="auto")
            plt.colorbar(label="Số lượng bài viết")
            plt.yticks(range(len(matrix.index)), matrix.index)
            plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
            plt.title("Độ phủ Tin tức theo Mã Ngân hàng và Tháng (News Coverage Heatmap)")
            plt.tight_layout()
            plt.savefig(output_png, dpi=200)
            plt.close()
            logger.info(f"Đã vẽ biểu đồ độ phủ: {output_png}")
        except Exception as e:
            logger.warning(f"Không thể vẽ biểu đồ heatmap tin tức ({e})")

    return matrix


def run_sample_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chạy thử nghiệm pipeline tin tức cho 1 tháng (ví dụ 01/2024) với các fixture hoặc tin mẫu."""
    from src.data.disclosures import HOSEDisclosureSource, save_disclosures

    config = load_config()
    tickers = config["tickers"]["state_owned"] + config["tickers"]["private"]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc mẫu từ fixture hoặc tạo mẫu thử nghiệm đại diện cho 7 mã trong tháng 01/2024
    sample_raw = [
        {
            "url": "https://cafef.vn/vietcombank-lai-41000-ty-2023.chn",
            "title": "Vietcombank báo lãi kỷ lục hơn 41.000 tỷ đồng trong năm 2023",
            "lead": "Ngân hàng TMCP Ngoại thương Việt Nam (VCB) tiếp tục giữ vị trí quán quân lợi nhuận.",
            "body": "Vietcombank ghi nhận tổng tài sản tăng trưởng vượt bậc, nợ xấu được kiểm soát chặt chẽ.",
            "published_at": pd.Timestamp("2024-01-15 08:30:00", tz="Asia/Ho_Chi_Minh"),
            "source": "cafef"
        },
        {
            "url": "https://vietstock.vn/mbb-tang-truong-tin-dung-manh-me.htm",
            "title": "MBB: Ngân hàng Quân đội đạt tăng trưởng tín dụng ấn tượng trong quý 4",
            "lead": "MB duy trì đà tăng trưởng cao nhờ số hóa mạnh mẽ.",
            "body": "Năm 2023, Ngân hàng Quân đội ghi nhận lợi nhuận trước thuế tăng trưởng trên 15%.",
            "published_at": pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Ho_Chi_Minh"),
            "source": "vietstock"
        },
        {
            "url": "https://cafef.vn/bidv-ha-lai-suat-ho-tro-doanh-nghiep.chn",
            "title": "BIDV tiên phong hạ lãi suất cho vay đầu năm 2024",
            "lead": "BID cam kết đồng hành cùng doanh nghiệp thúc đẩy sản xuất kinh doanh.",
            "body": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam triển khai gói tín dụng quy mô lớn.",
            "published_at": pd.Timestamp("2024-01-20 14:00:00", tz="Asia/Ho_Chi_Minh"),
            "source": "cafef"
        },
        {
            "url": "https://vietstock.vn/techcombank-casa-hoi-phuc.htm",
            "title": "Techcombank: Tỷ lệ CASA phục hồi mạnh mẽ về mức 40%",
            "lead": "TCB tiếp tục dẫn đầu ngành về hiệu quả chi phí vốn và tỷ lệ tiền gửi không kỳ hạn.",
            "body": "Ngân hàng Kỹ thương Việt Nam ghi nhận sự trở lại mạnh mẽ của dòng tiền gửi thanh toán.",
            "published_at": pd.Timestamp("2024-01-22 09:15:00", tz="Asia/Ho_Chi_Minh"),
            "source": "vietstock"
        },
        {
            "url": "https://cafef.vn/acb-chat-luong-tai-san-vung-chac.chn",
            "title": "ACB duy trì chất lượng tài sản lành mạnh nhất hệ thống",
            "lead": "Ngân hàng Á Châu không đầu tư trái phiếu doanh nghiệp rủi ro.",
            "body": "Tỷ lệ nợ xấu của ACB tiếp tục thuộc nhóm thấp nhất trong các ngân hàng tư nhân.",
            "published_at": pd.Timestamp("2024-01-25 15:30:00", tz="Asia/Ho_Chi_Minh"),
            "source": "cafef"
        },
        {
            "url": "https://cafef.vn/ctg-vietinbank-but-pha-loi-nhuan.chn",
            "title": "VietinBank: Lợi nhuận trước thuế năm 2023 vượt kế hoạch đề ra",
            "lead": "CTG đẩy mạnh thu hồi nợ xử lý rủi ro và tăng thu ngoài lãi.",
            "body": "Ngân hàng Công thương Việt Nam đạt kết quả kinh doanh vượt mong đợi.",
            "published_at": pd.Timestamp("2024-01-28 11:20:00", tz="Asia/Ho_Chi_Minh"),
            "source": "cafef"
        },
        {
            "url": "https://vietstock.vn/vpb-hoan-tat-tang-von-voi-smbc.htm",
            "title": "VPBank: Nền tảng vốn hàng đầu sau thương vụ với SMBC",
            "lead": "VPB có hệ số an toàn vốn (CAR) cao nhất hệ thống.",
            "body": "Ngân hàng Việt Nam Thịnh Vượng sẵn sàng cho các cơ hội bứt phá trong chu kỳ mới.",
            "published_at": pd.Timestamp("2024-01-30 16:45:00", tz="Asia/Ho_Chi_Minh"),
            "source": "vietstock"
        },
        # Bài trùng lặp
        {
            "url": "https://cafef.vn/vietcombank-lai-41000-ty-2023.chn?utm_source=share",
            "title": "Vietcombank báo lãi kỷ lục hơn 41.000 tỷ đồng trong năm 2023",
            "lead": "Ngân hàng TMCP Ngoại thương Việt Nam (VCB) tiếp tục giữ vị trí quán quân lợi nhuận.",
            "body": "Vietcombank ghi nhận tổng tài sản tăng trưởng vượt bậc.",
            "published_at": pd.Timestamp("2024-01-15 08:30:00", tz="Asia/Ho_Chi_Minh"),
            "source": "cafef"
        },
        # Bài thiếu published_at
        {
            "url": "https://cafef.vn/tin-khong-ngay.chn",
            "title": "Ngân hàng MB mở rộng mạng lưới chi nhánh",
            "lead": "MB phát triển điểm giao dịch mới.",
            "body": "Chi tiết mạng lưới...",
            "published_at": None,
            "source": "cafef"
        }
    ]

    valid_df, rej_df = clean_and_deduplicate_news(sample_raw)

    out_news = PROCESSED_DATA_DIR / "news.parquet"
    out_rej = PROCESSED_DATA_DIR / "rejected_news.parquet"
    valid_df.to_parquet(out_news, index=False)
    rej_df.to_parquet(out_rej, index=False)

    # Báo cáo độ phủ
    cov_csv = TABLES_DIR / "news_coverage.csv"
    cov_png = FIGURES_DIR / "news_coverage_heatmap.png"
    cov_matrix = generate_coverage_report(valid_df, tickers, cov_csv, cov_png)

    # Lưu disclosures mẫu
    save_disclosures(valid_df.copy())

    print("\n" + "=" * 70)
    print("CHẠY THỬ NGHIỆM PIPELINE TIN TỨC THÀNH CÔNG")
    print(f"Số bài hợp lệ lưu tại {out_news}: {len(valid_df)}")
    print(f"Số bài bị loại lưu tại {out_rej}: {len(rej_df)}")
    print(f"Báo cáo độ phủ lưu tại: {cov_csv}")
    print("=" * 70)

    return valid_df, rej_df, cov_matrix


if __name__ == "__main__":
    run_sample_pipeline()
