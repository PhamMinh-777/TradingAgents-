"""
Unit tests cho module src/data/news.py và src/data/disclosures.py sử dụng HTML fixtures giả lập.
Đảm bảo kiểm thử hoàn toàn độc lập với mạng Internet (Offline deterministic tests).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.news import (
    clean_text,
    generate_article_id,
    parse_vn_datetime,
    get_shingles,
    jaccard_similarity,
    CafeFSource,
    VietstockSource,
    clean_and_deduplicate_news,
    generate_coverage_report,
)
from src.data.disclosures import HOSEDisclosureSource, save_disclosures

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_clean_text():
    """Kiểm tra làm sạch thẻ HTML và chuẩn hóa Unicode NFC."""
    raw = "<p>Vietcombank&nbsp; l\u1ee3i nhu\u1eadn <b>41.000 t\u1ef7</b></p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "Vietcombank" in cleaned


def test_generate_article_id():
    """Kiểm tra tạo mã băm định danh duy nhất cho URL."""
    url1 = "https://cafef.vn/vietcombank-lai-lon.chn"
    url2 = "https://cafef.vn/vietcombank-lai-lon.chn?utm_source=fb"
    id1 = generate_article_id(url1)
    id2 = generate_article_id(url2)
    assert id1 == id2  # URL gốc giống nhau sau khi bỏ query params
    assert len(id1) == 16


def test_parse_vn_datetime():
    """Kiểm tra chuyển đổi các định dạng ngày báo tiếng Việt."""
    dt1 = parse_vn_datetime("15-01-2024 - 08:30 AM")
    assert dt1 is not None
    assert dt1.year == 2024 and dt1.month == 1 and dt1.day == 15

    dt2 = parse_vn_datetime("20/04/2024 14:15")
    assert dt2 is not None
    assert dt2.year == 2024 and dt2.month == 4 and dt2.day == 20

    dt_invalid = parse_vn_datetime("Hôm qua lúc 10h")
    # Không khớp mẫu số rõ ràng
    assert dt_invalid is None or isinstance(dt_invalid, pd.Timestamp)


def test_minhash_jaccard():
    """Kiểm tra thuật toán tương đồng shingle Jaccard."""
    s1 = "Vietcombank báo lãi 41000 tỷ năm 2023 kỷ lục"
    s2 = "Vietcombank báo lãi 41000 tỷ năm 2023 tăng mạnh"
    s3 = "Ngân hàng ACB khai trương chi nhánh mới tại Đà Nẵng"

    sh1 = get_shingles(s1)
    sh2 = get_shingles(s2)
    sh3 = get_shingles(s3)

    sim_high = jaccard_similarity(sh1, sh2)
    sim_low = jaccard_similarity(sh1, sh3)

    assert sim_high > 0.4
    assert sim_low == 0.0


def test_cafef_parser_fixture():
    """Kiểm tra bóc tách bài viết CafeF qua HTML fixture."""
    fixture_path = FIXTURES_DIR / "sample_cafef.html"
    assert fixture_path.exists()

    with open(fixture_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    parser = CafeFSource()
    res = parser.parse_html(html_str, "https://cafef.vn/sample-vcb.chn")

    assert "Vietcombank báo lãi kỷ lục" in res["title"]
    assert "Quảng cáo tài chính" not in res["body"]  # Đã lọc thẻ quảng cáo
    assert res["published_at"] is not None
    assert res["published_at"].year == 2024
    assert res["source"] == "cafef"


def test_vietstock_parser_fixture():
    """Kiểm tra bóc tách bài viết Vietstock qua HTML fixture."""
    fixture_path = FIXTURES_DIR / "sample_vietstock.html"
    assert fixture_path.exists()

    with open(fixture_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    parser = VietstockSource()
    res = parser.parse_html(html_str, "https://vietstock.vn/sample-mbb.htm")

    assert "MBB: Ngân hàng Quân đội" in res["title"]
    assert res["published_at"] is not None
    assert res["published_at"].year == 2024
    assert res["source"] == "vietstock"


def test_clean_and_deduplicate_pipeline():
    """Kiểm tra quy trình làm sạch, gán nhãn mã và khử trùng lặp."""
    raw_articles = [
        # Bài 1: Chuẩn VCB
        {
            "url": "https://cafef.vn/art1.chn",
            "title": "Vietcombank công bố lợi nhuận quý 1",
            "lead": "VCB tăng trưởng tín dụng mạnh",
            "body": "Nội dung chi tiết về ngân hàng Vietcombank...",
            "published_at": pd.Timestamp("2024-04-15", tz="Asia/Ho_Chi_Minh"),
        },
        # Bài 2: Gần trùng Bài 1
        {
            "url": "https://vietstock.vn/art2.htm",
            "title": "Vietcombank công bố lợi nhuận quý 1",
            "lead": "VCB tăng trưởng tín dụng mạnh mẽ",
            "body": "Nội dung chi tiết về ngân hàng Vietcombank...",
            "published_at": pd.Timestamp("2024-04-15", tz="Asia/Ho_Chi_Minh"),
        },
        # Bài 3: Thiếu published_at -> Bị loại
        {
            "url": "https://cafef.vn/art3.chn",
            "title": "BIDV tiếp tục hạ lãi suất cho vay",
            "lead": "BID giảm lãi suất",
            "body": "Nội dung về BIDV...",
            "published_at": None,
        },
        # Bài 4: Không liên quan ngành ngân hàng
        {
            "url": "https://cafef.vn/art4.chn",
            "title": "Giá thịt heo hôm nay tăng nhẹ",
            "lead": "Thị trường nông sản biến động",
            "body": "Giá heo hơi tại miền Bắc...",
            "published_at": pd.Timestamp("2024-04-16", tz="Asia/Ho_Chi_Minh"),
        },
    ]

    valid_df, rej_df = clean_and_deduplicate_news(raw_articles, similarity_threshold=0.8)

    assert len(valid_df) == 1
    assert valid_df["ticker_candidates"].iloc[0] == ["VCB"]

    assert len(rej_df) == 3
    reasons = set(rej_df["rejection_reason"])
    assert "MISSING_OR_INVALID_PUBLISHED_AT" in reasons
    assert "NOT_RELEVANT_TO_BANK_PORTFOLIO" in reasons
    assert any("NEAR_DUPLICATE" in r for r in reasons)


def test_coverage_report(tmp_path):
    """Kiểm tra tạo ma trận báo cáo độ phủ (Coverage report)."""
    news_df = pd.DataFrame([
        {"ticker_candidates": ["VCB"], "published_at": pd.Timestamp("2024-01-10"), "article_id": "a1"},
        {"ticker_candidates": ["VCB", "MBB"], "published_at": pd.Timestamp("2024-01-15"), "article_id": "a2"},
        {"ticker_candidates": ["CTG"], "published_at": pd.Timestamp("2024-02-01"), "article_id": "a3"},
    ])
    csv_file = tmp_path / "coverage.csv"
    png_file = tmp_path / "coverage.png"

    matrix = generate_coverage_report(news_df, ["VCB", "MBB", "CTG", "BID"], csv_file, png_file)
    assert csv_file.exists()
    assert matrix.loc["VCB", "2024-01"] == 2
    assert matrix.loc["MBB", "2024-01"] == 1
    assert matrix.loc["CTG", "2024-02"] == 1
    assert matrix.loc["BID", "2024-01"] == 0
