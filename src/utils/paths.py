"""
Quản lý các đường dẫn thư mục trong dự án vn-llm-agents.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Trả về đường dẫn thư mục gốc của dự án."""
    return Path(__file__).resolve().parent.parent.parent


ROOT_DIR = get_project_root()
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "prompts"
