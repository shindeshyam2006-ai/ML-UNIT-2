"""Environment configuration loader for ML Unit 2 Activity 2."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ames_housing.csv"
CLEAN_DATA_PATH = DATA_DIR / "ames_housing_cleaned.csv"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""

    serp_api_key: str = ""
    gemini_api_key: str = ""
    openweather_api_key: str = ""
    news_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            serp_api_key=os.getenv("SERP_API_KEY", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
            news_api_key=os.getenv("NEWS_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        )

    def api_status(self) -> dict[str, bool]:
        """Return non-sensitive boolean availability flags for external APIs."""
        return {
            "gemini": bool(self.gemini_api_key),
            "openweather": bool(self.openweather_api_key),
            "newsapi": bool(self.news_api_key),
            "serpapi": bool(self.serp_api_key),
        }


def get_settings() -> Settings:
    """Load fresh settings snapshot from environment."""
    return Settings.from_env()
