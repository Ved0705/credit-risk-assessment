"""Application configuration, read from environment variables / backend/.env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore", protected_namespaces=())

    app_name: str = "Explainable Credit Risk Assessment API"
    data_path: Path = PROJECT_DIR / "data" / "german_credit_raw.csv"
    model_dir: Path = BACKEND_DIR / "models"

    # Comma separated list of allowed frontend origins
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Stability analysis defaults
    stability_samples: int = 20
    stability_noise: float = 0.05

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
