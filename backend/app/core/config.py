from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine backend base directory and locate backend/.env
# Path(__file__) is backend/app/core/config.py -> .parent.parent.parent is backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """
    Application Settings powered by Pydantic Settings.
    Dynamically loads configuration from backend/.env, falling back to local .env.
    No credentials or API keys are hardcoded.
    """
    PROJECT_NAME: str = "MediTrust API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # MongoDB Atlas Connection Configuration (Configurable via backend/.env)
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "meditrust_db"

    # CORS Origins (Frontends allowed to access API)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Argon2 Password Hashing Configuration
    ARGON2_TIME_COST: int = 3
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 4
    ARGON2_HASH_LEN: int = 32
    ARGON2_SALT_LEN: int = 16

    # Document Storage Configuration
    DOCUMENT_STORAGE_PATH: str = "data/documents"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
