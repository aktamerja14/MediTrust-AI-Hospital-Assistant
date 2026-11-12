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

    # ==========================================
    # LLM CONFIGURATION
    # ==========================================
    # CHANGE HERE: Select your LLM provider.
    # Supported: gemini, groq, openrouter
    LLM_PROVIDER: str = "gemini"

    # CHANGE HERE: Set the model supported by the selected provider.
    # Gemini Free Tier: gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro
    # Groq Free Tier: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
    # OpenRouter Free Tier: meta-llama/llama-3.1-8b-instruct:free, mistralai/mistral-7b-instruct:free
    LLM_MODEL: str = "gemini-1.5-flash"

    # CHANGE HERE: Add the API key for the selected provider.
    # Gemini: Get from https://aistudio.google.com/apikey
    # Groq: Get from https://console.groq.com/keys
    # OpenRouter: Get from https://openrouter.ai/keys
    LLM_API_KEY: str = ""

    # CHANGE HERE: Configure base URL only when required by the provider (Groq/OpenRouter).
    # Groq: https://api.groq.com/openai/v1
    # OpenRouter: https://openrouter.ai/api/v1
    LLM_BASE_URL: str = ""

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60

    # ==========================================
    # EMBEDDING CONFIGURATION
    # ==========================================
    # CHANGE HERE: Select the embedding provider independently.
    # Supported: local, openai, gemini
    EMBEDDING_PROVIDER: str = "local"

    # CHANGE HERE: Select the embedding model.
    # local: sentence-transformers/all-MiniLM-L6-v2 (384 dims), sentence-transformers/all-mpnet-base-v2 (768 dims)
    # openai: text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims)
    # gemini: text-embedding-004 (768 dims)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # CHANGE HERE: Configure embedding API key only if required (OpenAI/Gemini).
    EMBEDDING_API_KEY: str = ""

    # ==========================================
    # VECTOR DATABASE CONFIGURATION
    # ==========================================
    # CHANGE HERE: Select vector store. Supported: faiss
    VECTOR_STORE: str = "faiss"

    # CHANGE HERE: Path for FAISS index persistence.
    VECTOR_STORE_PATH: str = "data/vector_store"

    # CHANGE HERE: RAG chunk size and overlap.
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200

    # CHANGE HERE: Number of chunks to retrieve for context.
    RAG_TOP_K: int = 4

    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()