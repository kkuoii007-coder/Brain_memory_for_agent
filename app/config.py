"""Загрузка и проверка переменных окружения через pydantic-settings.

Все настройки приложения читаются из .env и валидируются Pydantic.
Секреты (токены, ключи) в логи не попадают — pydantic-settings по умолчанию
не логирует значения полей.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Секреты ---
    telegram_bot_token: str = Field(..., min_length=1)
    openrouter_api_key: str = Field(..., min_length=1)
    openrouter_chat_model: str = Field(..., min_length=1)
    openrouter_embedding_model: str = Field(..., min_length=1)

    # --- OpenRouter ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Хранилища ---
    database_path: str = "data/bot.db"
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "telegram_memory"

    # --- Краткосрочная память ---
    recent_messages_limit: int = 10

    # --- Пороги долговременной памяти ---
    memory_min_importance: float = 0.60
    memory_min_confidence: float = 0.75
    memory_duplicate_threshold: float = 0.92
    memory_related_threshold: float = 0.78
    memory_retrieval_threshold: float = 0.35
    memory_retrieval_limit: int = 8

    # --- Интернет-поиск ---
    web_search_enabled: bool = True
    web_search_results_limit: int = 5

    # --- Логирование ---
    log_level: str = "INFO"

    # --- Сетевые таймауты/ретраи ---
    openrouter_timeout: float = 60.0
    openrouter_max_retries: int = 3
    openrouter_retry_base_delay: float = 1.0
    openrouter_embedding_dim: int = 0  # 0 = определить автоматически

    # --- Ограничения Telegram ---
    telegram_max_message_length: int = 4096
    max_input_length: int = 8000

    @property
    def db_file(self) -> Path:
        """Абсолютный/относительный путь к SQLite-файлу с созданием папки."""
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def qdrant_dir(self) -> Path:
        """Путь к папке Qdrant Local Mode с созданием родителя."""
        path = Path(self.qdrant_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_settings() -> Settings:
    """Загрузить и провалидировать настройки из .env."""
    return Settings()


__all__ = ["Settings", "load_settings"]
