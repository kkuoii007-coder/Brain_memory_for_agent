"""Общие фикстуры для тестов.

Все тесты работают без реальных запросов к Telegram, OpenRouter и DDGS.
- SQLite и Qdrant используют временные папки (tmp_path), изолированные между тестами.
- OpenRouter мокается через AsyncMock: chat, embed, chat_structured.
- DDGS мокается через monkeypatch.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from app.config import Settings
from app.database import Database
from app.memory import MemoryStore
from app.openrouter import OpenRouterClient
from app.web_search import WebSearchService


# pytest-asyncio в режиме strict: каждый async-тест явно помечается @pytest.mark.asyncio.


def _make_settings(tmp_path: Path) -> Settings:
    """Создать настройки с путями во временной папке."""
    return Settings(
        telegram_bot_token="test:token",
        openrouter_api_key="sk-test",
        openrouter_chat_model="test/chat",
        openrouter_embedding_model="test/embed",
        database_path=str(tmp_path / "bot.db"),
        qdrant_path=str(tmp_path / "qdrant"),
        openrouter_embedding_dim=8,
    )


@pytest_asyncio.fixture
async def settings(tmp_path: Path) -> Settings:
    """Настройки с путями во временной папке."""
    return _make_settings(tmp_path)


@pytest_asyncio.fixture
async def database(settings: Settings) -> Database:
    """Инициализированная SQLite во временной папке."""
    db = Database(settings)
    await db.connect()
    yield db
    await db.close()


def _make_fake_embed(dim: int = 8):
    """Создать синхронную фейковую функцию embed, возвращающую детерминированный вектор."""
    import hashlib

    def _embed(text: str) -> list[float]:
        # Детерминированный вектор на основе текста
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b / 255.0) for b in h[:dim]]
        # Нормализация (для cosine)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    return _embed


@pytest_asyncio.fixture
async def openrouter_mock(settings: Settings) -> OpenRouterClient:
    """Замоканный OpenRouter-клиент с детерминированным embed.

    Тесты могут переопределять методы `chat`, `chat_structured` при необходимости.
    """
    client = OpenRouterClient(settings)
    client._client = AsyncMock()  # чтобы проверка client не падала
    fake_embed = _make_fake_embed(settings.openrouter_embedding_dim)
    client.embed = AsyncMock(side_effect=fake_embed)
    client.chat = AsyncMock(return_value="Ответ ассистента")
    client.chat_structured = AsyncMock(side_effect=_raise_unsupported)
    yield client
    # close не нужен — _client это AsyncMock


def _raise_unsupported(*args, **kwargs):
    raise AssertionError("chat_structured вызван без явного переопределения в тесте")


@pytest_asyncio.fixture
async def memory_store(settings: Settings, openrouter_mock: OpenRouterClient) -> MemoryStore:
    """Инициализированный Qdrant Local во временной папке."""
    store = MemoryStore(settings)
    store.connect(settings.openrouter_embedding_dim)
    yield store
    store.close()


@pytest_asyncio.fixture
async def web_search_service(
    settings: Settings, openrouter_mock: OpenRouterClient
) -> WebSearchService:
    """Сервис поиска с замоканным OpenRouter."""
    return WebSearchService(settings, openrouter_mock)
