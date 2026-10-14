"""Тесты интернет-поиска (DDGS) и правил необходимости поиска.

Покрывают требования TASK.md, раздел 17:
  7. Запрос о погоде запускает интернет-поиск.
  8. Обычный вопрос о Python не запускает поиск.
  9. Ошибка DDGS не останавливает бота.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.web_search import WebSearchService, SearchDecision, format_search_results


# =====================================================================
# 7. Запрос о погоде запускает интернет-поиск
# =====================================================================
@pytest.mark.asyncio
async def test_weather_triggers_search(
    web_search_service: WebSearchService,
    openrouter_mock,
):
    # Правила должны сработать на слово "погода" без вызова LLM
    openrouter_mock.chat_structured = AsyncMock(side_effect=AssertionError("LLM не должен вызываться"))

    decision = await web_search_service.should_search(
        "Какая сейчас погода в Москве?", user_search_enabled=True
    )
    assert decision.needs_search is True
    assert decision.reason == "rules"
    assert "погода" in decision.query.lower() or "москв" in decision.query.lower()


# =====================================================================
# 8. Обычный вопрос о Python не запускает поиск
# =====================================================================
@pytest.mark.asyncio
async def test_python_question_no_search(
    web_search_service: WebSearchService,
    openrouter_mock,
):
    # Правила не сработают → вызывается LLM. Мокаем LLM: поиск не нужен.
    openrouter_mock.chat_structured = AsyncMock(return_value=None)
    from app.schemas import WebSearchDecision

    openrouter_mock.chat_structured = AsyncMock(
        return_value=WebSearchDecision(
            needs_web_search=False, search_query="", reason="теоретический вопрос"
        )
    )

    decision = await web_search_service.should_search(
        "Что такое переменная в Python?", user_search_enabled=True
    )
    assert decision.needs_search is False


# =====================================================================
# 7b. Дополнительные правила: курс валют, новости
# =====================================================================
@pytest.mark.asyncio
async def test_currency_triggers_search(
    web_search_service: WebSearchService,
    openrouter_mock,
):
    openrouter_mock.chat_structured = AsyncMock(side_effect=AssertionError("LLM не должен вызываться"))

    decision = await web_search_service.should_search(
        "Какой курс доллара сегодня?", user_search_enabled=True
    )
    assert decision.needs_search is True
    assert decision.reason == "rules"


# =====================================================================
# 9. Ошибка DDGS не останавливает бота (возвращает пустой список)
# =====================================================================
@pytest.mark.asyncio
async def test_ddgs_error_returns_empty_list(
    web_search_service: WebSearchService,
    monkeypatch,
):
    # Мокаем _sync_ddgs_text, чтобы он бросал исключение
    def _boom(*args, **kwargs):
        raise ConnectionError("Нет интернета")

    monkeypatch.setattr("app.web_search._sync_ddgs_text", _boom)

    results = await web_search_service.search("погода Москва")
    assert results == []  # бот не упал, поиск просто недоступен


# =====================================================================
# 9b. Пустой результат DDGS
# =====================================================================
@pytest.mark.asyncio
async def test_ddgs_empty_result(
    web_search_service: WebSearchService,
    monkeypatch,
):
    monkeypatch.setattr("app.web_search._sync_ddgs_text", lambda *a, **k: [])
    results = await web_search_service.search("что угодно")
    assert results == []


# =====================================================================
# Очистка результатов: дубли URL, пустые заголовки
# =====================================================================
@pytest.mark.asyncio
async def test_clean_results_dedup_and_filter(
    web_search_service: WebSearchService,
    monkeypatch,
):
    raw = [
        {"title": "Заголовок 1", "href": "https://example.com/1", "body": "описание 1"},
        {"title": "Заголовок 2", "href": "https://example.com/1", "body": "дубликат URL"},
        {"title": "", "href": "https://example.com/2", "body": "пустой заголовок"},
        {"title": "Без URL", "href": "", "body": "нет url"},
        {"title": "Заголовок 3", "href": "https://example.com/3", "body": "описание 3"},
    ]
    monkeypatch.setattr("app.web_search._sync_ddgs_text", lambda *a, **k: raw)
    results = await web_search_service.search("запрос")
    assert len(results) == 2
    assert results[0].url == "https://example.com/1"
    assert results[1].url == "https://example.com/3"


# =====================================================================
# Форматирование результатов
# =====================================================================
def test_format_search_results():
    from app.web_search import SearchResult

    results = [
        SearchResult(title="Тест", url="https://example.com", description="Сниппет"),
    ]
    text = format_search_results(results)
    assert "Тест" in text
    assert "https://example.com" in text
    assert "Сниппет" in text


def test_format_search_results_empty():
    text = format_search_results([])
    assert "не дал" in text.lower() or "нет" in text.lower()


# =====================================================================
# Глобальный флаг отключения поиска
# =====================================================================
@pytest.mark.asyncio
async def test_global_search_disabled(
    web_search_service: WebSearchService,
    settings,
    openrouter_mock,
):
    # Отключаем глобально
    settings.web_search_enabled = False
    openrouter_mock.chat_structured = AsyncMock(side_effect=AssertionError("LLM не должен вызываться"))

    decision = await web_search_service.should_search(
        "Какая погода в Москве?", user_search_enabled=True
    )
    assert decision.needs_search is False


# =====================================================================
# Per-user флаг отключения поиска
# =====================================================================
@pytest.mark.asyncio
async def test_user_search_disabled(
    web_search_service: WebSearchService,
    openrouter_mock,
):
    openrouter_mock.chat_structured = AsyncMock(side_effect=AssertionError("LLM не должен вызываться"))

    decision = await web_search_service.should_search(
        "Какая погода в Москве?", user_search_enabled=False
    )
    assert decision.needs_search is False
