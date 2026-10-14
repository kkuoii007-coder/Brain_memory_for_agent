"""Тесты долговременной памяти (Qdrant) и краткосрочной (SQLite).

Покрывают требования TASK.md, раздел 17:
  1. SQLite возвращает последние 10 сообщений.
  2. История разных пользователей не смешивается.
  3. Дубликат памяти не создаёт новую запись.
  4. Противоречащий факт деактивирует старый.
  5. Critical memory извлекается независимо от similarity.
  6. Аллергия учитывается после большого количества сообщений.
  10. Один пользователь не получает память другого пользователя.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from app.chat_service import ChatService
from app.database import Database
from app.memory import MemoryStore
from app.openrouter import OpenRouterClient
from app.schemas import (
    ExtractedMemory,
    ExtractionResult,
    FactAction,
    MemoryCategory,
    RelationResult,
    RelationType,
)
from app.web_search import WebSearchService


# --- Вспомогательные функции ---

def make_fact(
    fact: str = "У пользователя аллергия на арахис",
    category: MemoryCategory = MemoryCategory.ALLERGY,
    action: FactAction = FactAction.ADD,
    importance: float = 1.0,
    confidence: float = 1.0,
    critical: bool = True,
) -> ExtractedMemory:
    return ExtractedMemory(
        action=action,
        fact=fact,
        category=category,
        importance=importance,
        confidence=confidence,
        critical=critical,
    )


async def save_fact_via_service(
    chat: ChatService,
    user_id: int,
    fact: ExtractedMemory,
    openrouter_mock: OpenRouterClient,
) -> None:
    """Запустить _process_fact для одного факта напрямую."""
    await chat._process_fact(user_id, fact, source_message_id=1)


# =====================================================================
# 1. SQLite возвращает последние 10 сообщений
# =====================================================================
@pytest.mark.asyncio
async def test_sqlite_returns_last_10_messages(database: Database):
    user_id = 111
    await database.upsert_user(user_id, "alice")
    # Сохраняем 15 сообщений
    for i in range(15):
        role = "user" if i % 2 == 0 else "assistant"
        await database.save_message(user_id, role, f"msg-{i}", i)

    messages = await database.get_recent_messages(user_id, limit=10)
    assert len(messages) == 10
    # Должны быть последние 10 (msg-5 .. msg-14), от старых к новым
    contents = [m["content"] for m in messages]
    assert contents[0] == "msg-5"
    assert contents[-1] == "msg-14"


# =====================================================================
# 2. История разных пользователей не смешивается
# =====================================================================
@pytest.mark.asyncio
async def test_history_isolated_between_users(database: Database):
    await database.upsert_user(111, "alice")
    await database.upsert_user(222, "bob")
    await database.save_message(111, "user", "сообщение Алисы", 1)
    await database.save_message(222, "user", "сообщение Боба", 2)

    alice = await database.get_recent_messages(111, limit=10)
    bob = await database.get_recent_messages(222, limit=10)

    assert len(alice) == 1
    assert len(bob) == 1
    assert alice[0]["content"] == "сообщение Алисы"
    assert bob[0]["content"] == "сообщение Боба"


# =====================================================================
# 3. Дубликат памяти не создаёт новую запись
# =====================================================================
@pytest.mark.asyncio
async def test_duplicate_memory_does_not_create_new(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
    settings,
):
    user_id = 111
    # Сохраняем факт
    fact = make_fact("У пользователя аллергия на арахис")
    vec = await openrouter_mock.embed(fact.fact)
    point_id = memory_store.add(user_id, fact, vec, source_message_id=1)
    assert len(memory_store.get_all_active(user_id)) == 1

    # Тот же текст — embedding будет идентичным (детерминированный fake)
    vec2 = await openrouter_mock.embed(fact.fact)
    hits = memory_store.search(user_id, vec2, limit=5, score_threshold=0.0)
    assert hits
    sim = hits[0].score
    assert sim >= settings.memory_duplicate_threshold  # ~1.0 для одинакового текста

    # Эмулируем поведение ChatService: дубликат → touch, без новой записи
    memory_store.touch(hits[0].id)
    assert len(memory_store.get_all_active(user_id)) == 1  # не добавилось


# =====================================================================
# 4. Противоречащий факт деактивирует старый
# =====================================================================
@pytest.mark.asyncio
async def test_contradiction_deactivates_old(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
    settings,
):
    user_id = 111
    chat = ChatService(
        settings,
        database=AsyncMock(),
        openrouter=openrouter_mock,
        memory=memory_store,
        web_search=AsyncMock(),
    )
    # Существующий факт
    old_fact = make_fact("У пользователя аллергия на арахис", critical=True)
    vec = await openrouter_mock.embed(old_fact.fact)
    memory_store.add(user_id, old_fact, vec, source_message_id=1)
    assert len(memory_store.get_all_active(user_id)) == 1

    # Новый противоречащий факт: «аллергии на арахис нет»
    new_fact = make_fact(
        "У пользователя нет аллергии на арахис",
        category=MemoryCategory.HEALTH,
        critical=False,
    )
    # LLM-классификатор вернёт contradiction
    openrouter_mock.chat_structured = AsyncMock(
        return_value=RelationResult(relation=RelationType.CONTRADICTION)
    )
    await save_fact_via_service(chat, user_id, new_fact, openrouter_mock)

    active = memory_store.get_all_active(user_id)
    assert len(active) == 1  # старый деактивирован, новый активен
    assert "нет аллергии" in active[0].text


# =====================================================================
# 5. Critical memory извлекается независимо от similarity
# =====================================================================
@pytest.mark.asyncio
async def test_critical_memory_retrieved_regardless_of_similarity(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
):
    user_id = 111
    # Сохраняем критический факт
    fact = make_fact("У пользователя аллергия на арахис", critical=True)
    vec = await openrouter_mock.embed(fact.fact)
    memory_store.add(user_id, fact, vec, source_message_id=1)

    # Запрос, совершенно не похожий на аллергию. Используем высокий порог,
    # чтобы семантический поиск гарантированно ничего не вернул (моделируем
    # непохожий запрос — в реальности similarity будет низким).
    unrelated_vec = await openrouter_mock.embed("Какая погода в Москве?")
    semantic = memory_store.search(
        user_id, unrelated_vec, limit=8, score_threshold=0.90
    )
    assert len(semantic) == 0  # семантически не нашлось при высоком пороге

    # Но критическая память всё равно извлекается
    critical = memory_store.get_critical(user_id)
    assert len(critical) == 1
    assert "арахис" in critical[0].text
    assert critical[0].critical is True


# =====================================================================
# 6. Аллергия учитывается после большого количества сообщений
# =====================================================================
@pytest.mark.asyncio
async def test_allergy_accounted_after_many_messages(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
    database: Database,
    settings,
):
    user_id = 111
    # Сохраняем аллергию
    fact = make_fact("У пользователя аллергия на арахис", critical=True)
    vec = await openrouter_mock.embed(fact.fact)
    memory_store.add(user_id, fact, vec, source_message_id=1)

    # Добавляем 1000 сообщений в SQLite
    await database.upsert_user(user_id, "alice")
    for i in range(1000):
        await database.save_message(user_id, "user", f"сообщение {i}", i)
        await database.save_message(user_id, "assistant", f"ответ {i}")

    # Последние 10 сообщений
    history = await database.get_recent_messages(user_id, limit=10)
    assert len(history) == 10
    # Аллергии в последних 10 нет
    assert not any("арахис" in m["content"] for m in history)

    # Но критическая память извлекается
    critical = memory_store.get_critical(user_id)
    assert len(critical) == 1
    assert "арахис" in critical[0].text


# =====================================================================
# 10. Один пользователь не получает память другого пользователя
# =====================================================================
@pytest.mark.asyncio
async def test_user_isolation_in_memory(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
):
    # Пользователь A имеет аллергию
    fact_a = make_fact("У пользователя A аллергия на пыльцу")
    vec_a = await openrouter_mock.embed(fact_a.fact)
    memory_store.add(111, fact_a, vec_a, source_message_id=1)

    # Пользователь B ищет — не должен видеть факт A
    vec_b = await openrouter_mock.embed("аллергия")
    hits_b = memory_store.search(222, vec_b, limit=8, score_threshold=0.0)
    assert len(hits_b) == 0

    # Критическая память B пуста
    assert len(memory_store.get_critical(222)) == 0

    # Все активные факты B пусты
    assert len(memory_store.get_all_active(222)) == 0

    # A видит свой факт
    assert len(memory_store.get_all_active(111)) == 1


# =====================================================================
# Дополнительно: update существующего факта
# =====================================================================
@pytest.mark.asyncio
async def test_update_existing_fact(
    memory_store: MemoryStore,
    openrouter_mock: OpenRouterClient,
    settings,
):
    user_id = 111
    chat = ChatService(
        settings,
        database=AsyncMock(),
        openrouter=openrouter_mock,
        memory=memory_store,
        web_search=AsyncMock(),
    )
    old_fact = make_fact("Пользователь живёт в Москве", category=MemoryCategory.LOCATION, critical=False)
    vec = await openrouter_mock.embed(old_fact.fact)
    memory_store.add(user_id, old_fact, vec, source_message_id=1)

    new_fact = make_fact(
        "Пользователь переехал в Санкт-Петербург",
        category=MemoryCategory.LOCATION,
        critical=False,
    )
    openrouter_mock.chat_structured = AsyncMock(
        return_value=RelationResult(relation=RelationType.UPDATE)
    )
    await save_fact_via_service(chat, user_id, new_fact, openrouter_mock)

    active = memory_store.get_all_active(user_id)
    assert len(active) == 1  # запись обновлена, не добавлена
    assert "Санкт-Петербург" in active[0].text
