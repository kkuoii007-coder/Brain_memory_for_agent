"""Оркестрация обработки одного сообщения пользователя.

ChatService — главный сервис, объединяющий SQLite, Qdrant, OpenRouter
и интернет-поиск в единый пайплайн. Каждый шаг обёрнут в try/except, чтобы
ошибка одного компонента не мешала основному ответу.

Изоляция пользователей:
- Все запросы к SQLite и Qdrant фильтруются по telegram_user_id.
- Последовательная обработка сообщений одного пользователя (per-user lock),
  разные пользователи обрабатываются параллельно.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.database import Database
from app.memory import MemoryHit, MemoryStore
from app.openrouter import OpenRouterClient, OpenRouterError
from app.prompts import (
    EXTRACT_FACTS_PROMPT,
    MEMORY_BLOCK_TEMPLATE,
    RELATION_CLASSIFY_PROMPT,
    SYSTEM_PROMPT,
    WEB_SEARCH_BLOCK_TEMPLATE,
)
from app.schemas import (
    ExtractedMemory,
    ExtractionResult,
    FactAction,
    RelationResult,
    RelationType,
)
from app.web_search import WebSearchService, format_search_results

logger = logging.getLogger(__name__)


@dataclass
class HandleResult:
    """Результат обработки сообщения."""

    reply: str
    used_web_search: bool = False
    used_critical_memory: bool = False
    used_relevant_memory: bool = False
    error: str | None = None


class ChatService:
    """Главный сервис обработки сообщений с гибридной памятью."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        openrouter: OpenRouterClient,
        memory: MemoryStore,
        web_search: WebSearchService,
    ) -> None:
        self._settings = settings
        self._db = database
        self._openrouter = openrouter
        self._memory = memory
        self._web_search = web_search
        # Per-user блокировки для последовательной обработки
        self._user_locks: dict[int, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------------
    # Per-user блокировка
    # ------------------------------------------------------------------
    async def _get_user_lock(self, user_id: int) -> asyncio.Lock:
        """Получить блокировку для конкретного пользователя."""
        async with self._locks_guard:
            lock = self._user_locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._user_locks[user_id] = lock
            return lock

    # ------------------------------------------------------------------
    # Главный пайплайн
    # ------------------------------------------------------------------
    async def handle_message(
        self,
        telegram_user_id: int,
        username: str | None,
        text: str,
        telegram_message_id: int | None = None,
    ) -> HandleResult:
        """Обработать сообщение пользователя и вернуть ответ."""
        lock = await self._get_user_lock(telegram_user_id)
        async with lock:
            try:
                return await self._handle(telegram_user_id, username, text, telegram_message_id)
            except Exception as exc:
                logger.exception("Непредвиденная ошибка обработки сообщения")
                return HandleResult(
                    reply="Произошла ошибка при обработке. Попробуйте позже.",
                    error=str(exc),
                )

    async def _handle(
        self,
        user_id: int,
        username: str | None,
        text: str,
        telegram_message_id: int | None,
    ) -> HandleResult:
        # Шаг 1: upsert user + сохранение текущего сообщения в SQLite
        await self._db.upsert_user(user_id, username)
        await self._db.save_message(user_id, "user", text, telegram_message_id)

        # Шаг 2: последние N сообщений (включая текущее) от старых к новым
        history = await self._db.get_recent_messages(
            user_id, self._settings.recent_messages_limit
        )

        # Шаг 3: критическая память (всегда, независимо от similarity)
        critical: list[MemoryHit] = []
        try:
            critical = self._memory.get_critical(user_id)
        except Exception as exc:
            logger.warning("Не удалось получить критическую память: %s", exc)

        # Шаги 4-5: embedding текущего сообщения + семантический поиск памяти
        relevant: list[MemoryHit] = []
        try:
            query_vector = await self._openrouter.embed(text)
            relevant = self._memory.search(
                user_id,
                query_vector,
                limit=self._settings.memory_retrieval_limit,
                score_threshold=self._settings.memory_retrieval_threshold,
            )
            # Исключить факты, уже вошедшие в критическую память (по ID)
            critical_ids = {h.id for h in critical}
            relevant = [h for h in relevant if h.id not in critical_ids]
        except Exception as exc:
            logger.warning("Семантический поиск памяти недоступен: %s", exc)

        # Шаги 6-7: интернет-поиск
        used_web_search = False
        search_context = ""
        try:
            user_search_on = await self._db.get_web_search_enabled(user_id)
            decision = await self._web_search.should_search(
                text, user_search_enabled=user_search_on
            )
            if decision.needs_search and decision.query:
                results = await self._web_search.search(decision.query)
                used_web_search = bool(results)
                if used_web_search:
                    search_context = WEB_SEARCH_BLOCK_TEMPLATE.format(
                        search_results=format_search_results(results)
                    )
        except Exception as exc:
            logger.warning("Интернет-поиск недоступен: %s", exc)

        # Шаг 8: сборка системного промпта
        system_prompt = self._build_system_prompt(critical, relevant, search_context)

        # Шаг 9: запрос к диалоговой модели
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        try:
            reply = await self._openrouter.chat(messages, temperature=0.7)
        except OpenRouterError as exc:
            logger.error("Ошибка OpenRouter: %s", exc)
            return HandleResult(
                reply="Сервис диалоговой модели временно недоступен. Попробуйте позже.",
                error=str(exc),
                used_web_search=used_web_search,
            )

        # Шаг 10: сохранение ответа ассистента в SQLite
        try:
            await self._db.save_message(user_id, "assistant", reply)
        except Exception as exc:
            logger.warning("Не удалось сохранить ответ в SQLite: %s", exc)

        # Шаг 11: фоновое извлечение и сохранение фактов (без блокировки ответа)
        # Делаем в той же корутине, но перехватываем все ошибки.
        try:
            await self._extract_and_save_memory(user_id, text, telegram_message_id)
        except Exception as exc:
            logger.warning("Не удалось сохранить долговременную память: %s", exc)

        return HandleResult(
            reply=reply,
            used_web_search=used_web_search,
            used_critical_memory=bool(critical),
            used_relevant_memory=bool(relevant),
        )

    # ------------------------------------------------------------------
    # Сборка системного промпта
    # ------------------------------------------------------------------
    def _build_system_prompt(
        self,
        critical: list[MemoryHit],
        relevant: list[MemoryHit],
        search_context: str,
    ) -> str:
        """Собрать системный промпт: базовый + память + (опц.) поиск."""
        if not critical and not relevant:
            memory_block = ""
        else:
            critical_lines = (
                "\n".join(f"- {h.text}" for h in critical) if critical else "—"
            )
            relevant_lines = (
                "\n".join(f"- {h.text}" for h in relevant) if relevant else "—"
            )
            memory_block = MEMORY_BLOCK_TEMPLATE.format(
                critical_block=critical_lines,
                relevant_block=relevant_lines,
            )
        prompt = SYSTEM_PROMPT
        if memory_block:
            prompt += "\n" + memory_block
        if search_context:
            prompt += "\n" + search_context
        return prompt

    # ------------------------------------------------------------------
    # Извлечение и сохранение фактов
    # ------------------------------------------------------------------
    async def _extract_and_save_memory(
        self,
        user_id: int,
        user_message: str,
        source_message_id: int | None,
    ) -> None:
        """Извлечь факты из сообщения и сохранить/обновить память."""
        # Запрос к LLM на извлечение фактов
        try:
            extraction = await self._openrouter.chat_structured(
                EXTRACT_FACTS_PROMPT.format(user_message=user_message),
                "",
                ExtractionResult,
            )
        except OpenRouterError as exc:
            logger.warning("Извлечение фактов недоступно: %s", exc)
            return
        except Exception as exc:
            logger.warning("Неожиданная ошибка извлечения фактов: %s", exc)
            return

        for fact in extraction.memories:
            try:
                await self._process_fact(user_id, fact, source_message_id)
            except Exception as exc:
                logger.warning("Ошибка обработки факта: %s", exc)

    async def _process_fact(
        self,
        user_id: int,
        fact: ExtractedMemory,
        source_message_id: int | None,
    ) -> None:
        """Обработать один извлечённый факт согласно action и порогам."""
        if fact.action == FactAction.NONE:
            return

        # Проверка минимальных порогов важности/уверенности для add/update
        if fact.action in (FactAction.ADD, FactAction.UPDATE):
            if (
                fact.importance < self._settings.memory_min_importance
                or fact.confidence < self._settings.memory_min_confidence
            ):
                logger.debug(
                    "Факт не прошёл порог: importance=%s confidence=%s",
                    fact.importance,
                    fact.confidence,
                )
                return

        # DELETE — best-effort деактивация по тексту
        if fact.action == FactAction.DELETE:
            self._deactivate_by_text(user_id, fact.fact)
            return

        # Получаем embedding факта
        try:
            vector = await self._openrouter.embed(fact.fact)
        except OpenRouterError as exc:
            logger.warning("Не получить embedding факта: %s", exc)
            return

        # Поиск похожих воспоминаний
        hits = self._memory.search(
            user_id,
            vector,
            limit=5,
            score_threshold=0.0,  # ищем все, пороги проверяем вручную
        )
        top = hits[0] if hits else None
        sim = top.score if top else 0.0

        if sim >= self._settings.memory_duplicate_threshold:
            # Дубликат — обновить updated_at существующей записи
            self._memory.touch(top.id)
            logger.debug("Дубликат факта (sim=%.3f), обновлён updated_at", sim)
            return

        if sim >= self._settings.memory_related_threshold:
            # Серая зона — LLM-классификатор отношения
            relation = await self._classify_relation(top.text, fact.fact)
            if relation == RelationType.DUPLICATE:
                self._memory.touch(top.id)
            elif relation == RelationType.UPDATE:
                self._memory.update_text(top.id, fact, vector)
            elif relation == RelationType.CONTRADICTION:
                self._memory.deactivate(top.id)
                self._memory.add(
                    user_id, fact, vector, source_message_id=source_message_id,
                    supersedes_id=top.id,
                )
            else:
                # independent — новый факт
                self._memory.add(
                    user_id, fact, vector, source_message_id=source_message_id
                )
            return

        # similarity < related_threshold — новый факт
        self._memory.add(
            user_id, fact, vector, source_message_id=source_message_id
        )

    async def _classify_relation(
        self, existing_fact: str, new_fact: str
    ) -> RelationType:
        """Определить отношение между фактом и существующим воспоминанием."""
        prompt = RELATION_CLASSIFY_PROMPT.format(
            existing_fact=existing_fact, new_fact=new_fact
        )
        try:
            result = await self._openrouter.chat_structured(
                prompt, "", RelationResult
            )
            return result.relation
        except OpenRouterError as exc:
            logger.warning("Классификатор отношений недоступен: %s", exc)
            # Если классификатор недоступен — считаем независимым (безопасный дефолт)
            return RelationType.INDEPENDENT

    # ------------------------------------------------------------------
    # Best-effort удаление по тексту
    # ------------------------------------------------------------------
    def _deactivate_by_text(self, user_id: int, text: str) -> None:
        """Деактивировать все активные факты пользователя с совпадающим текстом.

        Best-effort: если совпадений нет — ничего не делаем.
        """
        all_active = self._memory.get_all_active(user_id)
        for hit in all_active:
            if hit.text.strip().lower() == text.strip().lower():
                self._memory.deactivate(hit.id)
                logger.debug("Деактивирован факт по DELETE: %s", hit.id)

    # ------------------------------------------------------------------
    # Публичные методы для команд бота
    # ------------------------------------------------------------------
    def list_user_memory(self, user_id: int) -> list[MemoryHit]:
        """Вернуть все активные факты пользователя для /memory."""
        return self._memory.get_all_active(user_id)

    def delete_user_memory(self, user_id: int) -> None:
        """Удалить всю долговременную память пользователя (/forget_all, /reset)."""
        self._memory.delete_all_for_user(user_id)

    async def clear_user_history(self, user_id: int) -> int:
        """Очистить краткосрочную историю SQLite (/clear_history, /reset)."""
        return await self._db.clear_history(user_id)

    async def set_user_search(self, user_id: int, enabled: bool) -> None:
        """Включить/выключить автопоиск для пользователя."""
        await self._db.set_web_search_enabled(user_id, enabled)

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        """Статистика пользователя для /privacy."""
        return await self._db.get_user_stats(user_id)


__all__ = ["ChatService", "HandleResult"]
