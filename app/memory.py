"""Долговременная векторная память на Qdrant Local Mode.

Qdrant работает в локальном режиме (path=...), без отдельного сервера.
Все запросы фильтруются по user_id и active=true — память разных
пользователей никогда не смешивается.

Каждое воспоминание — атомарный факт с payload (см. schemas.py).
Дубликаты определяются по cosine similarity с порогами из настроек.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.schemas import CRITICAL_CATEGORIES, ExtractedMemory, MemoryCategory

logger = logging.getLogger(__name__)


@dataclass
class MemoryHit:
    """Один найденный факт из Qdrant."""

    id: str
    text: str
    category: str
    importance: float
    confidence: float
    critical: bool
    score: float = 0.0  # cosine similarity (только для семантического поиска)


def _now_iso() -> str:
    """Текущее время в ISO-формате (без микросекунд)."""
    return datetime.now().isoformat(timespec="seconds")


def _is_critical(fact: ExtractedMemory) -> bool:
    """Принудительно устанавливает critical=True для опасных категорий."""
    return bool(fact.critical) or fact.category in CRITICAL_CATEGORIES


class MemoryStore:
    """Обёртка над Qdrant Local Mode для долговременной памяти."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: QdrantClient | None = None
        self._vector_size: int | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def connect(self, vector_size: int) -> None:
        """Инициализировать локальный Qdrant и создать коллекцию.

        vector_size — размерность embedding-модели (определяется один раз
        при старте приложения через пробный embed).
        """
        path = str(self._settings.qdrant_dir)
        self._client = QdrantClient(path=path)
        self._vector_size = vector_size
        self._ensure_collection()
        logger.info("Qdrant Local готов: path=%s, vector_size=%s", path, vector_size)

    def close(self) -> None:
        """Корректно закрыть клиент Qdrant."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant закрыт")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("MemoryStore не инициализирован: вызовите connect()")
        return self._client

    def _ensure_collection(self) -> None:
        """Создать коллекцию, если её нет."""
        collection = self._settings.qdrant_collection
        existing = self.client.get_collections()
        names = {c.name for c in existing.collections}
        if collection not in names:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Создана коллекция Qdrant: %s", collection)

    # ------------------------------------------------------------------
    # Фильтры
    # ------------------------------------------------------------------
    def _user_filter(self, user_id: int, *, active_only: bool = True) -> models.Filter:
        """Фильтр по user_id (и опционально active=true)."""
        conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
        if active_only:
            conditions.append(
                models.FieldCondition(
                    key="active",
                    match=models.MatchValue(value=True),
                )
            )
        return models.Filter(must=conditions)

    def _user_critical_filter(self, user_id: int) -> models.Filter:
        """Фильтр: user_id AND active=true AND critical=true."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                ),
                models.FieldCondition(
                    key="active", match=models.MatchValue(value=True)
                ),
                models.FieldCondition(
                    key="critical", match=models.MatchValue(value=True)
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def add(
        self,
        user_id: int,
        fact: ExtractedMemory,
        vector: list[float],
        *,
        source_message_id: int | None = None,
        supersedes_id: str | None = None,
    ) -> str:
        """Добавить новый факт в Qdrant. Возвращает ID созданной точки."""
        point_id = str(uuid.uuid4())
        now = _now_iso()
        payload: dict[str, Any] = {
            "user_id": user_id,
            "text": fact.fact,
            "category": fact.category.value,
            "importance": float(fact.importance),
            "confidence": float(fact.confidence),
            "critical": _is_critical(fact),
            "active": True,
            "created_at": now,
            "updated_at": now,
            "source_message_id": source_message_id,
            "supersedes_id": supersedes_id,
        }
        self.client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            ],
        )
        logger.debug("Добавлен факт user=%s category=%s", user_id, fact.category.value)
        return point_id

    # ------------------------------------------------------------------
    # Семантический поиск
    # ------------------------------------------------------------------
    def search(
        self,
        user_id: int,
        vector: list[float],
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[MemoryHit]:
        """Найти релевантные факты пользователя по вектору.

        Возвращает список MemoryHit, отсортированный по убыванию score.
        Фильтр: user_id == user_id AND active == true.
        """
        if limit is None:
            limit = self._settings.memory_retrieval_limit
        if score_threshold is None:
            score_threshold = self._settings.memory_retrieval_threshold

        results = self.client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=vector,
            query_filter=self._user_filter(user_id, active_only=True),
            limit=limit,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for point in results.points:
            score = float(point.score or 0.0)
            if score < score_threshold:
                continue
            p = point.payload or {}
            hits.append(
                MemoryHit(
                    id=str(point.id),
                    text=str(p.get("text", "")),
                    category=str(p.get("category", "other")),
                    importance=float(p.get("importance", 0.0)),
                    confidence=float(p.get("confidence", 0.0)),
                    critical=bool(p.get("critical", False)),
                    score=score,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Критическая память (без similarity)
    # ------------------------------------------------------------------
    def get_critical(self, user_id: int) -> list[MemoryHit]:
        """Получить все активные критические факты пользователя.

        Извлекается независимо от cosine similarity — например, аллергия
        должна учитываться даже спустя 1000 сообщений.
        """
        results = self.client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=self._user_critical_filter(user_id),
            limit=1000,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for point in results[0]:
            p = point.payload or {}
            hits.append(
                MemoryHit(
                    id=str(point.id),
                    text=str(p.get("text", "")),
                    category=str(p.get("category", "other")),
                    importance=float(p.get("importance", 0.0)),
                    confidence=float(p.get("confidence", 0.0)),
                    critical=True,
                    score=0.0,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Обновление / деактивация
    # ------------------------------------------------------------------
    def touch(self, point_id: str) -> None:
        """Обновить updated_at существующей записи (дубликат/пропуск)."""
        self.client.set_payload(
            collection_name=self._settings.qdrant_collection,
            payload={"updated_at": _now_iso()},
            points=[point_id],
        )

    def update_text(
        self,
        point_id: str,
        fact: ExtractedMemory,
        vector: list[float],
    ) -> None:
        """Обновить текст и метаданные существующего факта."""
        payload = {
            "text": fact.fact,
            "category": fact.category.value,
            "importance": float(fact.importance),
            "confidence": float(fact.confidence),
            "critical": _is_critical(fact),
            "updated_at": _now_iso(),
        }
        self.client.set_payload(
            collection_name=self._settings.qdrant_collection,
            payload=payload,
            points=[point_id],
        )
        # Обновляем вектор
        self.client.update_vectors(
            collection_name=self._settings.qdrant_collection,
            points=[
                models.PointVectors(id=point_id, vector=vector),
            ],
        )

    def deactivate(self, point_id: str) -> None:
        """Пометить факт неактивным (active=false)."""
        self.client.set_payload(
            collection_name=self._settings.qdrant_collection,
            payload={"active": False, "updated_at": _now_iso()},
            points=[point_id],
        )

    # ------------------------------------------------------------------
    # Получение всех активных фактов (для команды /memory)
    # ------------------------------------------------------------------
    def get_all_active(self, user_id: int) -> list[MemoryHit]:
        """Вернуть все активные факты пользователя (для /memory)."""
        results = self.client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=self._user_filter(user_id, active_only=True),
            limit=1000,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for point in results[0]:
            p = point.payload or {}
            hits.append(
                MemoryHit(
                    id=str(point.id),
                    text=str(p.get("text", "")),
                    category=str(p.get("category", "other")),
                    importance=float(p.get("importance", 0.0)),
                    confidence=float(p.get("confidence", 0.0)),
                    critical=bool(p.get("critical", False)),
                    score=0.0,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Полное удаление памяти пользователя
    # ------------------------------------------------------------------
    def delete_all_for_user(self, user_id: int) -> None:
        """Удалить ВСЕ факты пользователя (активные и неактивные)."""
        self.client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        )
                    ]
                )
            ),
        )
        logger.info("Удалена вся память пользователя %s", user_id)


__all__ = ["MemoryStore", "MemoryHit"]
