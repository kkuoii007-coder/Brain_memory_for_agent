"""Pydantic-схемы ответов LLM для валидации JSON из OpenRouter.

Все ответы моделей (извлечение фактов, классификация отношений,
решение об интернет-поиске) проходят через эти схемы. При некорректном
JSON Pydantic выбросит исключение — оно ловится в вызывающем коде,
логируется, и бот продолжает работу.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Категории долговременной памяти ---
class MemoryCategory(str, Enum):
    """Категории воспоминаний. Используются как значения payload Qdrant."""

    ALLERGY = "allergy"
    HEALTH = "health"
    MEDICATION = "medication"
    DIETARY_RESTRICTION = "dietary_restriction"
    IDENTITY = "identity"
    LOCATION = "location"
    PROFESSION = "profession"
    PREFERENCE = "preference"
    GOAL = "goal"
    EDUCATION = "education"
    PROJECT = "project"
    OTHER = "other"


# Категории, для которых critical=true устанавливается принудительно
CRITICAL_CATEGORIES = frozenset(
    {
        MemoryCategory.ALLERGY,
        MemoryCategory.MEDICATION,
        MemoryCategory.DIETARY_RESTRICTION,
    }
)


# --- Извлечение фактов ---
class FactAction(str, Enum):
    """Допустимые действия над памятью при извлечении фактов."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class RelationType(str, Enum):
    """Тип отношения между новым и существующим фактом."""

    DUPLICATE = "duplicate"
    UPDATE = "update"
    CONTRADICTION = "contradiction"
    INDEPENDENT = "independent"


class ExtractedMemory(BaseModel):
    """Один извлечённый факт о пользователе."""

    action: FactAction = Field(..., description="Действие над памятью")
    fact: str = Field(..., min_length=1, description="Атомарный факт о пользователе")
    category: MemoryCategory = Field(
        default=MemoryCategory.OTHER, description="Категория факта"
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Важность факта 0..1"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Уверенность 0..1"
    )
    critical: bool = Field(default=False, description="Критический факт (аллергия и т.п.)")


class ExtractionResult(BaseModel):
    """Результат извлечения фактов: список воспоминаний."""

    memories: list[ExtractedMemory] = Field(default_factory=list)


class RelationResult(BaseModel):
    """Результат классификации отношения между двумя фактами."""

    relation: RelationType = Field(..., description="Тип отношения")
    reason: str = Field(default="", description="Краткое пояснение")


# --- Интернет-поиск ---
class WebSearchDecision(BaseModel):
    """Решение LLM-классификатора о необходимости интернет-поиска."""

    needs_web_search: bool = Field(..., description="Нужен ли поиск")
    search_query: str = Field(default="", description="Поисковый запрос")
    reason: str = Field(default="", description="Причина решения")


__all__ = [
    "MemoryCategory",
    "CRITICAL_CATEGORIES",
    "FactAction",
    "RelationType",
    "ExtractedMemory",
    "ExtractionResult",
    "RelationResult",
    "WebSearchDecision",
]
