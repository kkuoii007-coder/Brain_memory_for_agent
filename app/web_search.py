"""Условный интернет-поиск через DDGS.

Два этапа:
1. should_search — простые правила по словам/намерениям, при неоднозначности
   обращается к LLM-классификатору. Если правила уже дали "да", LLM не вызывается.
2. search — синхронный вызов DDGS обёрнут в asyncio.to_thread, чтобы не
   блокировать асинхронного бота. Результаты очищаются от дублей URL и пустых
   заголовков. Все ошибки ловятся — бот не падает.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from app.config import Settings
from app.openrouter import OpenRouterClient, OpenRouterError
from app.prompts import WEB_SEARCH_CLASSIFY_PROMPT
from app.schemas import WebSearchDecision

logger = logging.getLogger(__name__)


# --- Маркеры актуальности (без учёта регистра) ---
_ACTUALITY_MARKERS = [
    "сейчас",
    "сегодня",
    "вчера",
    "завтра",
    "актуальный",
    "актуально",
    "последний",
    "последняя",
    "свежий",
    "свежая",
    "текущий",
    "текущая",
    "на данный момент",
    "на сейчас",
    "прямо сейчас",
    "в этом году",
    "в этом месяце",
]

# --- Намерения, требующие поиска (по подстроке/regex) ---
_SEARCH_INTENT_PATTERNS = [
    r"\bпогод(а|у|ы)\b",
    r"\bкурс\s+(доллар|евро|рубл|валют|биткоин|криптовалют)",
    r"\bкурс\s+\w+\b.*\b(сегодня|сейчас|актуальн|текущ)\b",
    r"\bцен[аы]\s+(акци|нефти|золота|биткоин|криптовалют)",
    r"\bсколько\s+(стоит|стоят)\b.*\b(акци|нефти|золота|доллар|евро)\b",
    r"\bновост(и|ь)\b",
    r"\bрасписани[ея]\b",
    r"\bпрезидент\b",
    r"\bдействующ\w+\s+(редакци|верси|закон)\b",
    r"\bредакци\w+\s+(закон|кодекс)\b.*\bдейству\w*\b",
    r"\b(?:найди|поищи|искать|ищи|проверь\s+в\s+интернете|поиск\s+в\s+интернете)\b",
    r"\bчто\s+нового\b",
    r"\bитоги\b.*\b(недел|дня|год|месяц)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SEARCH_INTENT_PATTERNS]

# Стоп-фразы — если их явно нет, но есть маркер, поиск не запускаем
_NO_SEARCH_HINTS = [
    "переведи",
    "перевод",
    "исправь",
    "напиши поздравление",
    "объясни",
    "проанализируй",
    "что такое переменная",
]


@dataclass
class SearchDecision:
    """Решение о необходимости поиска."""

    needs_search: bool
    query: str = ""
    reason: str = ""


@dataclass
class SearchResult:
    """Один результат поиска."""

    title: str
    url: str
    description: str


def _matches_rules(text: str) -> bool:
    """Проверить текст по простым правилам: маркеры актуальности или намерения.

    Возвращает True, если правила однозначно указывают на необходимость поиска.
    """
    lower = text.lower()
    # Если есть явный стоп-намек на теоретический запрос — не ищем по правилам
    for hint in _NO_SEARCH_HINTS:
        if hint in lower:
            return False
    # Маркеры актуальности
    for marker in _ACTUALITY_MARKERS:
        if marker in lower:
            return True
    # Намерения (regex)
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _query_from_text(text: str) -> str:
    """Простая очистка текста для поискового запроса."""
    # Убираем служебные фразы-обращения к боту
    cleaned = re.sub(
        r"^(?:бот|ассистент|помощник)[,!.\s]+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    # Обрезаем до разумной длины
    return cleaned.strip()[:200]


class WebSearchService:
    """Сервис интернет-поиска с правилами и LLM-классификатором."""

    def __init__(self, settings: Settings, openrouter: OpenRouterClient) -> None:
        self._settings = settings
        self._openrouter = openrouter

    async def should_search(
        self,
        text: str,
        *,
        user_search_enabled: bool = True,
    ) -> SearchDecision:
        """Определить, нужен ли интернет-поиск.

        Сначала проверяем глобальный и per-user флаги.
        Затем — простые правила. Если правила неоднозначны — LLM-классификатор.
        """
        if not self._settings.web_search_enabled:
            return SearchDecision(False, reason="Поиск отключён глобально")
        if not user_search_enabled:
            return SearchDecision(False, reason="Поиск отключён пользователем")

        # Этап 1: простые правила
        if _matches_rules(text):
            return SearchDecision(True, query=_query_from_text(text), reason="rules")

        # Этап 2: LLM-классификатор
        try:
            decision = await self._openrouter.chat_structured(
                WEB_SEARCH_CLASSIFY_PROMPT.format(user_message=text),
                "",
                WebSearchDecision,
            )
        except OpenRouterError as exc:
            logger.warning("LLM-классификатор поиска недоступен: %s", exc)
            return SearchDecision(False, reason="classifier_error")
        except Exception as exc:  # pragma: no cover - защита от неожиданных ошибок
            logger.warning("Неожиданная ошибка классификатора поиска: %s", exc)
            return SearchDecision(False, reason="classifier_error")

        if decision.needs_web_search:
            query = decision.search_query.strip() or _query_from_text(text)
            return SearchDecision(True, query=query, reason=decision.reason)
        return SearchDecision(False, reason=decision.reason or "not_needed")

    # ------------------------------------------------------------------
    # Поиск через DDGS
    # ------------------------------------------------------------------
    async def search(self, query: str) -> list[SearchResult]:
        """Выполнить поиск через DDGS в отдельном потоке.

        Все ошибки (нет интернета, таймаут, блокировка, пустой результат)
        ловятся — возвращается пустой список, бот не падает.
        """
        if not query.strip():
            return []
        limit = self._settings.web_search_results_limit
        try:
            raw = await asyncio.to_thread(_sync_ddgs_text, query, limit)
        except Exception as exc:
            logger.warning("DDGS поиск недоступен: %s: %s", exc.__class__.__name__, exc)
            return []
        return _clean_results(raw, limit)


def _sync_ddgs_text(query: str, limit: int) -> list[dict]:
    """Синхронный вызов DDGS.text. Выполняется в отдельном потоке."""
    from ddgs import DDGS

    ddgs = DDGS()
    return list(ddgs.text(query=query, max_results=limit, region="ru-ru"))


def _clean_results(raw: list[dict], limit: int) -> list[SearchResult]:
    """Очистить результаты: убрать дубли URL, пустые заголовки/URL."""
    seen_urls: set[str] = set()
    clean: list[SearchResult] = []
    for item in raw:
        title = str(item.get("title", "")).strip()
        # DDGS.text возвращает 'href', иногда 'url'
        url = str(item.get("href") or item.get("url") or "").strip()
        body = str(item.get("body") or item.get("description") or "").strip()
        if not title or not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        clean.append(SearchResult(title=title, url=url, description=body))
        if len(clean) >= limit:
            break
    return clean


def format_search_results(results: list[SearchResult]) -> str:
    """Отформатировать результаты для подстановки в системный промпт."""
    if not results:
        return "Поиск не дал результатов."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   URL: {r.url}")
        if r.description:
            lines.append(f"   Описание: {r.description}")
    return "\n".join(lines)


__all__ = [
    "WebSearchService",
    "SearchDecision",
    "SearchResult",
    "format_search_results",
]
