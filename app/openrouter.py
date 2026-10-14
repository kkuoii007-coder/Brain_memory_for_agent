"""Клиент OpenRouter для диалоговых и embedding-моделей.

Использует httpx с ретраями на временные ошибки (429, 5xx, сетевые).
Чат-вызовы и извлечение структурированного JSON проходят через единый
метод _post_json. Эмбеддинги возвращают list[float].
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Сетевые/таймаут-ошибки, которые считаем "временными" для ретраев
_RETRYABLE_NETWORK_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


class OpenRouterError(Exception):
    """Критическая ошибка OpenRouter (после всех ретраев)."""


class OpenRouterClient:
    """Асинхронный клиент OpenRouter с ретраями."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Создать HTTP-клиент с заголовками авторизации."""
        self._client = httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.openrouter_timeout),
        )
        logger.info("OpenRouter клиент готов (модели не логируются)")

    async def close(self) -> None:
        """Корректно закрыть HTTP-клиент."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("OpenRouter клиент закрыт")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OpenRouterClient не инициализирован: вызовите connect()")
        return self._client

    # ------------------------------------------------------------------
    # Внутренний POST с ретраями
    # ------------------------------------------------------------------
    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST-запрос к OpenRouter с ретраями на временные ошибки."""
        max_retries = self._settings.openrouter_max_retries
        base_delay = self._settings.openrouter_retry_base_delay
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = await self.client.post(endpoint, json=payload)
            except _RETRYABLE_NETWORK_EXC as exc:
                last_exc = exc
                logger.warning(
                    "Сетевая ошибка OpenRouter (попытка %s/%s): %s",
                    attempt,
                    max_retries,
                    exc.__class__.__name__,
                )
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "OpenRouter вернул HTTP %s (попытка %s/%s)",
                        resp.status_code,
                        attempt,
                        max_retries,
                    )
                    last_exc = OpenRouterError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    # Не временная ошибка — не ретраим
                    raise OpenRouterError(
                        f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}"
                    )

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        raise OpenRouterError(
            f"OpenRouter недоступен после {max_retries} попыток: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Чат-комплишен
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Отправить запрос к диалоговой модели и вернуть текст ответа.

        messages: список {"role": "system"|"user"|"assistant", "content": ...}.
        """
        payload: dict[str, Any] = {
            "model": self._settings.openrouter_chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = await self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Некорректный ответ chat: {data}") from exc

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------
    async def embed(self, text: str) -> list[float]:
        """Получить embedding текста. Возвращает list[float]."""
        payload = {
            "model": self._settings.openrouter_embedding_model,
            "input": text,
        }
        data = await self._post("/embeddings", payload)
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Некорректный ответ embeddings: {data}") from exc

    # ------------------------------------------------------------------
    # Структурированный запрос: чат + парсинг в Pydantic-схему
    # ------------------------------------------------------------------
    async def chat_structured(
        self,
        system_prompt: str,
        user_text: str,
        schema: Type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        """Вызвать чат, распарсить JSON и провалидировать через Pydantic.

        Из ответа модели извлекается JSON (в т.ч. если он обёрнут в ```json ... ```),
        валидируется схемой schema и возвращается объект.
        При некорректном JSON/схеме бросает OpenRouterError.
        """
        raw = await self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
        )
        json_text = _extract_json(raw)
        try:
            return schema.model_validate_json(json_text)
        except ValidationError as exc:
            logger.warning(
                "Pydantic-валидация не прошла: %s. Сырой ответ: %.500s",
                exc,
                raw,
            )
            raise OpenRouterError("Некорректный структурированный ответ") from exc


def _extract_json(text: str) -> str:
    """Извлечь JSON из ответа модели.

    Поддерживает:
    - чистый JSON;
    - JSON, обёрнутый в ```json ... ```;
    - JSON, обёрнутый в ``` ... ```.
    """
    text = text.strip()
    # Убираем markdown code-fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # Ищем первый {...} блок
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


__all__ = ["OpenRouterClient", "OpenRouterError"]
