"""Краткосрочная память на SQLite через aiosqlite.

Хранит:
- пользователей (users);
- сообщения (messages) — последние N для контекста;
- per-user настройки (user_settings) — флаг автопоиска и т.п.

Все SQL-запросы параметризованные. История разных пользователей
изолирована фильтром WHERE telegram_user_id = ?.
"""
from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from app.config import Settings

logger = logging.getLogger(__name__)


class Database:
    """Асинхронная обёртка над SQLite для краткосрочной памяти."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Открыть соединение и создать таблицы, если их нет."""
        path = self._settings.db_file
        self._db = await aiosqlite.connect(str(path))
        # Включаем WAL для лучшей конкурентности (best-effort, ошибки не критичны)
        try:
            await self._db.execute("PRAGMA journal_mode=WAL;")
        except Exception:  # pragma: no cover - зависит от ОС
            pass
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._create_tables()
        await self._db.commit()
        logger.info("SQLite подключена: %s", path)

    async def close(self) -> None:
        """Корректно закрыть соединение."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("SQLite закрыта")

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database не инициализирована: вызовите connect() сначала")
        return self._db

    # ------------------------------------------------------------------
    # Создание схемы
    # ------------------------------------------------------------------
    async def _create_tables(self) -> None:
        db = self.db
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER NOT NULL UNIQUE,
                username TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER NOT NULL,
                telegram_message_id INTEGER,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_created
                ON messages(telegram_user_id, created_at DESC);
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                telegram_user_id INTEGER PRIMARY KEY,
                web_search_enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Пользователи
    # ------------------------------------------------------------------
    async def upsert_user(self, telegram_user_id: int, username: str | None) -> None:
        """Создать или обновить запись пользователя."""
        db = self.db
        await db.execute(
            """
            INSERT INTO users (telegram_user_id, username)
            VALUES (?, ?)
            ON CONFLICT(telegram_user_id) DO UPDATE SET
                username = excluded.username,
                updated_at = datetime('now');
            """,
            (telegram_user_id, username),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Сообщения
    # ------------------------------------------------------------------
    async def save_message(
        self,
        telegram_user_id: int,
        role: str,
        content: str,
        telegram_message_id: int | None = None,
    ) -> None:
        """Сохранить сообщение в SQLite.

        role: 'user' или 'assistant'. Текущее сообщение сохраняется здесь,
        и get_recent_messages затем возвращает его вместе с историей —
        поэтому дублирования в контексте не происходит.
        """
        db = self.db
        await db.execute(
            """
            INSERT INTO messages
                (telegram_user_id, telegram_message_id, role, content)
            VALUES (?, ?, ?, ?);
            """,
            (telegram_user_id, telegram_message_id, role, content),
        )
        await db.commit()

    async def get_recent_messages(
        self, telegram_user_id: int, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Получить последние N сообщений пользователя от старых к новым.

        Возвращает список словарей {"role": ..., "content": ...}.
        Считаются суммарно user и assistant. История разных пользователей
        не смешивается благодаря WHERE telegram_user_id = ?.
        """
        if limit is None:
            limit = self._settings.recent_messages_limit
        db = self.db
        # Берём последние N по убыванию created_at, id (для детерминизма
        # при одинаковых created_at), затем реверсим в Python → старые→новые.
        cursor = await db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE telegram_user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?;
            """,
            (telegram_user_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        rows = list(reversed(rows))
        return [{"role": role, "content": content} for role, content in rows]

    async def clear_history(self, telegram_user_id: int) -> int:
        """Удалить все сообщения пользователя из SQLite.

        Возвращает количество удалённых строк. Долговременная память
        при этом не затрагивается.
        """
        db = self.db
        cursor = await db.execute(
            "DELETE FROM messages WHERE telegram_user_id = ?;",
            (telegram_user_id,),
        )
        await db.commit()
        count = cursor.rowcount
        await cursor.close()
        logger.info(
            "Очищена история сообщений пользователя %s: удалено %s",
            telegram_user_id,
            count,
        )
        return count

    # ------------------------------------------------------------------
    # Настройки пользователя
    # ------------------------------------------------------------------
    async def get_web_search_enabled(self, telegram_user_id: int) -> bool:
        """Вернуть флаг автопоиска для пользователя (по умолчанию True)."""
        db = self.db
        cursor = await db.execute(
            "SELECT web_search_enabled FROM user_settings WHERE telegram_user_id = ?;",
            (telegram_user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            # Глобальный дефолт
            return self._settings.web_search_enabled
        return bool(row[0])

    async def set_web_search_enabled(self, telegram_user_id: int, enabled: bool) -> None:
        """Установить флаг автопоиска для пользователя."""
        db = self.db
        await db.execute(
            """
            INSERT INTO user_settings (telegram_user_id, web_search_enabled, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(telegram_user_id) DO UPDATE SET
                web_search_enabled = excluded.web_search_enabled,
                updated_at = datetime('now');
            """,
            (telegram_user_id, int(enabled)),
        )
        await db.commit()

    async def get_user_stats(self, telegram_user_id: int) -> dict[str, Any]:
        """Краткая статистика пользователя для команды /privacy."""
        db = self.db
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE telegram_user_id = ?;",
            (telegram_user_id,),
        )
        (messages_count,) = await cursor.fetchone()
        await cursor.close()
        return {"messages_count": messages_count}


__all__ = ["Database"]
