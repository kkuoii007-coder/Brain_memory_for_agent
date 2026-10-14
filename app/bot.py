"""Точка входа: Telegram-бот на aiogram 3.x.

Обрабатывает текстовые сообщения и команды. Оркестрация делегирована
ChatService. Длинные ответы разбиваются на части ≤4096 символов.

Команды:
  /start         — описание бота
  /help          — список команд
  /memory        — показать активные факты
  /forget_all    — удалить долговременную память (с подтверждением)
  /clear_history — очистить краткосрочную историю SQLite
  /reset         — удалить историю и память (с подтверждением)
  /privacy       — какие данные сохраняются
  /search_on     — включить автопоиск
  /search_off    — отключить автопоиск
"""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.chat_service import ChatService
from app.config import Settings, load_settings
from app.database import Database
from app.memory import MemoryHit, MemoryStore
from app.openrouter import OpenRouterClient
from app.web_search import WebSearchService

logger = logging.getLogger(__name__)


# --- Состояния FSM для подтверждений ---
class ConfirmState(StatesGroup):
    """Состояния подтверждения деструктивных команд."""

    forget_all = State()
    reset = State()


# --- Тексты команд ---
START_TEXT = (
    "Привет! Я бот с гибридной памятью и условным интернет-поиском.\n\n"
    "Я запоминаю важные факты о тебе (аллергии, предпочтения, цели) и "
    "учитываю их в будущих диалогах. Краткосрочная история хранится в SQLite, "
    "долговременные факты — в локальной векторной базе Qdrant.\n\n"
    "Отправь /help, чтобы увидеть все команды."
)

HELP_TEXT = (
    "Доступные команды:\n\n"
    "/start — описание бота\n"
    "/help — этот текст\n"
    "/memory — показать активные сохранённые факты\n"
    "/forget_all — удалить долговременную память (с подтверждением)\n"
    "/clear_history — очистить краткосрочную историю сообщений\n"
    "/reset — удалить историю и долговременную память (с подтверждением)\n"
    "/privacy — какие данные сохраняются\n"
    "/search_on — включить автоматический интернет-поиск\n"
    "/search_off — отключить автоматический интернет-поиск\n\n"
    "Просто напиши сообщение — я отвечу, используя память и при необходимости интернет."
)

PRIVACY_TEXT = (
    "Какие данные я сохраняю:\n\n"
    "1. Краткосрочная память (SQLite, локально):\n"
    "   — последние {limit} сообщений диалога для контекста.\n\n"
    "2. Долговременная память (Qdrant, локально):\n"
    "   — важные факты (аллергии, предпочтения, цели и т.п.);\n"
    "   — критические факты (аллергии, лекарства) учитываются всегда;\n"
    "   — не сохраняются пароли, токены, платёжные данные, приветствия и одноразовые фразы.\n\n"
    "3. Настройки (SQLite):\n"
    "   — флаг автопоиска для твоего аккаунта.\n\n"
    "Все данные хранятся только локально на твоём компьютере и изолированы по твоему Telegram ID. "
    "Никакие данные не передаются третьим лицам, кроме запросов к OpenRouter (диалоговая/embedding-модель) "
    "и DuckDuckGo (поиск)."
)


def _format_memory(facts: list[MemoryHit]) -> str:
    """Отформатировать список фактов для команды /memory."""
    if not facts:
        return "Активных сохранённых фактов нет."
    lines: list[str] = ["Активные факты о тебе:\n"]
    critical = [f for f in facts if f.critical]
    normal = [f for f in facts if not f.critical]
    if critical:
        lines.append("[Критические]")
        for f in critical:
            lines.append(f"  • {f.text}  (категория: {f.category})")
        lines.append("")
    if normal:
        lines.append("[Прочие]")
        for f in normal:
            lines.append(f"  • {f.text}  (категория: {f.category})")
    return "\n".join(lines)


def _split_text(text: str, max_length: int = 4000) -> list[str]:
    """Разбить длинный текст на части для Telegram (≤4096 символов).

    Пытаемся разбивать по границам строк, затем — по пробелам.
    """
    if len(text) <= max_length:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        # Ищем перевод строки ближе к концу лимита
        cut = remaining.rfind("\n", 0, max_length)
        if cut == -1 or cut < max_length // 2:
            # Ищем пробел
            cut = remaining.rfind(" ", 0, max_length)
        if cut == -1:
            cut = max_length
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


# =====================================================================
# Контекст приложения (глобальные ресурсы)
# =====================================================================
class AppContext:
    """Контейнер общих ресурсов приложения."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.openrouter = OpenRouterClient(settings)
        self.memory = MemoryStore(settings)
        self.web_search = WebSearchService(settings, self.openrouter)
        self.chat_service = ChatService(
            settings, self.database, self.openrouter, self.memory, self.web_search
        )
        self._vector_size: int | None = None

    async def startup(self) -> None:
        """Инициализация всех ресурсов."""
        await self.database.connect()
        await self.openrouter.connect()
        # Определяем размерность embedding-модели пробным запросом
        vector_size = self.settings.openrouter_embedding_dim
        if not vector_size:
            try:
                vec = await self.openrouter.embed("test")
                vector_size = len(vec)
                logger.info("Определена размерность embedding: %s", vector_size)
            except Exception as exc:
                logger.error("Не удалось определить размерность embedding: %s", exc)
                raise
        self._vector_size = vector_size
        # Qdrant Local Mode (синхронный клиент)
        self.memory.connect(vector_size)
        logger.info("Приложение готово к работе")

    async def shutdown(self) -> None:
        """Корректное закрытие всех ресурсов."""
        self.memory.close()
        await self.openrouter.close()
        await self.database.close()
        logger.info("Приложение остановлено")


# =====================================================================
# Handlers
# =====================================================================
async def cmd_start(message: Message, context: AppContext) -> None:
    """Обработчик /start."""
    _ = context  # контекст доступен, но /start его не требует (зарегистрирован единообразно)
    await message.answer(START_TEXT)


async def cmd_help(message: Message, context: AppContext) -> None:
    """Обработчик /help."""
    _ = context
    await message.answer(HELP_TEXT)


async def cmd_privacy(message: Message, context: AppContext) -> None:
    """Обработчик /privacy."""
    user_id = message.from_user.id
    stats = await context.chat_service.get_user_stats(user_id)
    text = PRIVACY_TEXT.format(limit=context.settings.recent_messages_limit)
    text += f"\n\nВаша краткосрочная история: {stats['messages_count']} сообщений в SQLite."
    facts = context.chat_service.list_user_memory(user_id)
    text += f"\nДолговременных фактов: {len(facts)}."
    await message.answer(text)


async def cmd_memory(message: Message, context: AppContext) -> None:
    """Обработчик /memory."""
    user_id = message.from_user.id
    facts = context.chat_service.list_user_memory(user_id)
    text = _format_memory(facts)
    for part in _split_text(text):
        await message.answer(part)


async def cmd_clear_history(message: Message, context: AppContext) -> None:
    """Обработчик /clear_history — без подтверждения (история не критична)."""
    user_id = message.from_user.id
    count = await context.chat_service.clear_user_history(user_id)
    await message.answer(
        f"Краткосрочная история очищена: удалено {count} сообщений.\n"
        "Долговременная память сохранена."
    )


async def cmd_forget_all(message: Message, state: FSMContext) -> None:
    """Обработчик /forget_all — запрашивает подтверждение."""
    await state.set_state(ConfirmState.forget_all)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, удалить", callback_data="forget_all:yes"),
                InlineKeyboardButton(text="Отмена", callback_data="forget_all:no"),
            ]
        ]
    )
    await message.answer(
        "Удалить всю долговременную память? Это действие необратимо.",
        reply_markup=keyboard,
    )


async def cmd_reset(message: Message, state: FSMContext) -> None:
    """Обработчик /reset — запрашивает подтверждение."""
    await state.set_state(ConfirmState.reset)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, удалить всё", callback_data="reset:yes"),
                InlineKeyboardButton(text="Отмена", callback_data="reset:no"),
            ]
        ]
    )
    await message.answer(
        "Удалить и краткосрочную историю, и долговременную память? Это действие необратимо.",
        reply_markup=keyboard,
    )


async def cmd_search_on(message: Message, context: AppContext) -> None:
    """Обработчик /search_on."""
    user_id = message.from_user.id
    await context.chat_service.set_user_search(user_id, True)
    await message.answer("Автоматический интернет-поиск включён.")


async def cmd_search_off(message: Message, context: AppContext) -> None:
    """Обработчик /search_off."""
    user_id = message.from_user.id
    await context.chat_service.set_user_search(user_id, False)
    await message.answer("Автоматический интернет-поиск отключён.")


async def on_confirm(callback: CallbackQuery, state: FSMContext, context: AppContext) -> None:
    """Обработчик кнопок подтверждения для forget_all и reset."""
    user_id = callback.from_user.id
    data = callback.data or ""
    current_state = await state.get_state()

    if current_state == ConfirmState.forget_all.state:
        if data == "forget_all:yes":
            context.chat_service.delete_user_memory(user_id)
            await callback.message.edit_text("Долговременная память удалена.")
        else:
            await callback.message.edit_text("Отменено.")
        await state.clear()
    elif current_state == ConfirmState.reset.state:
        if data == "reset:yes":
            await context.chat_service.clear_user_history(user_id)
            context.chat_service.delete_user_memory(user_id)
            await callback.message.edit_text("История и долговременная память удалены.")
        else:
            await callback.message.edit_text("Отменено.")
        await state.clear()
    else:
        await callback.answer("Действие устарело.")
    await callback.answer()


async def on_text_message(message: Message, context: AppContext) -> None:
    """Обработчик произвольного текстового сообщения."""
    if not message.text or not message.from_user:
        return
    user_id = message.from_user.id
    username = message.from_user.username
    text = message.text.strip()
    # Ограничение длины входного сообщения
    if len(text) > context.settings.max_input_length:
        await message.answer(
            f"Сообщение слишком длинное (макс. {context.settings.max_input_length} символов). "
            "Сократи его, пожалуйста."
        )
        return

    # Индикатор набора ("печатает...")
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    result = await context.chat_service.handle_message(
        telegram_user_id=user_id,
        username=username,
        text=text,
        telegram_message_id=message.message_id,
    )

    # Разбиваем длинный ответ
    for part in _split_text(result.reply, context.settings.telegram_max_message_length - 10):
        try:
            await message.answer(part)
        except TelegramNetworkError as exc:
            logger.warning("Не удалось отправить часть ответа: %s", exc)


# =====================================================================
# Регистрация роутеров
# =====================================================================
def register_handlers(dp: Dispatcher, ctx: AppContext) -> None:
    """Зарегистрировать все обработчики."""
    # Команды
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_privacy, Command("privacy"))
    dp.message.register(cmd_memory, Command("memory"))
    dp.message.register(cmd_clear_history, Command("clear_history"))
    dp.message.register(cmd_forget_all, Command("forget_all"))
    dp.message.register(cmd_reset, Command("reset"))
    dp.message.register(cmd_search_on, Command("search_on"))
    dp.message.register(cmd_search_off, Command("search_off"))
    # Подтверждения (inline кнопки)
    dp.callback_query.register(on_confirm, F.data.startswith(("forget_all:", "reset:")))
    # Текстовые сообщения (последним)
    dp.message.register(on_text_message, F.text)


# =====================================================================
# Точка входа
# =====================================================================
async def main() -> None:
    """Запуск бота."""
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Не логируем секреты: в формат не выводятся переменные окружения

    ctx = AppContext(settings)
    await ctx.startup()

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    register_handlers(dp, ctx)

    # aiogram 3.x автоматически прокидывает ключи workflow_data в параметры
    # обработчиков по имени: параметр `context` получит значение ctx.
    dp["context"] = ctx

    logger.info("Бот запускается…")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        await ctx.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановлен пользователем")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Фатальная ошибка: %s", exc)
        sys.exit(1)
