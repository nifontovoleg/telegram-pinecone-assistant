"""Telegram-бот-помощник на pyTelegramBotAPI с памятью в Pinecone."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telebot.async_telebot import AsyncTeleBot
from telebot.types import Message

from pinecone_manager import MemoryWriteResult, PineconeManager, SearchHit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    pinecone_index_name: str
    pinecone_integrated: bool
    embedding_model: str
    memory_top_k: int
    system_prompt: str


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise ValueError("OPENAI_API_KEY не задан в .env")

    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME не задан в .env")

    return Settings(
        telegram_bot_token=token,
        openai_api_key=openai_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        pinecone_index_name=index_name,
        pinecone_integrated=_env_bool("PINECONE_INTEGRATED", False),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        memory_top_k=int(os.getenv("MEMORY_TOP_K", "8")),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            (
                "Ты дружелюбный Telegram-помощник. Отвечай по-русски, кратко и по делу. "
                "Используй известные факты о пользователе из контекста памяти. "
                "Не начинай каждый ответ с приветствия — здоровайся только если это "
                "первое сообщение пользователя. "
                "Если информации недостаточно — честно скажи об этом."
            ),
        ),
    )


class TelegramBot:
    """Telegram-бот с долговременной памятью в Pinecone."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bot = AsyncTeleBot(settings.telegram_bot_token)
        self._pinecone = PineconeManager(
            index_name=settings.pinecone_index_name,
            integrated=settings.pinecone_integrated,
        )
        self._openai = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    @staticmethod
    def _user_filter(user_id: int) -> dict[str, Any]:
        return {"user_id": str(user_id), "type": "user_message"}

    @staticmethod
    def _build_message_id(user_id: int, timestamp: str) -> str:
        return f"user_{user_id}_msg_{timestamp}"

    @staticmethod
    def _build_timestamp() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def _build_metadata(message: Message, text: str, timestamp: str) -> dict[str, str]:
        user = message.from_user
        if user is None:
            raise ValueError("message.from_user is required")

        first_name = user.first_name or ""
        return {
            "first_name": first_name,
            "language_code": user.language_code or "",
            "text": text,
            "timestamp": timestamp,
            "type": "user_message",
            "user_id": str(user.id),
            "username": user.username or "",
        }

    async def _embed(self, text: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=self._settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def _save_user_message(self, message: Message) -> MemoryWriteResult | None:
        user = message.from_user
        if user is None:
            return None

        text = (message.text or "").strip()
        if not text:
            return None

        timestamp = self._build_timestamp()
        memory_id = self._build_message_id(user.id, timestamp)
        metadata = self._build_metadata(message, text, timestamp)
        user_filter = self._user_filter(user.id)

        if self._pinecone.integrated:
            return await self._pinecone.remember_text(
                text,
                memory_id=memory_id,
                metadata_filter=user_filter,
                **metadata,
            )

        vector = await self._embed(text)
        return await self._pinecone.remember_vector(
            vector,
            memory_id=memory_id,
            metadata=metadata,
            metadata_filter=user_filter,
        )

    async def _get_user_context(
        self,
        user_id: int,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        limit = top_k or self._settings.memory_top_k
        user_filter = self._user_filter(user_id)

        if self._pinecone.integrated:
            return await self._pinecone.search_by_text(
                query,
                top_k=limit,
                metadata_filter=user_filter,
            )

        vector = await self._embed(query)
        return await self._pinecone.search_by_vector(
            vector,
            top_k=limit,
            metadata_filter=user_filter,
        )

    @staticmethod
    def _format_context(hits: list[SearchHit]) -> str:
        if not hits:
            return "Сохранённых сообщений пока нет."

        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            text = _extract_text(hit)
            if text:
                lines.append(f"{index}. {text}")

        return "\n".join(lines) if lines else "Сохранённых сообщений пока нет."

    @staticmethod
    def _extract_preferred_name(message: Message, text: str) -> str | None:
        """Ник из префикса ;name или профиля Telegram."""
        nick_match = re.match(r"^;(\S+)", text.strip())
        if nick_match:
            return nick_match.group(1)

        user = message.from_user
        if user is None:
            return None
        if user.first_name:
            return user.first_name
        if user.username:
            return user.username.lstrip("@")
        return None

    async def _has_prior_messages(self, user_id: int) -> bool:
        """Есть ли у пользователя сохранённые сообщения (до текущего)."""
        user_filter = self._user_filter(user_id)

        if self._pinecone.integrated:
            hits = await self._pinecone.search_by_text(
                "история сообщений пользователя",
                top_k=1,
                metadata_filter=user_filter,
            )
        else:
            vector = await self._embed("история сообщений пользователя")
            hits = await self._pinecone.search_by_vector(
                vector,
                top_k=1,
                metadata_filter=user_filter,
            )

        return bool(hits)

    @staticmethod
    def _format_user_profile(
        message: Message,
        text: str,
        *,
        is_first_message: bool,
    ) -> str:
        user = message.from_user
        if user is None:
            return "Профиль пользователя неизвестен."

        preferred_name = TelegramBot._extract_preferred_name(message, text)
        lines = [
            f"Имя в Telegram: {user.first_name or 'не указано'}",
            f"Username: @{user.username}" if user.username else "Username: не указан",
        ]
        if preferred_name:
            lines.append(f"Как обращаться: {preferred_name}")
        if is_first_message and preferred_name:
            lines.append(
                "Это первое сообщение пользователя — один раз поприветствуй его по имени "
                f"(без символа «;», например: «Привет, {preferred_name}!»)."
            )
        elif not is_first_message:
            lines.append(
                "Диалог уже идёт — отвечай по существу, без «Привет» и без повторного "
                "приветствия по имени."
            )
        return "\n".join(lines)

    async def _generate_reply(
        self,
        message: Message,
        user_message: str,
        *,
        is_first_message: bool,
    ) -> str:
        user = message.from_user
        if user is None:
            return "Не удалось определить пользователя."

        hits = await self._get_user_context(user.id, user_message)
        memory_context = self._format_context(hits)
        user_profile = self._format_user_profile(
            message,
            user_message,
            is_first_message=is_first_message,
        )

        response = await self._openai.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{self._settings.system_prompt}\n\n"
                        f"{user_profile}\n\n"
                        f"Известные сообщения пользователя:\n{memory_context}"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )

        answer = response.choices[0].message.content
        if not answer:
            return "Не удалось сформировать ответ. Попробуйте переформулировать сообщение."
        return answer.strip()

    async def _clear_user_memory(self, user_id: int) -> None:
        await self._pinecone.delete_by_filter(self._user_filter(user_id))

    def register_handlers(self) -> None:
        @self._bot.message_handler(commands=["start"])
        async def handle_start(message: Message) -> None:
            await self._bot.reply_to(
                message,
                (
                    "Привет! Я персональный помощник с долговременной памятью.\n\n"
                    "Пишите мне как обычному собеседнику — я запоминаю ваши сообщения "
                    "и использую их в следующих разговорах.\n\n"
                    "Команды:\n"
                    "/help — справка\n"
                    "/memory — что я помню о вас\n"
                    "/forget — очистить мою память о вас"
                ),
            )

        @self._bot.message_handler(commands=["help"])
        async def handle_help(message: Message) -> None:
            await self._bot.reply_to(
                message,
                (
                    "Я отвечаю на ваши сообщения и сохраняю только ваши сообщения в Pinecone.\n\n"
                    "Примеры:\n"
                    "• «Меня зовут Олег, я Python-разработчик»\n"
                    "• «Напомни, чем я занимаюсь?»\n\n"
                    "/memory — показать релевантные воспоминания\n"
                    "/forget — удалить все сохранённые данные о вас"
                ),
            )

        @self._bot.message_handler(commands=["memory"])
        async def handle_memory(message: Message) -> None:
            user = message.from_user
            if user is None:
                return

            query = (message.text or "").replace("/memory", "", 1).strip()
            if not query:
                query = "вся информация о пользователе"

            hits = await self._get_user_context(user.id, query, top_k=10)
            context = self._format_context(hits)
            await self._bot.reply_to(message, f"Что я помню:\n\n{context}")

        @self._bot.message_handler(commands=["forget"])
        async def handle_forget(message: Message) -> None:
            user = message.from_user
            if user is None:
                return

            await self._clear_user_memory(user.id)
            await self._bot.reply_to(
                message,
                "Память о вас очищена. Можем начать с чистого листа.",
            )

        @self._bot.message_handler(content_types=["text"])
        async def handle_text(message: Message) -> None:
            await self._handle_text(message)

    async def _handle_text(self, message: Message) -> None:
        user = message.from_user
        text = (message.text or "").strip()
        if user is None or not text or text.startswith("/"):
            return

        await self._bot.send_chat_action(message.chat.id, "typing")

        try:
            is_first_message = not await self._has_prior_messages(user.id)

            save_result = await self._save_user_message(message)
            if save_result is not None:
                logger.info(
                    "Сообщение сохранено: %s action: %s",
                    save_result.memory_id,
                    save_result.action,
                )

            answer = await self._generate_reply(
                message,
                text,
                is_first_message=is_first_message,
            )
            await self._bot.reply_to(message, answer)
        except Exception:
            logger.exception("Ошибка обработки сообщения user_id=%s", user.id)
            await self._bot.reply_to(
                message,
                "Произошла ошибка при обработке сообщения. Попробуйте позже.",
            )

    async def run(self) -> None:
        await self._pinecone.connect()
        self.register_handlers()

        logger.info("Бот инициализирован успешно")
        logger.info("Запуск бота...")
        index_type = "integrated" if self._settings.pinecone_integrated else "standard"
        logger.info(
            "Индекс Pinecone: %s (%s)",
            self._settings.pinecone_index_name,
            index_type,
        )

        try:
            await self._bot.infinity_polling(timeout=30, request_timeout=60)
        finally:
            await self._pinecone.close()


def _extract_text(hit: SearchHit) -> str:
    if hit.fields:
        for key in ("chunk_text", "text", "content"):
            value = hit.fields.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    metadata = hit.metadata or {}
    for key in ("chunk_text", "text", "content"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


async def main() -> None:
    bot = TelegramBot(load_settings())
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
