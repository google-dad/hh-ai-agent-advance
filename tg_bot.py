from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InputRichBlockDetails,
    InputRichBlockDivider,
    InputRichBlockParagraph,
    InputRichBlockSectionHeading,
    InputRichMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    RichTextBold,
    RichTextUrl,
)

from approval import ApprovalService
from config import Settings
from database import Database, Vacancy, VacancyStatus
from fit_summary import FIT_SUMMARY_FALLBACK
from llm.errors import LLMError
from llm.api_keys import ApiKeyCheckResult, ApiKeyManager, ApiKeyView
from llm.errors import LLMError
from updater import ReleaseInfo, check_github_release
from version import __version__


logger = logging.getLogger(__name__)
PRIVATE_REPLY = "This bot is private."
CARD_TEXT_LIMIT = 3000
API_KEY_INPUT_TTL = timedelta(minutes=15)
# Backward-compatible alias used by older tests/imports.
MISTRAL_KEY_INPUT_TTL = API_KEY_INPUT_TTL
MENU_PARSE = "Парсинг"
MENU_APPROVE = "Апрув"
MENU_PAUSE = "Пауза"
MENU_RESUME = "Продолжить"
MENU_APPLIED = "Отклики"
MENU_PENDING = "Ожидают"
MENU_STATUS = "Статус"
MENU_DIAGNOSTICS = "Диагностика"
MENU_LABELS = frozenset(
    {
        MENU_PARSE,
        MENU_APPROVE,
        MENU_PAUSE,
        MENU_RESUME,
        MENU_APPLIED,
        MENU_PENDING,
        MENU_STATUS,
        MENU_DIAGNOSTICS,
    }
)


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=MENU_PARSE), KeyboardButton(text=MENU_APPROVE)],
            [KeyboardButton(text=MENU_PAUSE), KeyboardButton(text=MENU_RESUME)],
            [KeyboardButton(text=MENU_APPLIED), KeyboardButton(text=MENU_PENDING)],
            [KeyboardButton(text=MENU_STATUS), KeyboardButton(text=MENU_DIAGNOSTICS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


@dataclass
class AgentControl:
    paused: bool = False
    app_mode: str = ""
    circuit_reason: str = ""
    consecutive_search_errors: int = 0
    next_run_at: datetime | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)




@dataclass
class ApiKeyInputSession:
    expires_at: datetime


MistralKeyInputSession = ApiKeyInputSession


class CoverLetterGenerator(Protocol):
    async def generate_cover_letter(
        self, vacancy_title: str, vacancy_description: str
    ) -> str: ...


class TelegramService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        approval_service: ApprovalService,
        control: AgentControl,
        *,
        api_keys: ApiKeyManager | None = None,
        mistral_keys: ApiKeyManager | None = None,
        bot: Any | None = None,
        dispatcher: Dispatcher | None = None,
        now_factory: Callable[[], datetime] | None = None,
        cover_letters: CoverLetterGenerator | None = None,
    ):
        self.settings = settings
        self.database = database
        self.approval_service = approval_service
        self.control = control
        self.cover_letters = cover_letters
        self.api_keys = api_keys if api_keys is not None else mistral_keys
        self.mistral_keys = self.api_keys  # backward-compatible attribute
        self.bot = bot or Bot(token=settings.tg_bot_token)
        self.dispatcher = dispatcher or Dispatcher()
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())
        self._captcha_future: asyncio.Future[str | None] | None = None
        self._edit_job_id: str | None = None
        self._edit_future: asyncio.Future[str | None] | None = None
        self._mistral_key_input: MistralKeyInputSession | None = None
        self.latest_release: ReleaseInfo | None = None
        self._notified_release_tag: str | None = None
        self._register_handlers()

    def authorized(self, user_id: int) -> bool:
        return user_id == self.settings.tg_user_id

    def command(self, name: str, user_id: int) -> str:
        if not self.authorized(user_id):
            return PRIVATE_REPLY
        if name == "start":
            return (
                "Personal HH assistant is ready. "
                "Кнопки под полем ввода: режим, пауза, отклики и статус."
            )
        if name == "pause":
            self.control.paused = True
            self.control.next_run_at = None
            logger.info("agent_paused")
            return "Agent paused."
        if name == "resume":
            self.control.paused = False
            self.control.circuit_reason = ""
            self.control.consecutive_search_errors = 0
            self.control.next_run_at = None
            self.control.wake_event.set()
            logger.info("agent_resumed")
            return "Agent resumed."
        if name == "status":
            stats = self.database.stats()
            processed = sum(stats.values())
            version_str = f"v{__version__}"
            if self.latest_release is not None:
                version_str += f" (доступно обновление {self.latest_release.tag_name})"
            return (
                f"версия: {version_str}\n"
                f"mode: {self._app_mode()}\n"
                f"state: {'paused' if self.control.paused else 'running'}\n"
                f"processed: {processed}\n"
                f"applied today: {self.database.applied_today(self.now_factory())}"
            )
        if name == "pending":
            pending = self.database.pending()
            return (
                "No pending vacancies."
                if not pending
                else "\n".join(f"{item.id}: {item.title}" for item in pending)
            )
        if name == "stats":
            return "\n".join(
                f"{status}: {count}"
                for status, count in self.database.stats().items()
                if count
            ) or "No vacancies recorded."
        if name == "diagnostics":
            return self._diagnostics()
        if name == "cancel":
            if self._captcha_future and not self._captcha_future.done():
                self._captcha_future.set_result(None)
            if self._edit_future and not self._edit_future.done():
                self._edit_future.set_result(None)
            return "Current input request cancelled."
        return "Unknown command."

    def _app_mode(self) -> str:
        return self.control.app_mode or self.settings.app_mode

    def menu_text(self, label: str, user_id: int) -> str | None:
        if label not in MENU_LABELS:
            return None
        if not self.authorized(user_id):
            return PRIVATE_REPLY
        if label == MENU_PARSE:
            self.control.app_mode = "dry_run"
            logger.info("agent_mode_changed app_mode=dry_run")
            return "Режим: парсинг. Новые карточки без кнопок отклика."
        if label == MENU_APPROVE:
            self.control.app_mode = "approval"
            logger.info("agent_mode_changed app_mode=approval")
            return "Режим: апрув. Новые карточки с письмом и откликом."
        if label == MENU_PAUSE:
            return self.command("pause", user_id)
        if label == MENU_RESUME:
            return self.command("resume", user_id)
        if label == MENU_APPLIED:
            return self._applied_list()
        if label == MENU_PENDING:
            return self.command("pending", user_id)
        if label == MENU_STATUS:
            return self.command("status", user_id)
        return self.command("diagnostics", user_id)

    def _applied_list(self) -> str:
        items = self.database.recent_applied(limit=15)
        if not items:
            return "Откликов нет."
        lines = []
        for item in items:
            company = f" — {item.company}" if item.company else ""
            lines.append(f"{item.title}{company}\n{item.url}")
        return "\n\n".join(lines)

    def _diagnostics(self) -> str:
        run = self.database.latest_search_run()
        version_str = f"v{__version__}"
        if self.latest_release is not None:
            version_str += f" (доступно: {self.latest_release.tag_name})"
        if run is None:
            return f"версия: {version_str}\nNo search diagnostics recorded."
        started = datetime.fromisoformat(run.started_at)
        finished = datetime.fromisoformat(run.finished_at)
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.rejection_reasons.items())
        ) or "нет"
        errors = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(run.error_reasons.items())
        ) or "нет"
        breaker = (
            f"open ({self.control.circuit_reason})"
            if self.control.circuit_reason
            else "closed"
        )
        if self.control.paused:
            next_run = "paused"
        elif self.control.next_run_at is None:
            next_run = "после текущего цикла"
        else:
            next_run = self.control.next_run_at.astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        return (
            f"версия: {version_str}\n"
            f"последний цикл: {started.astimezone():%Y-%m-%d %H:%M:%S}\n"
            f"состояние: {run.state}\n"
            f"длительность: {int((finished - started).total_seconds())} с\n"
            f"запросы: {run.query_count}\n"
            f"найдено: {run.found_results}\n"
            f"новые: {run.new_vacancies}\n"
            f"дубли: {run.duplicates}\n"
            f"отклонено — фильтр: {run.rejected_by_filter}, LLM: {run.rejected_by_llm}\n"
            f"причины: {reasons}\n"
            f"карточки: {run.telegram_cards}\n"
            f"ошибки: {run.error_count} ({errors})\n"
            f"последняя ошибка: {run.last_safe_error or 'нет'}\n"
            f"breaker: {breaker}\n"
            f"last breaker: {run.circuit_reason or 'none'}\n"
            f"следующий запуск: {next_run}"
        )

    def _register_handlers(self) -> None:
        for name in (
            "start",
            "status",
            "pause",
            "resume",
            "pending",
            "stats",
            "diagnostics",
            "mistral_keys",
            "keys",
            "cancel",
        ):
            self.dispatcher.message.register(self._command_handler, Command(name))
        self.dispatcher.callback_query.register(
            self._callback_handler,
            F.data.startswith("apply:")
            | F.data.startswith("skip:")
            | F.data.startswith("edit:")
            | F.data.startswith("letter:")
            | F.data.startswith("mk:"),
        )
        self.dispatcher.message.register(self._text_handler)

    async def _command_handler(self, message: Message) -> None:
        name = (message.text or "").split()[0].lstrip("/").split("@")[0]
        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        if name in {"mistral_keys", "keys"}:
            await self._send_mistral_key_menu(message)
            return
        if name == "cancel" and self._mistral_key_input is not None:
            self._mistral_key_input = None
            await message.answer("Current API key input cancelled.")
            return
        reply = self.command(name, message.from_user.id)
        if name == "start" and self.authorized(message.from_user.id):
            await message.answer(reply, reply_markup=main_menu_keyboard())
            return
        await message.answer(reply)

    async def _callback_handler(self, callback: CallbackQuery) -> None:
        if not self.authorized(callback.from_user.id):
            await self._answer_callback(callback, PRIVATE_REPLY, show_alert=True)
            return
        action, separator, job_id = (callback.data or "").partition(":")
        if len((callback.data or "").encode()) > 64:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if action == "mk":
            await self._mistral_key_callback(callback, (callback.data or "").split(":"))
            return
        if not separator or action not in {"apply", "skip", "edit", "letter"} or not job_id:
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        if action == "letter":
            await self._generate_letter(callback, job_id)
        elif action == "apply":
            # Отвечаем на callback немедленно — Telegram требует ответ в течение ~10 сек,
            # а браузерный отклик может занять значительно больше времени.
            await callback.answer("⏳ Отправляем отклик...", show_alert=False)
            result = await self.approval_service.approve_and_apply(
                job_id, callback.from_user.id
            )
            vacancy = self.database.get(job_id)
            if result.ok:
                await self.notify("✓ Отклик отправлен")
                if callback.message:
                    await callback.message.edit_reply_markup(reply_markup=None)
            else:
                if vacancy and vacancy.error_text == "questionnaire_required":
                    await self.notify_questionnaire_required(vacancy.title, vacancy.url)
                    if callback.message:
                        await callback.message.edit_reply_markup(reply_markup=None)
                else:
                    await self.notify(f"✗ {result.message}")
        elif action == "edit":
            self._mistral_key_input = None
            await callback.answer("Пришлите новый текст сопроводительного письма сообщением (или /cancel):", show_alert=True)
            loop = asyncio.get_running_loop()
            self._edit_job_id = job_id
            self._edit_future = loop.create_future()
            try:
                new_letter = await asyncio.wait_for(self._edit_future, timeout=300)
                if new_letter:
                    self.database.update_cover_letter(job_id, new_letter)
                    await self.notify("✓ Сопроводительное письмо обновлено!")
                    updated_vacancy = self.database.get(job_id)
                    if updated_vacancy:
                        await self.send_preview(updated_vacancy, include_actions=True)
                else:
                    await self.notify("Редактирование письма отменено.")
            except TimeoutError:
                await self.notify("Время ожидания редактирования письма истекло.")
            finally:
                self._edit_job_id = None
                self._edit_future = None
        else:
            # skip выполняется быстро — можно отвечать обычным способом.
            result = self.approval_service.skip(job_id, callback.from_user.id)
            await self._answer_callback(
                callback, result.message, show_alert=not result.ok
            )
            if result.ok and callback.message:
                await callback.message.edit_reply_markup(reply_markup=None)

    def _keys_provider_label(self) -> str:
        if self.api_keys is not None:
            return self.api_keys.provider
        return self.settings.llm.provider

    async def _send_mistral_key_menu(self, message: Any) -> None:
        if self.api_keys is None:
            await message.answer(
                "Управление API-ключами недоступно для текущего LLM-провайдера "
                f"({self.settings.llm.provider})."
            )
            return
        try:
            keys = self.api_keys.list_keys()
        except LLMError as exc:
            await message.answer(f"Ключи LLM недоступны: {exc.category}.")
            return
        text, keyboard = self._mistral_key_menu(keys)
        await message.answer(text, reply_markup=keyboard)

    def _mistral_key_menu(
        self, keys: tuple[ApiKeyView, ...]
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        label = self._keys_provider_label()
        lines = [f"Ключи LLM ({label}):"]
        rows: list[list[InlineKeyboardButton]] = []
        for key in keys:
            if key.is_current:
                marker = "✅ текущий"
            elif key.status == "ready":
                marker = "✅ рабочий"
            elif key.status == "cooldown":
                until = key.cooldown_until
                marker = (
                    f"⏸ до {until.astimezone():%H:%M}"
                    if until is not None
                    else "⏸ cooldown"
                )
            elif key.status == "disabled":
                marker = "❌ отключён"
            else:
                marker = key.status
            lines.append(f"{key.id}. {marker} ····{key.suffix}")
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Проверить {key.id}",
                        callback_data=f"mk:check:{key.id}",
                    ),
                    InlineKeyboardButton(
                        text=f"Удалить {key.id}",
                        callback_data=f"mk:delete:{key.id}",
                    ),
                ]
            )
        if not keys:
            lines.append("Ключей нет.")
        rows.append(
            [
                InlineKeyboardButton(text="➕ Добавить", callback_data="mk:add"),
                InlineKeyboardButton(text="Проверить все", callback_data="mk:check"),
            ]
        )
        rows.append(
            [InlineKeyboardButton(text="Обновить", callback_data="mk:list")]
        )
        return "\n".join(lines), self._mistral_keyboard(rows)

    async def _mistral_key_callback(
        self, callback: CallbackQuery, parts: list[str]
    ) -> None:
        if self.api_keys is None:
            await self._answer_callback(
                callback,
                "Управление API-ключами недоступно.",
                show_alert=True,
            )
            return
        if parts == ["mk", "add"]:
            if self._captcha_future and not self._captcha_future.done():
                await self._answer_callback(
                    callback, "Сначала завершите CAPTCHA.", show_alert=True
                )
                return
            if self._edit_future is not None:
                await self._answer_callback(
                    callback,
                    "Сначала завершите редактирование письма.",
                    show_alert=True,
                )
                return
            self._mistral_key_input = ApiKeyInputSession(
                self.now_factory() + API_KEY_INPUT_TTL
            )
            await self._answer_callback(
                callback, "Отправьте ключ одним сообщением."
            )
            return
        if parts == ["mk", "list"]:
            try:
                text, keyboard = self._mistral_key_menu(self.api_keys.list_keys())
            except LLMError as exc:
                await self._answer_callback(
                    callback, f"Ключи LLM недоступны: {exc.category}.", show_alert=True
                )
                return
            await self._edit_mistral_message(callback, text, reply_markup=keyboard)
            return
        if parts == ["mk", "check"]:
            await self._answer_callback(callback, "Проверяем ключи.")
            try:
                results = await self.api_keys.check_all()
            except LLMError as exc:
                await self._edit_mistral_message(
                    callback,
                    f"Проверка не выполнена: {exc.category}.",
                    acknowledge=False,
                )
                return
            text = "\n".join(self._mistral_check_text(result) for result in results)
            await self._edit_mistral_message(
                callback, text or "Ключей для проверки нет.", acknowledge=False
            )
            return
        if len(parts) != 3 or not parts[2].isdecimal():
            await self._answer_callback(
                callback, "Некорректное действие.", show_alert=True
            )
            return
        action, key_id = parts[1], int(parts[2])
        if action == "delete":
            keyboard = self._mistral_keyboard(
                [
                    [
                        InlineKeyboardButton(
                            text="Удалить", callback_data=f"mk:confirm:{key_id}"
                        ),
                        InlineKeyboardButton(text="Отмена", callback_data="mk:list"),
                    ]
                ]
            )
            await self._edit_mistral_message(
                callback, f"Удалить ключ {key_id}?", reply_markup=keyboard
            )
            return
        if action == "check":
            await self._answer_callback(callback, "Проверяем ключ.")
            try:
                result = await self.api_keys.check_key(key_id)
            except LLMError as exc:
                await self._edit_mistral_message(
                    callback,
                    f"Проверка не выполнена: {exc.category}.",
                    acknowledge=False,
                )
                return
            if result is None:
                await self._edit_mistral_message(
                    callback, "Ключ не найден.", acknowledge=False
                )
                return
            await self._edit_mistral_message(
                callback, self._mistral_check_text(result), acknowledge=False
            )
            return
        if action == "confirm":
            try:
                deleted = await self.api_keys.delete_key(key_id)
            except LLMError as exc:
                await self._answer_callback(
                    callback, f"Ключ не удалён: {exc.category}.", show_alert=True
                )
                return
            if not deleted:
                await self._answer_callback(
                    callback, "Ключ не найден.", show_alert=True
                )
                return
            await self._edit_mistral_message(callback, f"Ключ {key_id} удалён.")
            return
        await self._answer_callback(
            callback, "Некорректное действие.", show_alert=True
        )

    @staticmethod
    def _mistral_check_text(result: ApiKeyCheckResult) -> str:
        outcome = "работает" if result.success else result.error_type
        return f"Ключ {result.key.id} ····{result.key.suffix}: {outcome}."

    @staticmethod
    def _mistral_keyboard(
        rows: list[list[InlineKeyboardButton]],
    ) -> InlineKeyboardMarkup | None:
        if any(
            len(button.callback_data.encode()) > 64
            for row in rows
            for button in row
        ):
            return None
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def _edit_mistral_message(
        self,
        callback: CallbackQuery,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
        acknowledge: bool = True,
    ) -> None:
        if callback.message is None:
            await self._answer_callback(
                callback, "Меню недоступно.", show_alert=True
            )
            return
        try:
            await callback.message.edit_text(text=text, reply_markup=reply_markup)
        except TelegramAPIError:
            if acknowledge:
                await self._answer_callback(
                    callback,
                    "Действие выполнено, но меню не обновлено.",
                    show_alert=True,
                )
            return
        if acknowledge:
            await self._answer_callback(callback, "Готово.")

    @staticmethod
    async def _answer_callback(
        callback: CallbackQuery, text: str, *, show_alert: bool = False
    ) -> None:
        try:
            await callback.answer(text, show_alert=show_alert)
        except TelegramAPIError:
            logger.warning("telegram_callback_answer_failed")

    async def _text_handler(self, message: Message) -> None:
        if not self.authorized(message.from_user.id):
            await message.answer(PRIVATE_REPLY)
            return
        text = message.text
        if text is None or (
            text.startswith("/") and self._mistral_key_input is None
        ):
            return
        if self._captcha_future and not self._captcha_future.done():
            self._captcha_future.set_result(text.strip())
            await message.answer("CAPTCHA input received.")
            return
        if self._edit_future and not self._edit_future.done() and message.text:
            self._edit_future.set_result(message.text.strip())
            return
        key_session = self._mistral_key_input
        if key_session is None:
            reply = self.menu_text(text, message.from_user.id)
            if reply is not None:
                await message.answer(reply, reply_markup=main_menu_keyboard())
            return
        if self.now_factory() >= key_session.expires_at:
            self._mistral_key_input = None
            await message.answer("Время ввода ключа истекло.")
            return
        raw_key = text.strip()
        try:
            await message.delete()
        except TelegramAPIError:
            self._mistral_key_input = None
            await message.answer(
                "Не удалось удалить сообщение. Удалите его вручную и повторите добавление."
            )
            return
        self._mistral_key_input = None
        try:
            result = await self.api_keys.add_key(raw_key)
        except LLMError as exc:
            await message.answer(f"Ключ не добавлен: {exc.category}.")
            return
        await message.answer(f"Ключ {result.id} ····{result.suffix} добавлен.")

    async def _generate_letter(self, callback: CallbackQuery, job_id: str) -> None:
        await callback.answer("Готовлю сопроводительное письмо...")
        vacancy = self.database.get(job_id)
        if vacancy is None or vacancy.status is not VacancyStatus.PENDING_APPROVAL:
            await self.notify("Вакансия уже не ожидает решения.")
            return
        if not vacancy.description.strip():
            await self.notify("Текст вакансии не сохранён, письмо не собрать.")
            return
        if self.cover_letters is None:
            await self.notify("Генерация письма недоступна.")
            return
        letter = await self.cover_letters.generate_cover_letter(
            vacancy.title, vacancy.description
        )
        if not letter.strip():
            await self.notify("Не удалось составить письмо. Нажмите кнопку ещё раз.")
            return
        self.database.update_cover_letter(job_id, letter)
        updated = self.database.get(job_id)
        if callback.message is not None:
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramAPIError:
                logger.warning("telegram_clear_keyboard_failed job_id=%s", job_id)
        if updated is not None:
            await self.send_preview(updated, include_actions=True)

    async def send_preview(self, vacancy: Vacancy, include_actions: bool) -> None:
        keyboard = self._action_keyboard(vacancy) if include_actions else None
        try:
            await self.bot.send_rich_message(
                chat_id=self.settings.tg_user_id,
                rich_message=self._rich_card(vacancy),
                reply_markup=keyboard,
            )
        except TelegramBadRequest:
            logger.warning("telegram_rich_message_fallback job_id=%s", vacancy.id)
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=self._html_card(vacancy),
                parse_mode="HTML",
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )
        logger.info(
            "%s job_id=%s",
            "approval_requested" if include_actions else "preview_sent",
            vacancy.id,
        )

    @staticmethod
    def _action_keyboard(vacancy: Vacancy) -> InlineKeyboardMarkup:
        if vacancy.cover_letter.strip():
            rows = [
                [
                    InlineKeyboardButton(
                        text="Откликнуться", callback_data=f"apply:{vacancy.id}"
                    ),
                    InlineKeyboardButton(
                        text="Пропустить", callback_data=f"skip:{vacancy.id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="✏️ Изменить письмо", callback_data=f"edit:{vacancy.id}"
                    ),
                ],
            ]
        else:
            rows = [
                [
                    InlineKeyboardButton(
                        text="Сгенерировать письмо",
                        callback_data=f"letter:{vacancy.id}",
                    ),
                    InlineKeyboardButton(
                        text="Пропустить", callback_data=f"skip:{vacancy.id}"
                    ),
                ]
            ]
        return InlineKeyboardMarkup(inline_keyboard=rows)

    @staticmethod
    def _clip(text: str, limit: int = CARD_TEXT_LIMIT) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    @staticmethod
    def _confidence(vacancy: Vacancy) -> str:
        return "нет данных" if vacancy.confidence is None else f"{vacancy.confidence:.0%}"

    @staticmethod
    def _rating(vacancy: Vacancy) -> str | None:
        if vacancy.company_rating is None or vacancy.company_reviews_count is None:
            return None
        rating = f"{vacancy.company_rating:.1f}".replace(".", ",")
        return f"★ {rating}/5 · {vacancy.company_reviews_count} отзывов"

    @staticmethod
    def _fit_lines(vacancy: Vacancy) -> list[tuple[str, str]]:
        source = vacancy.fit_summary or (
            FIT_SUMMARY_FALLBACK if vacancy.confidence is not None else ""
        )
        return [
            (category, value)
            for line in source.splitlines()
            for category, separator, value in [line.partition(": ")]
            if separator
        ]

    @classmethod
    def _rich_card(cls, vacancy: Vacancy) -> InputRichMessage:
        company_blocks = [
            InputRichBlockParagraph(
                text=[RichTextBold(text="Компания: "), vacancy.company or "не указана"]
            )
        ]
        if rating := cls._rating(vacancy):
            company_blocks.append(
                InputRichBlockParagraph(
                    text=[RichTextBold(text="Рейтинг HH: "), rating]
                )
            )
        blocks: list[Any] = [
            InputRichBlockSectionHeading(text=vacancy.title, size=2),
            InputRichBlockParagraph(
                text=RichTextUrl(text="Открыть вакансию", url=vacancy.url)
            ),
            InputRichBlockDivider(),
            InputRichBlockDetails(
                summary="Компания", blocks=company_blocks, is_open=True
            ),
        ]
        if vacancy.search_query:
            blocks.append(
                InputRichBlockParagraph(
                    text=[RichTextBold(text="Запрос: "), vacancy.search_query]
                )
            )
        if vacancy.description.strip():
            blocks.append(
                InputRichBlockDetails(
                    summary="Текст вакансии",
                    blocks=[
                        InputRichBlockParagraph(text=cls._clip(vacancy.description))
                    ],
                    is_open=False,
                )
            )
        fit_lines = cls._fit_lines(vacancy)
        if fit_lines:
            blocks.append(InputRichBlockSectionHeading(text="Почему мне подходит", size=3))
            blocks.extend(
                InputRichBlockParagraph(
                    text=[RichTextBold(text=f"{category}: "), value]
                )
                for category, value in fit_lines
            )
        if vacancy.confidence is not None:
            blocks.append(
                InputRichBlockParagraph(
                    text=[RichTextBold(text="Уверенность: "), cls._confidence(vacancy)]
                )
            )
        if vacancy.cover_letter.strip():
            blocks.append(
                InputRichBlockDetails(
                    summary="Сопроводительное письмо",
                    blocks=[InputRichBlockParagraph(text=vacancy.cover_letter)],
                    is_open=False,
                )
            )
        return InputRichMessage(skip_entity_detection=True, blocks=blocks)

    @classmethod
    def _html_card(cls, vacancy: Vacancy) -> str:
        rating = cls._rating(vacancy)
        company = html.escape(vacancy.company or "не указана")
        company_text = f"<b>Компания</b>\nКомпания: {company}"
        if rating:
            company_text += f"\nРейтинг HH: {rating}"
        parts = [
            f"<b>{html.escape(vacancy.title)}</b>",
            f"<a href=\"{html.escape(vacancy.url, quote=True)}\">Открыть вакансию</a>",
            "",
            company_text,
        ]
        if vacancy.search_query:
            parts.extend(["", f"<b>Запрос:</b> {html.escape(vacancy.search_query)}"])
        description = cls._clip(vacancy.description)
        if description:
            parts.extend(["", f"<b>Текст вакансии</b>\n{html.escape(description)}"])
        fit_lines = cls._fit_lines(vacancy)
        if fit_lines:
            fit_text = "\n".join(
                f"<b>{html.escape(category)}:</b> {html.escape(value)}"
                for category, value in fit_lines
            )
            parts.extend(["", f"<b>Почему мне подходит</b>\n{fit_text}"])
        if vacancy.confidence is not None:
            parts.append(f"Уверенность: {cls._confidence(vacancy)}")
        if vacancy.cover_letter.strip():
            parts.extend(
                ["", f"<b>Сопроводительное письмо</b>\n{html.escape(vacancy.cover_letter)}"]
            )
        text = "\n".join(parts)
        if len(text) <= 4096:
            return text
        return text[:4095] + "…"

    async def notify(self, text: str) -> None:
        try:
            await self.bot.send_message(chat_id=self.settings.tg_user_id, text=text)
        except Exception as exc:
            logger.warning("telegram_notify_failed error=%s", exc)

    async def notify_analysis_failed(self, title: str, url: str, error_type: str) -> None:
        text = (
            f"⚠️ <b>Ошибка AI-анализа вакансии</b>\n"
            f"<b>{html.escape(title)}</b>\n"
            f"<a href=\"{html.escape(url, quote=True)}\">Открыть вакансию</a>\n\n"
            f"Причина: <code>{html.escape(error_type)}</code>"
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_analysis_failed_error error=%s", exc)

    async def notify_questionnaire_required(self, title: str, url: str) -> None:
        text = (
            f"📋 <b>Требуется ручной отклик (тестовое / анкета)</b>\n"
            f"<b>{html.escape(title)}</b>\n"
            f"<a href=\"{html.escape(url, quote=True)}\">Открыть вакансию на HH.ru</a>\n\n"
            f"Работодатель требует заполнении анкеты или выполнение тестового задания."
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_questionnaire_failed error=%s", exc)

    async def request_captcha(
        self, screenshot: Path, title: str, timeout_seconds: int
    ) -> str | None:
        self._mistral_key_input = None
        loop = asyncio.get_running_loop()
        self._captcha_future = loop.create_future()
        await self.bot.send_photo(
            chat_id=self.settings.tg_user_id,
            photo=FSInputFile(screenshot),
            caption=f"CAPTCHA detected for: {title}\nReply with the text or use /cancel.",
        )
        try:
            return await asyncio.wait_for(self._captcha_future, timeout_seconds)
        except TimeoutError:
            return None
        finally:
            self._captcha_future = None

    async def check_updates(self, notify: bool = True) -> ReleaseInfo | None:
        try:
            release = await check_github_release(current_version=__version__)
        except Exception as exc:
            logger.debug("update_check_failed error=%s", exc)
            return None
        if release is not None:
            self.latest_release = release
            if notify and self._notified_release_tag != release.tag_name:
                self._notified_release_tag = release.tag_name
                await self.notify_update_available(release)
        return release

    async def notify_update_available(self, release: ReleaseInfo) -> None:
        text = (
            f"🚀 <b>Доступно обновление HH Agent {html.escape(release.tag_name)}!</b>\n"
            f"Текущая версия: <code>v{__version__}</code>\n\n"
            f"<a href=\"{html.escape(release.html_url, quote=True)}\">Посмотреть список изменений на GitHub</a>"
        )
        try:
            await self.bot.send_message(
                chat_id=self.settings.tg_user_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning("notify_update_available_failed error=%s", exc)

    async def start_polling(self) -> None:
        while True:
            try:
                await self.dispatcher.start_polling(self.bot)
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("telegram_polling_error error=%s, retrying in 5s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        session = getattr(self.bot, "session", None)
        if session is not None:
            await session.close()
