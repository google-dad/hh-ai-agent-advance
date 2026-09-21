import asyncio
import inspect
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from aiogram.exceptions import TelegramNetworkError

from approval import ApprovalService
from config import load_settings
from database import Database
from llm.errors import LLMConfigurationError
from llm.mistral_keys import MistralKeyCheckResult, MistralKeyView
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl, TelegramService


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
RAW_KEY = "mistral-super-secret"


@dataclass
class SafeCall:
    name: str
    value: object = field(default=None, repr=False)


def key_view(
    key_id: int,
    suffix: str,
    *,
    status: str = "ready",
    current: bool = False,
    cooldown_until: datetime | None = None,
) -> MistralKeyView:
    return MistralKeyView(
        id=key_id,
        suffix=suffix,
        status=status,
        cooldown_until=cooldown_until,
        last_checked_at=None,
        last_error_type="",
        is_current=current,
    )


class FakeMistralKeys:
    def __init__(
        self,
        views: tuple[MistralKeyView, ...] = (),
        *,
        provider: str = "mistral",
        add_error: Exception | None = None,
        check_error: Exception | None = None,
        delete_error: Exception | None = None,
        delete_result: bool = True,
    ) -> None:
        self.views = list(views)
        self.provider = provider
        self.calls: list[SafeCall] = []
        self.add_error = add_error
        self.check_error = check_error
        self.delete_error = delete_error
        self.delete_result = delete_result

    def list_keys(self) -> tuple[MistralKeyView, ...]:
        self.calls.append(SafeCall("list_keys"))
        return tuple(self.views)

    async def add_key(self, raw: str) -> MistralKeyView:
        self.calls.append(SafeCall("add_key", raw))
        if self.add_error:
            raise self.add_error
        view = key_view(max((item.id for item in self.views), default=0) + 1, raw[-4:])
        self.views.append(view)
        return view

    async def check_key(self, key_id: int) -> MistralKeyCheckResult | None:
        self.calls.append(SafeCall("check_key", key_id))
        if self.check_error:
            raise self.check_error
        view = next((item for item in self.views if item.id == key_id), None)
        return (
            None
            if view is None
            else MistralKeyCheckResult(view, success=True, error_type="")
        )

    async def check_all(self) -> tuple[MistralKeyCheckResult, ...]:
        self.calls.append(SafeCall("check_all"))
        return tuple(
            MistralKeyCheckResult(item, success=True, error_type="")
            for item in self.views
        )

    async def delete_key(self, key_id: int) -> bool:
        self.calls.append(SafeCall("delete_key", key_id))
        if self.delete_error:
            raise self.delete_error
        if self.delete_result:
            self.views = [item for item in self.views if item.id != key_id]
        return self.delete_result


class FakeMessage:
    def __init__(
        self,
        text: str,
        *,
        user_id: int = 42,
        delete_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.from_user = SimpleNamespace(id=user_id)
        self.answers: list[dict] = []
        self.deleted = False
        self.delete_error = delete_error

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append({"text": text, **kwargs})

    async def delete(self) -> None:
        if self.delete_error:
            raise self.delete_error
        self.deleted = True


class FakeCallbackMessage:
    def __init__(self, edit_error: Exception | None = None) -> None:
        self.edits: list[dict] = []
        self.edit_error = edit_error

    async def edit_text(self, text: str, **kwargs) -> None:
        if self.edit_error:
            raise self.edit_error
        self.edits.append({"text": text, **kwargs})


class FakeCallback:
    def __init__(
        self,
        data: str,
        *,
        user_id: int = 42,
        message: FakeCallbackMessage | None = None,
    ) -> None:
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = message or FakeCallbackMessage()
        self.answers: list[dict] = []

    async def answer(self, text: str, *, show_alert: bool = False) -> None:
        self.answers.append({"text": text, "show_alert": show_alert})


class FakeBot:
    def __init__(self) -> None:
        self.photo_sent = asyncio.Event()

    async def send_photo(self, **_kwargs) -> None:
        self.photo_sent.set()


class FakeApprovalService:
    async def approve_and_apply(self, *_args):
        raise AssertionError("physical application must not run")

    def skip(self, *_args):
        raise AssertionError("vacancy mutation must not run")


def service(
    tmp_path: Path,
    *,
    views: tuple[MistralKeyView, ...] = (),
    keys: FakeMistralKeys | None = None,
) -> tuple[TelegramService, FakeMistralKeys, FakeBot]:
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "TG_USER_ID": "42"},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    manager = keys or FakeMistralKeys(views)
    bot = FakeBot()

    async def no_rewrite(*_args) -> str:
        return ""

    kwargs = {
        "api_keys": manager,
        "bot": bot,
        "now_factory": lambda: NOW,
    }
    if "validate_cover_letter" in inspect.signature(TelegramService).parameters:
        kwargs.update(
            validate_cover_letter=lambda text: text,
            rewrite_cover_letter=no_rewrite,
        )
    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        AgentControl(),
        **kwargs,
    )
    return telegram, manager, bot


def callback_data(answer: dict) -> list[str]:
    markup = answer["reply_markup"]
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
    ]


def test_foreign_user_cannot_open_or_mutate_mistral_keys(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    message = FakeMessage("/mistral_keys", user_id=99)

    asyncio.run(telegram._command_handler(message))
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add", user_id=99)))

    assert message.answers == [{"text": "This bot is private."}]
    assert keys.calls == []


def test_list_masks_keys_and_uses_only_numeric_callback_ids(tmp_path: Path) -> None:
    telegram, _, _ = service(tmp_path, views=(key_view(1, "aB7x", current=True),))
    message = FakeMessage("/mistral_keys")

    asyncio.run(telegram._command_handler(message))

    rendered = repr(message.answers)
    assert "····aB7x" in rendered
    assert "mk:check:1" in rendered
    assert "mk:delete:1" in rendered
    assert RAW_KEY not in rendered
    assert all(
        part.isdecimal()
        for data in callback_data(message.answers[-1])
        for part in data.split(":")[2:]
    )


def test_add_is_blocked_by_captcha_or_edit_session(tmp_path: Path) -> None:
    async def exercise() -> None:
        telegram, keys, _ = service(tmp_path)
        telegram._captcha_future = asyncio.get_running_loop().create_future()
        captcha_callback = FakeCallback("mk:add")

        await telegram._callback_handler(captcha_callback)

        assert telegram._mistral_key_input is None
        assert "CAPTCHA" in captcha_callback.answers[-1]["text"]
        telegram._captcha_future.set_result(None)
        telegram._edit_future = asyncio.get_running_loop().create_future()
        edit_callback = FakeCallback("mk:add")

        await telegram._callback_handler(edit_callback)

        assert telegram._mistral_key_input is None
        assert "редактир" in edit_callback.answers[-1]["text"].lower()
        assert keys.calls == []
        telegram._edit_future.cancel()

    asyncio.run(exercise())


def test_secret_message_is_deleted_before_key_is_added(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))
    message = FakeMessage(RAW_KEY)

    asyncio.run(telegram._text_handler(message))

    assert message.deleted
    assert [call.name for call in keys.calls] == ["add_key"]
    assert telegram._mistral_key_input is None
    assert "····cret" in message.answers[-1]["text"]
    assert RAW_KEY not in repr(message.answers)
    assert RAW_KEY not in repr(keys.calls)


def test_secret_starting_with_slash_is_still_deleted(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))
    message = FakeMessage("/mistral-super-secret")

    asyncio.run(telegram._text_handler(message))

    assert message.deleted
    assert [call.name for call in keys.calls] == ["add_key"]


def test_failed_message_delete_never_adds_key(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))
    message = FakeMessage(
        RAW_KEY,
        delete_error=TelegramNetworkError(method=object(), message="delete failed"),
    )

    asyncio.run(telegram._text_handler(message))

    assert keys.calls == []
    assert telegram._mistral_key_input is None
    assert message.answers == [
        {
            "text": "Не удалось удалить сообщение. Удалите его вручную и повторите добавление."
        }
    ]


def test_add_error_exposes_only_safe_category(tmp_path: Path, caplog) -> None:
    keys = FakeMistralKeys(add_error=LLMConfigurationError(RAW_KEY))
    telegram, _, _ = service(tmp_path, keys=keys)
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))
    message = FakeMessage(RAW_KEY)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(telegram._text_handler(message))

    assert message.answers == [{"text": "Ключ не добавлен: configuration."}]
    assert RAW_KEY not in caplog.text
    assert RAW_KEY not in repr(message.answers)
    assert RAW_KEY not in repr(keys.calls)


def test_cancel_and_expiry_discard_key_input(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))

    cancel = FakeMessage("/cancel")
    asyncio.run(telegram._command_handler(cancel))

    assert telegram._mistral_key_input is None
    assert "cancel" in cancel.answers[-1]["text"].lower()
    asyncio.run(telegram._callback_handler(FakeCallback("mk:add")))
    telegram._mistral_key_input.expires_at = NOW - timedelta(seconds=1)
    expired = FakeMessage(RAW_KEY)

    asyncio.run(telegram._text_handler(expired))

    assert not expired.deleted
    assert keys.calls == []
    assert telegram._mistral_key_input is None
    assert "истек" in expired.answers[-1]["text"].lower()


def test_request_captcha_cancels_key_input(tmp_path: Path) -> None:
    async def exercise() -> None:
        telegram, _, bot = service(tmp_path)
        await telegram._callback_handler(FakeCallback("mk:add"))

        task = asyncio.create_task(
            telegram.request_captcha(tmp_path / "captcha.png", "Vacancy", 60)
        )
        await bot.photo_sent.wait()

        assert telegram._mistral_key_input is None
        telegram._captcha_future.set_result(None)
        assert await task is None

    asyncio.run(exercise())


def test_check_one_and_check_all_use_manager_api(tmp_path: Path) -> None:
    telegram, keys, _ = service(
        tmp_path, views=(key_view(1, "one1"), key_view(2, "two2"))
    )

    one = FakeCallback("mk:check:2")
    asyncio.run(telegram._callback_handler(one))
    all_keys = FakeCallback("mk:check")
    asyncio.run(telegram._callback_handler(all_keys))

    assert [(call.name, call.value) for call in keys.calls] == [
        ("check_key", 2),
        ("check_all", None),
    ]
    assert "····two2" in one.message.edits[-1]["text"]
    assert "2" in all_keys.message.edits[-1]["text"]


def test_key_checks_acknowledge_callback_before_network_work(tmp_path: Path) -> None:
    order: list[str] = []

    class OrderedKeys(FakeMistralKeys):
        async def check_key(self, key_id: int):
            order.append("check_key")
            return await super().check_key(key_id)

        async def check_all(self):
            order.append("check_all")
            return await super().check_all()

    class OrderedCallback(FakeCallback):
        async def answer(self, text: str, *, show_alert: bool = False) -> None:
            order.append("ack")
            await super().answer(text, show_alert=show_alert)

    keys = OrderedKeys((key_view(2, "aB7x"),))
    telegram, _, _ = service(tmp_path, keys=keys)

    asyncio.run(telegram._callback_handler(OrderedCallback("mk:check:2")))
    assert order[:2] == ["ack", "check_key"]
    order.clear()
    asyncio.run(telegram._callback_handler(OrderedCallback("mk:check")))
    assert order[:2] == ["ack", "check_all"]


def test_check_error_exposes_only_safe_category(tmp_path: Path, caplog) -> None:
    keys = FakeMistralKeys(
        (key_view(1, "one1"),), check_error=LLMConfigurationError(RAW_KEY)
    )
    telegram, _, _ = service(tmp_path, keys=keys)
    callback = FakeCallback("mk:check:1")

    with caplog.at_level(logging.DEBUG):
        asyncio.run(telegram._callback_handler(callback))

    rendered = repr((callback.answers, callback.message.edits, keys.calls))
    assert "configuration" in rendered
    assert RAW_KEY not in rendered
    assert RAW_KEY not in caplog.text


def test_delete_requires_confirmation_and_numeric_id(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path, views=(key_view(7, "key7"),))
    preview = FakeCallback("mk:delete:7")

    asyncio.run(telegram._callback_handler(preview))

    assert keys.calls == []
    assert "mk:confirm:7" in repr(preview.message.edits)
    confirm = FakeCallback("mk:confirm:7")
    asyncio.run(telegram._callback_handler(confirm))

    assert [(call.name, call.value) for call in keys.calls] == [("delete_key", 7)]
    assert "удал" in repr((confirm.answers, confirm.message.edits)).lower()
    for data in ("mk:check:", "mk:delete:nope", "mk:confirm:1:2"):
        invalid = FakeCallback(data)
        asyncio.run(telegram._callback_handler(invalid))
        assert invalid.answers[-1]["show_alert"]
    assert len(keys.calls) == 1


def test_callback_length_guard_precedes_mistral_routing(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path)
    callback = FakeCallback("mk:check:" + "1" * 60)

    asyncio.run(telegram._callback_handler(callback))

    assert callback.answers[-1]["show_alert"]
    assert keys.calls == []


def test_unavailable_manager_is_safe(tmp_path: Path) -> None:
    telegram, _, _ = service(tmp_path)
    telegram.mistral_keys = None
    telegram.api_keys = None
    command = FakeMessage("/mistral_keys")
    callback = FakeCallback("mk:add")

    asyncio.run(telegram._command_handler(command))
    asyncio.run(telegram._callback_handler(callback))

    assert "недоступ" in repr((command.answers, callback.answers)).lower()


def test_keys_command_alias_opens_menu(tmp_path: Path) -> None:
    telegram, _, _ = service(tmp_path, views=(key_view(1, "ab12"),))
    message = FakeMessage("/keys")

    asyncio.run(telegram._command_handler(message))

    assert message.answers
    assert "openai_compatible" not in message.answers[0]["text"]
    assert "Ключи LLM (mistral):" in message.answers[0]["text"]


def test_keys_menu_shows_openai_compatible_provider(tmp_path: Path) -> None:
    keys = FakeMistralKeys((key_view(1, "ab12"),), provider="openai_compatible")
    telegram, _, _ = service(tmp_path, keys=keys)
    message = FakeMessage("/keys")

    asyncio.run(telegram._command_handler(message))

    assert "Ключи LLM (openai_compatible):" in message.answers[0]["text"]


def test_successful_delete_survives_telegram_edit_error(tmp_path: Path) -> None:
    telegram, keys, _ = service(tmp_path, views=(key_view(9, "key9"),))
    callback = FakeCallback(
        "mk:confirm:9",
        message=FakeCallbackMessage(
            TelegramNetworkError(method=object(), message="edit failed")
        ),
    )

    asyncio.run(telegram._callback_handler(callback))

    assert [(call.name, call.value) for call in keys.calls] == [("delete_key", 9)]
    assert callback.answers


def test_delete_error_exposes_only_safe_category(tmp_path: Path, caplog) -> None:
    keys = FakeMistralKeys(
        (key_view(9, "key9"),), delete_error=LLMConfigurationError(RAW_KEY)
    )
    telegram, _, _ = service(tmp_path, keys=keys)
    callback = FakeCallback("mk:confirm:9")

    with caplog.at_level(logging.DEBUG):
        asyncio.run(telegram._callback_handler(callback))

    rendered = repr((callback.answers, callback.message.edits, keys.calls))
    assert "configuration" in rendered
    assert RAW_KEY not in rendered
    assert RAW_KEY not in caplog.text
