import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendRichMessage
from aiogram.types import InputRichBlockDetails, InputRichMessage

from config import load_settings
from database import Database
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl, TelegramService


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


class FakeBot:
    def __init__(self, rich_error: Exception | None = None):
        self.messages: list[dict] = []
        self.rich_messages: list[dict] = []
        self.rich_error = rich_error

    async def send_rich_message(self, **kwargs):
        if self.rich_error:
            raise self.rich_error
        self.rich_messages.append(kwargs)

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeApprovalService:
    async def approve_and_apply(self, job_id: str, user_id: int):
        raise AssertionError("approval must not be called by command tests")

    def skip(self, job_id: str, user_id: int):
        raise AssertionError("skip must not be called by command tests")


def service(
    tmp_path: Path,
    app_mode: str = "dry_run",
    rich_error: Exception | None = None,
):
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "TG_USER_ID": "42", "APP_MODE": app_mode},
    )
    settings = replace(settings, database_path=tmp_path / "agent.db")
    database = Database(settings.database_path)
    database.init()
    control = AgentControl()
    bot = FakeBot(rich_error)
    telegram = TelegramService(
        settings,
        database,
        FakeApprovalService(),
        control,
        bot=bot,
        now_factory=lambda: NOW,
    )
    return telegram, database, control, bot


def add_preview_vacancy(database: Database) -> None:
    assert database.discover(
        job_id="job-1",
        title="Python <Developer>",
        company="Example & Co",
        company_url="https://hh.ru/employer/123",
        url="https://example.com/vacancy/job-1",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.store_company_details("job-1", rating=4.7, reviews_count=128)
    assert database.request_approval(
        job_id="job-1",
        cover_letter="Hello <team>",
        llm_decision=True,
        llm_reason="Relevant & local",
        fit_summary="Опыт: backend-сервисы.\nНавыки: Python.",
        confidence=0.87,
        now=NOW,
    )


def add_search_run(database: Database) -> None:
    database.save_search_run(
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=12),
        state="completed",
        query_count=2,
        found_results=9,
        new_vacancies=4,
        duplicates=5,
        rejected_by_filter=2,
        rejected_by_llm=2,
        telegram_cards=0,
        error_count=1,
        rejection_reasons={"title": 2},
        error_reasons={"network_error": 1},
        last_safe_error="network_error",
        circuit_reason="page_structure_changed",
    )


def test_foreign_user_cannot_pause_or_read_configuration(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path)

    reply = telegram.command("pause", user_id=99)

    assert reply == "This bot is private."
    assert not control.paused
    assert "42" not in reply
    assert "TG_USER_ID" not in reply


def test_pause_resume_and_status_use_live_state(tmp_path: Path) -> None:
    telegram, _, control, _ = service(tmp_path)

    assert telegram.command("pause", user_id=42) == "Agent paused."
    assert control.paused
    status = telegram.command("status", user_id=42)
    assert "mode: dry_run" in status
    assert "state: paused" in status
    assert telegram.command("resume", user_id=42) == "Agent resumed."
    assert not control.paused
    assert control.wake_event.is_set()


def test_diagnostics_shows_last_search_run(tmp_path: Path) -> None:
    telegram, database, control, _ = service(tmp_path)
    add_search_run(database)
    control.circuit_reason = "technical_failure_ratio"
    control.next_run_at = NOW + timedelta(minutes=30)

    reply = telegram.command("diagnostics", user_id=42)

    assert "длительность: 12 с" in reply
    assert "запросы: 2" in reply
    assert "найдено: 9" in reply
    assert "новые: 4" in reply
    assert "дубли: 5" in reply
    assert "фильтр: 2" in reply
    assert "LLM: 2" in reply
    assert "title=2" in reply
    assert "карточки: 0" in reply
    assert "ошибки: 1" in reply
    assert "network_error" in reply
    assert "breaker: open (technical_failure_ratio)" in reply
    assert "следующий запуск:" in reply


def test_resume_clears_runtime_breaker_but_keeps_saved_diagnostics(
    tmp_path: Path,
) -> None:
    telegram, database, control, _ = service(tmp_path)
    add_search_run(database)
    control.paused = True
    control.circuit_reason = "page_structure_changed"
    control.consecutive_search_errors = 3
    control.next_run_at = NOW + timedelta(minutes=30)

    assert telegram.command("resume", 42) == "Agent resumed."
    assert not control.paused
    assert control.circuit_reason == ""
    assert control.consecutive_search_errors == 0
    assert control.next_run_at is None
    assert control.wake_event.is_set()
    assert database.latest_search_run().circuit_reason == "page_structure_changed"
    diagnostics = telegram.command("diagnostics", 42)
    assert "breaker: closed" in diagnostics
    assert "last breaker: page_structure_changed" in diagnostics


def test_dry_run_preview_uses_rich_card_with_company_details(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "dry_run")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=False))

    assert bot.messages == []
    message = bot.rich_messages[0]
    assert message["reply_markup"] is None
    details = next(
        block
        for block in message["rich_message"].blocks
        if isinstance(block, InputRichBlockDetails)
        and block.summary == "Компания"
    )
    rendered = repr(details.blocks)
    assert "Example & Co" in rendered
    assert "★ 4,7/5 · 128 отзывов" in rendered
    assert "Опыт" in repr(message["rich_message"].blocks)


def rich_error(error_type: type[Exception], message: str) -> Exception:
    method = SendRichMessage(
        chat_id=42,
        rich_message=InputRichMessage(html="test"),
    )
    return error_type(method=method, message=message)


def test_unsupported_rich_message_falls_back_to_escaped_html(tmp_path: Path) -> None:
    telegram, database, _, bot = service(
        tmp_path,
        rich_error=rich_error(TelegramBadRequest, "method is not supported"),
    )
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=False))

    assert len(bot.messages) == 1
    assert "Python &lt;Developer&gt;" in bot.messages[0]["text"]
    assert "Example &amp; Co" in bot.messages[0]["text"]
    assert "<b>Опыт:</b> backend-сервисы." in bot.messages[0]["text"]


def test_transport_error_does_not_send_duplicate_fallback(tmp_path: Path) -> None:
    telegram, database, _, bot = service(
        tmp_path,
        rich_error=rich_error(TelegramNetworkError, "network disconnected"),
    )
    add_preview_vacancy(database)

    with pytest.raises(TelegramNetworkError):
        asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=False))

    assert bot.messages == []


def test_approval_preview_has_apply_and_skip_buttons(tmp_path: Path) -> None:
    telegram, database, _, bot = service(tmp_path, "approval")
    add_preview_vacancy(database)

    asyncio.run(telegram.send_preview(database.get("job-1"), include_actions=True))

    buttons = bot.rich_messages[0]["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == ["Откликнуться", "Пропустить"]
    assert [button.callback_data for button in buttons] == ["apply:job-1", "skip:job-1"]


def test_pending_and_stats_read_sqlite(tmp_path: Path) -> None:
    telegram, database, _, _ = service(tmp_path, "approval")
    add_preview_vacancy(database)

    assert "job-1" in telegram.command("pending", user_id=42)
    assert "pending_approval: 1" in telegram.command("stats", user_id=42)
