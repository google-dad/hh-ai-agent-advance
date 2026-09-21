import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from updater import ReleaseInfo, check_github_release
from version import __version__, is_newer_version, parse_version


def test_parse_version_handles_various_formats() -> None:
    assert parse_version("1.2.5") == (1, 2, 5)
    assert parse_version("v1.2.6") == (1, 2, 6)
    assert parse_version("v2.0.0-rc1") == (2, 0, 0, 1)
    assert parse_version("0.1") == (0, 1)
    assert parse_version("invalid") == (0,)


def test_is_newer_version() -> None:
    assert is_newer_version("v1.2.6", "1.2.5") is True
    assert is_newer_version("1.3.0", "1.2.6") is True
    assert is_newer_version("2.0.0", "1.9.9") is True
    assert is_newer_version("1.2.5", "1.2.5") is False
    assert is_newer_version("v1.2.4", "1.2.5") is False
    assert is_newer_version("1.2.0", "1.2.5") is False


@pytest.mark.asyncio
async def test_check_github_release_finds_newer_version() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "tag_name": "v1.3.0",
            "name": "Release 1.3.0",
            "html_url": "https://github.com/fikstt2/hh-ai-agent/releases/tag/v1.3.0",
            "published_at": "2026-08-13T12:00:00Z",
            "body": "New awesome features",
        }
    )

    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get

    release = await check_github_release(
        current_version="1.2.6", session=mock_session
    )

    assert release is not None
    assert release.tag_name == "v1.3.0"
    assert release.name == "Release 1.3.0"
    assert "v1.3.0" in release.html_url


@pytest.mark.asyncio
async def test_check_github_release_returns_none_when_up_to_date() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "tag_name": "v1.2.5",
            "name": "Release 1.2.5",
            "html_url": "https://github.com/fikstt2/hh-ai-agent/releases/tag/v1.2.5",
            "published_at": "2026-08-13T12:00:00Z",
            "body": "Previous release",
        }
    )

    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get

    release = await check_github_release(
        current_version="1.2.6", session=mock_session
    )

    assert release is None


@pytest.mark.asyncio
async def test_check_github_release_handles_network_error_gracefully() -> None:
    import aiohttp

    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection failed"))
    mock_get.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get

    release = await check_github_release(
        current_version="1.2.6", session=mock_session
    )

    assert release is None


@pytest.mark.asyncio
async def test_telegram_check_updates_sends_notification_once(tmp_path, monkeypatch) -> None:
    from approval import ApprovalGuard, ApprovalService
    from config import load_settings
    from database import Database
    from tests.test_config import VALID_ENV, write_profile
    from tg_bot import AgentControl, TelegramService

    settings = load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV)
    database = Database(tmp_path / "test.db")
    database.init()
    guard = ApprovalGuard(settings, database)
    hh_client = MagicMock()
    approval_service = ApprovalService(settings, database, hh_client)
    control = AgentControl()

    fake_bot = MagicMock()
    fake_bot.send_message = AsyncMock()

    telegram = TelegramService(
        settings, database, approval_service, control, bot=fake_bot
    )

    fake_release = ReleaseInfo(
        tag_name="v1.3.0",
        name="Release 1.3.0",
        html_url="https://github.com/fikstt2/hh-ai-agent/releases/tag/v1.3.0",
        published_at="2026-08-13T12:00:00Z",
        body="Release notes",
    )

    async def fake_check_release(*_args, **_kwargs):
        return fake_release

    monkeypatch.setattr("tg_bot.check_github_release", fake_check_release)

    # First check: notifies
    res1 = await telegram.check_updates(notify=True)
    assert res1 == fake_release
    assert fake_bot.send_message.call_count == 1
    call_args = fake_bot.send_message.call_args[1]
    assert "v1.3.0" in call_args["text"]
    assert "https://github.com/fikstt2/hh-ai-agent/releases/tag/v1.3.0" in call_args["text"]

    # Second check with same release: deduplicated, no new notification
    res2 = await telegram.check_updates(notify=True)
    assert res2 == fake_release
    assert fake_bot.send_message.call_count == 1

    # Diagnostics contains update info
    diag = telegram.command("diagnostics", settings.tg_user_id)
    assert "доступно: v1.3.0" in diag

    # Status contains update info
    status = telegram.command("status", settings.tg_user_id)
    assert "доступно обновление v1.3.0" in status
