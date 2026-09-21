import asyncio
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import main as main_module
from ai_analyzer import AnalysisError, SuitabilityResult, VacancyAnalyzer
from config import load_settings
from database import Database, VacancyStatus
from hh_client import (
    CompanyDetails,
    PageState,
    VacancyDetails,
    VacancySearchResult,
    VacancySummary,
)
from llm.errors import LLMAuthenticationError, LLMTimeoutError
from llm.managed import ManagedLLMProvider
from llm.providers.fake import FakeProvider
from llm.types import LLMResponse
from main import (
    VacancyProcessResult,
    SearchRunStats,
    agent_loop,
    process_vacancy,
    run_search_cycle,
)
from tests.test_config import VALID_ENV, write_profile
from tg_bot import AgentControl


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)
SUMMARY = VacancySummary(
    "job-1", "Python developer", "https://example.com/vacancy/job-1", "Python"
)


class FakeHHClient:
    def __init__(
        self,
        details: VacancyDetails,
        search_results: list[VacancySearchResult] | None = None,
    ):
        self.details = details
        self.search_results = search_results or []
        self.read_calls = 0
        self.search_calls = 0
        self.message_checks = 0

    async def read_vacancy(self, summary, captcha_solver=None):
        self.read_calls += 1
        return self.details

    async def search_vacancies(self, query, areas, experience_filters):
        self.search_calls += 1
        return (
            self.search_results.pop(0)
            if self.search_results
            else VacancySearchResult([], 0, 0)
        )

    async def check_messages(self, notify):
        self.message_checks += 1

    async def read_company_details(self, company_url: str):
        return CompanyDetails(4.7, 128)


class SearchReached(Exception):
    pass


class StopAtSearchHHClient(FakeHHClient):
    async def search_vacancies(self, *_args):
        raise SearchReached


class FakeAnalyzer:
    def __init__(self):
        self.assess_calls = 0

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        self.assess_calls += 1
        return SuitabilityResult(
            suitable=True,
            confidence=0.91,
            reason="Relevant work",
            fit_points=[{"category": "Навыки", "text": "Python"}],
        )

    async def generate_cover_letter(self, title: str, description: str) -> str:
        return "Safe local-profile letter"


class SequenceAnalyzer(FakeAnalyzer):
    def __init__(self, outcomes: list[SuitabilityResult | Exception]):
        self.outcomes = iter(outcomes)

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PausingAnalyzer(FakeAnalyzer):
    def __init__(self, control: AgentControl):
        super().__init__()
        self.control = control
        self.calls = 0

    async def assess(self, title: str, description: str) -> SuitabilityResult:
        self.calls += 1
        self.control.paused = True
        return await super().assess(title, description)


class FakeTelegram:
    def __init__(self):
        self.previews: list[tuple[str, bool]] = []
        self.notifications: list[str] = []

    async def send_preview(self, vacancy, include_actions: bool):
        self.previews.append((vacancy.id, include_actions))

    async def notify(self, text: str):
        self.notifications.append(text)

    async def notify_analysis_failed(self, title: str, url: str, error_type: str):
        self.notifications.append(f"analysis_failed:{error_type}")

    async def request_captcha(self, screenshot, title: str, timeout_seconds: int):
        return None


def settings(tmp_path: Path, mode: str):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, "TG_USER_ID": "42", "APP_MODE": mode},
    )
    return replace(loaded, database_path=tmp_path / "agent.db")


def test_run_shares_mistral_manager_and_assigns_telegram_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_settings = settings(tmp_path, "approval")
    captured: dict[str, object] = {}

    class FakeBackend:
        async def start(self):
            return object()

        async def close(self):
            return None

    class FakeMistralManager:
        def __init__(self):
            self.notifier = None

        def set_notifier(self, notifier):
            self.notifier = notifier

        async def close(self):
            return None

    manager = FakeMistralManager()

    class FakeHHClient:
        def __init__(self, *_args):
            pass

        async def ensure_login(self) -> bool:
            return True

    class FakeAnalyzer:
        def __init__(self, _settings, provider):
            captured["analyzer_provider"] = provider

    class FakeTelegramService:
        def __init__(
            self,
            _settings,
            _database,
            _approval_service,
            _control,
            *,
            api_keys=None,
            mistral_keys=None,
            **_kwargs,
        ):
            captured["telegram"] = self
            captured["telegram_manager"] = (
                api_keys if api_keys is not None else mistral_keys
            )

        async def notify(self, _text: str) -> None:
            return None

        async def check_updates(self, notify: bool = True) -> None:
            return None

        async def start_polling(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def fake_agent_loop(*_args) -> None:
        return None

    monkeypatch.setattr(main_module, "ApiKeyManager", FakeMistralManager, raising=False)
    monkeypatch.setattr(main_module, "create_browser_backend", lambda _settings: FakeBackend())
    monkeypatch.setattr(main_module, "create_llm_provider", lambda *_args: manager)
    monkeypatch.setattr(main_module, "HHClient", FakeHHClient)
    monkeypatch.setattr(main_module, "VacancyAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "TelegramService", FakeTelegramService)
    monkeypatch.setattr(main_module, "agent_loop", fake_agent_loop)

    asyncio.run(main_module.run(app_settings))

    telegram = captured["telegram"]
    assert captured["analyzer_provider"] is manager
    assert captured["telegram_manager"] is manager
    assert manager.notifier == telegram.notify


def test_dry_run_records_and_previews_without_pending_actions(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.DISCOVERED
    assert vacancy.cover_letter == "Safe local-profile letter"
    assert vacancy.fit_summary.startswith("Навыки: Python")
    assert vacancy.company_rating == 4.7
    assert telegram.previews == [("job-1", False)]


def test_approval_mode_records_pending_and_sends_actions(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [("job-1", True)]


def test_browser_read_error_is_persisted_as_apply_failed(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "dry_run")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.NETWORK_ERROR, error="navigation failed"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            FakeAnalyzer(),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "navigation failed" in vacancy.error_text


def test_llm_failure_records_analysis_failure_without_requesting_approval(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    provider = ManagedLLMProvider(
        FakeProvider([LLMAuthenticationError()]),
        database,
        max_retries=1,
        max_requests_per_day=100,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            VacancyAnalyzer(app_settings, provider),
            telegram,
            now_factory=lambda: NOW,
        )
    )

    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.ANALYSIS_FAILED
    assert vacancy.llm_decision is None
    assert vacancy.error_text == "authentication"
    assert vacancy.analysis_retry_count == 1
    assert vacancy.analysis_next_retry_at == (
        NOW + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()
    assert vacancy.cover_letter == ""
    assert telegram.previews == []
    assert telegram.notifications == ["analysis_failed:authentication"]


def test_analysis_retry_delay_starts_when_failed_analysis_finishes(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    times = iter((NOW, NOW + timedelta(minutes=5)))

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            SequenceAnalyzer([AnalysisError("timeout")]),
            telegram,
            now_factory=lambda: next(times),
        )
    )

    assert database.get(SUMMARY.id).analysis_next_retry_at == (
        NOW
        + timedelta(minutes=5)
        + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()


def test_failed_retry_delay_starts_when_retry_finishes(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    assert database.mark_analysis_failed(
        SUMMARY.id,
        error_type="timeout",
        now=NOW,
        retry_after=timedelta(minutes=30),
        max_attempts=3,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    times = iter(
        (
            NOW + timedelta(minutes=30),
            NOW + timedelta(minutes=30),
            NOW + timedelta(minutes=35),
        )
    )

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            SequenceAnalyzer([AnalysisError("timeout")]),
            telegram,
            AgentControl(),
            now_factory=lambda: next(times),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.analysis_retry_count == 2
    assert vacancy.analysis_next_retry_at == (
        NOW
        + timedelta(minutes=35)
        + timedelta(minutes=app_settings.check_interval_minutes)
    ).isoformat()


def test_pause_stops_retry_backlog_before_next_vacancy(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    second = VacancySummary(
        "job-2", "Backend developer", "https://example.com/vacancy/job-2", "Python"
    )
    for offset, summary in enumerate((SUMMARY, second)):
        discovered_at = NOW + timedelta(seconds=offset)
        assert database.discover(
            job_id=summary.id,
            title=summary.title,
            company="Example",
            url=summary.url,
            description_hash=f"hash-{offset}",
            search_query=summary.search_query,
            discovered_at=discovered_at,
        )
        assert database.mark_analysis_failed(
            summary.id,
            error_type="timeout",
            now=discovered_at,
            retry_after=timedelta(minutes=30),
            max_attempts=3,
        )
    control = AgentControl()
    analyzer = PausingAnalyzer(control)
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            FakeTelegram(),
            control,
            now_factory=lambda: NOW + timedelta(minutes=31),
        )
    )

    assert analyzer.calls == 1
    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert database.get(second.id).status is VacancyStatus.DISCOVERED


def test_three_timeouts_are_retried_in_later_cycle(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    adapter = FakeProvider(
        [
            LLMTimeoutError(),
            LLMTimeoutError(),
            LLMTimeoutError(),
            LLMResponse(
                text='{"suitable":true,"confidence":0.91,"reason":"Relevant work"}',
                provider="fake",
                model="test-model",
            ),
            LLMResponse(
                text="Safe local-profile letter", provider="fake", model="test-model"
            ),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        pass

    provider = ManagedLLMProvider(
        adapter,
        database,
        max_retries=2,
        max_requests_per_day=100,
        sleep=no_sleep,
        now_factory=lambda: NOW,
    )
    analyzer = VacancyAnalyzer(app_settings, provider)
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            now_factory=lambda: NOW,
        )
    )
    assert database.get(SUMMARY.id).status is VacancyStatus.ANALYSIS_FAILED

    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            AgentControl(),
            now_factory=lambda: NOW + timedelta(minutes=30),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert vacancy.llm_decision is True
    assert len(adapter.requests) == 5


def test_agent_loop_retries_due_analysis_before_new_search(tmp_path: Path) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    assert database.mark_analysis_failed(
        SUMMARY.id,
        error_type="timeout",
        now=NOW,
        retry_after=timedelta(minutes=30),
        max_attempts=3,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    with pytest.raises(SearchReached):
        asyncio.run(
            agent_loop(
                app_settings,
                database,
                StopAtSearchHHClient(details),
                SequenceAnalyzer(
                    [
                        SuitabilityResult(
                            suitable=True,
                            confidence=0.91,
                            reason="Relevant work",
                        )
                    ]
                ),
                telegram,
                AgentControl(),
            )
        )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert vacancy.error_text == ""
    assert telegram.previews == [(SUMMARY.id, True)]


def test_agent_loop_resumes_interrupted_discovered_analysis_before_search(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    assert database.discover(
        job_id=SUMMARY.id,
        title=SUMMARY.title,
        company="Example",
        url=SUMMARY.url,
        description_hash="abc123",
        search_query=SUMMARY.search_query,
        discovered_at=NOW,
    )
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )

    with pytest.raises(SearchReached):
        asyncio.run(
            agent_loop(
                app_settings,
                database,
                StopAtSearchHHClient(details),
                SequenceAnalyzer(
                    [
                        SuitabilityResult(
                            suitable=True,
                            confidence=0.91,
                            reason="Relevant work",
                        )
                    ]
                ),
                telegram,
                AgentControl(),
            )
        )

    assert database.get(SUMMARY.id).status is VacancyStatus.PENDING_APPROVAL
    assert telegram.previews == [(SUMMARY.id, True)]


def test_retry_uses_valid_negative_decision_instead_of_technical_error(
    tmp_path: Path,
) -> None:
    app_settings = settings(tmp_path, "approval")
    database = Database(app_settings.database_path)
    database.init()
    telegram = FakeTelegram()
    details = VacancyDetails(
        SUMMARY, PageState.VACANCY_LOADED, "Example", "Build Python services"
    )
    analyzer = SequenceAnalyzer(
        [
            AnalysisError("timeout"),
            SuitabilityResult(
                suitable=False,
                confidence=0.2,
                reason="Role mismatch",
            ),
        ]
    )

    asyncio.run(
        process_vacancy(
            SUMMARY,
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            now_factory=lambda: NOW,
        )
    )
    asyncio.run(
        main_module.retry_due_analyses(
            app_settings,
            database,
            FakeHHClient(details),
            analyzer,
            telegram,
            AgentControl(),
            now_factory=lambda: NOW + timedelta(minutes=30),
        )
    )

    vacancy = database.get(SUMMARY.id)
    assert vacancy.status is VacancyStatus.REJECTED_BY_LLM
    assert vacancy.llm_decision is False
    assert vacancy.llm_reason == "Role mismatch"
    assert vacancy.error_text == ""
    assert telegram.previews == []


def test_cli_reports_configuration_error_without_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--check-config",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--profile",
            str(tmp_path / "missing.yaml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.startswith("Configuration error:")
    assert "Traceback" not in result.stderr
