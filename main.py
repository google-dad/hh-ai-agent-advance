from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aiogram.exceptions import TelegramAPIError

from ai_analyzer import AnalysisError, VacancyAnalyzer
from approval import ApprovalGuard, ApprovalService
from browser_backend import BrowserLaunchError, create_browser_backend
from config import ConfigError, Settings, load_settings
from database import Database, SearchRun, VacancyStatus
from fit_summary import normalize_fit_summary
from hh_client import HHClient, PageState, VacancySummary
from llm.api_keys import ApiKeyManager
from llm.base import LLMProvider
from llm.errors import LLMError
from llm.factory import create_llm_provider
from llm.types import LLMRequest
from logging_setup import configure_logging
from tg_bot import AgentControl, TelegramService
from vacancy_filter import title_rejection_reason
from version import __version__


logger = logging.getLogger(__name__)
ANALYSIS_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class VacancyProcessResult:
    outcome: str
    reason: str = ""
    page_state: PageState | None = None


@dataclass
class SearchRunStats:
    query_count: int = 0
    found_results: int = 0
    new_vacancies: int = 0
    duplicates: int = 0
    rejected_by_filter: int = 0
    rejected_by_llm: int = 0
    telegram_cards: int = 0
    error_count: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    error_reasons: Counter[str] = field(default_factory=Counter)
    last_safe_error: str = ""
    read_attempts: int = 0
    technical_failures: int = 0
    consecutive_page_errors: int = 0

    def record_error(self, reason: str) -> None:
        self.error_count += 1
        self.error_reasons[reason] += 1
        self.last_safe_error = reason

    def record(self, result: VacancyProcessResult) -> None:
        if result.outcome == "rejected_by_filter":
            self.rejected_by_filter += 1
            self.rejection_reasons[result.reason or "filter_rejected"] += 1
        elif result.outcome == "rejected_by_llm":
            self.rejected_by_llm += 1
            self.rejection_reasons["llm_rejected"] += 1
        elif result.outcome == "telegram_card":
            self.telegram_cards += 1
        elif result.outcome == "other_error":
            self.record_error(result.reason or "other_error")

        if result.page_state is None:
            return
        self.read_attempts += 1
        if result.page_state is PageState.PAGE_STRUCTURE_CHANGED:
            self.consecutive_page_errors += 1
        else:
            self.consecutive_page_errors = 0
        if result.page_state in {
            PageState.ACCESS_DENIED,
            PageState.CAPTCHA_DETECTED,
            PageState.NETWORK_ERROR,
        }:
            self.technical_failures += 1

    def circuit_reason(self, settings: Settings) -> str:
        if self.consecutive_page_errors >= settings.circuit_breaker_page_errors:
            return "page_structure_changed"
        if (
            self.read_attempts >= settings.circuit_breaker_min_sample
            and self.technical_failures / self.read_attempts
            >= settings.circuit_breaker_unknown_ratio
        ):
            return "technical_failure_ratio"
        return ""


async def process_vacancy(
    summary: VacancySummary,
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> VacancyProcessResult:
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()
    existing = database.get(summary.id)
    if summary.previously_sent:
        if existing is None or existing.status is not VacancyStatus.PENDING_APPROVAL:
            return VacancyProcessResult("ignored")
        try:
            await telegram.send_preview(existing, include_actions=True)
        except TelegramAPIError:
            return VacancyProcessResult("other_error", "telegram_error")
        return VacancyProcessResult("telegram_card")
    if existing is not None and (
        existing.status is not VacancyStatus.DISCOVERED
        or existing.llm_decision is not None
    ):
        return VacancyProcessResult("ignored")
    details = await hh_client.read_vacancy(summary, telegram.request_captcha)
    description_hash = (
        hashlib.sha256(details.description.encode()).hexdigest()
        if details.description
        else ""
    )
    if existing is None:
        if not database.discover(
            job_id=summary.id,
            title=summary.title,
            company=details.company,
            company_url=details.company_url,
            url=summary.url,
            description_hash=description_hash,
            search_query=summary.search_query,
            discovered_at=now,
        ):
            return VacancyProcessResult("ignored")
        logger.info("vacancy_discovered job_id=%s", summary.id)
    else:
        logger.info("vacancy_resumed job_id=%s", summary.id)
    if details.state is not PageState.VACANCY_LOADED:
        error = details.error or details.state.value
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text=error,
        )
        if details.state is PageState.CAPTCHA_DETECTED:
            await telegram.notify(f"CAPTCHA was not completed for vacancy {summary.id}.")
        logger.error("vacancy_read_failed job_id=%s state=%s", summary.id, details.state.value)
        return VacancyProcessResult("other_error", details.state.value, details.state)

    rejection = title_rejection_reason(
        summary.title, settings.profile.candidate.excluded_positions
    )
    if rejection:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_FILTER,
            llm_reason=f"Title matched excluded term: {rejection}",
        )
        logger.info("vacancy_rejected job_id=%s source=filter", summary.id)
        return VacancyProcessResult(
            "rejected_by_filter", rejection, PageState.VACANCY_LOADED
        )

    try:
        suitability = await analyzer.assess(summary.title, details.description)
    except AnalysisError as exc:
        database.mark_analysis_failed(
            summary.id,
            error_type=exc.error_type,
            now=clock(),
            retry_after=timedelta(minutes=settings.check_interval_minutes),
            max_attempts=ANALYSIS_MAX_ATTEMPTS,
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, exc.error_type)
        logger.warning("vacancy_analysis_failed job_id=%s error_type=%s", summary.id, exc.error_type)
        return VacancyProcessResult(
            "other_error",
            f"analysis_failed:{exc.error_type}",
            PageState.VACANCY_LOADED,
        )

    if not suitability.suitable:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_LLM,
            llm_decision=False,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            error_text="",
        )
        logger.info("vacancy_rejected job_id=%s source=llm", summary.id)
        return VacancyProcessResult(
            "rejected_by_llm", suitability.reason, PageState.VACANCY_LOADED
        )

    fit_summary = normalize_fit_summary(suitability.fit_points)
    company = await hh_client.read_company_details(details.company_url)
    database.store_company_details(
        summary.id, rating=company.rating, reviews_count=company.reviews_count
    )

    letter = await analyzer.generate_cover_letter(summary.title, details.description)
    if not letter.strip():
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text="cover_letter_failed",
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, "cover_letter_failed")
        return VacancyProcessResult(
            "other_error", "cover_letter_failed", PageState.VACANCY_LOADED
        )

    include_actions = settings.app_mode != "dry_run"
    if settings.app_mode == "dry_run":
        database.store_analysis(
            summary.id,
            cover_letter=letter,
            llm_decision=True,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            fit_summary=fit_summary,
        )
    elif not database.request_approval(
        job_id=summary.id,
        cover_letter=letter,
        llm_decision=True,
        llm_reason=suitability.reason,
        confidence=suitability.confidence,
        fit_summary=fit_summary,
        now=now,
    ):
        return VacancyProcessResult(
            "other_error", "approval_transition_failed", PageState.VACANCY_LOADED
        )
    try:
        await telegram.send_preview(
            database.get(summary.id), include_actions=include_actions
        )
    except TelegramAPIError:
        return VacancyProcessResult(
            "other_error", "telegram_error", PageState.VACANCY_LOADED
        )
    return VacancyProcessResult("telegram_card", page_state=PageState.VACANCY_LOADED)


async def run_search_cycle(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> SearchRun:
    now = now_factory or (lambda: datetime.now(UTC))
    started_at = now()
    stats = SearchRunStats()
    handled_ids: set[str] = set()
    state = "completed"
    circuit_reason = ""
    failure: Exception | None = None
    logger.info(
        "search_cycle_started queries=%s", len(settings.profile.hh.search_queries)
    )

    async def process(summary: VacancySummary) -> None:
        nonlocal circuit_reason, state
        if summary.id in handled_ids:
            return
        handled_ids.add(summary.id)
        result = await process_vacancy(
            summary,
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            now_factory=now,
        )
        stats.record(result)
        circuit_reason = stats.circuit_reason(settings)
        if circuit_reason:
            state = "paused_by_circuit_breaker"
            control.paused = True
            control.circuit_reason = circuit_reason

    try:
        database.expire_approved(now())
        database.requeue_due_analysis_failures(
            now(), max_attempts=ANALYSIS_MAX_ATTEMPTS
        )
        for vacancy in database.unprocessed_discovered():
            if control.paused:
                break
            await process(
                VacancySummary(
                    vacancy.id,
                    vacancy.title,
                    vacancy.url,
                    vacancy.search_query,
                )
            )
        for query in settings.profile.hh.search_queries:
            if control.paused:
                break
            logger.info("search_query_started query=%r", query)
            search = await hh_client.search_vacancies(
                query,
                settings.profile.hh.areas,
                settings.profile.hh.experience_filters,
            )
            stats.query_count += 1
            stats.found_results += search.found_results
            stats.new_vacancies += sum(
                not summary.previously_sent for summary in search.summaries
            )
            stats.duplicates += search.duplicates
            logger.info(
                "search_query_finished query=%r found=%s new=%s duplicates=%s",
                query,
                search.found_results,
                sum(not summary.previously_sent for summary in search.summaries),
                search.duplicates,
            )
            if search.error_reason:
                stats.record_error(search.error_reason)
                control.consecutive_search_errors += 1
            else:
                control.consecutive_search_errors = 0
            if (
                control.consecutive_search_errors
                >= settings.circuit_breaker_page_errors
            ):
                circuit_reason = "search_errors"
                state = "paused_by_circuit_breaker"
                control.paused = True
                control.circuit_reason = circuit_reason
            for summary in search.summaries:
                if control.paused:
                    break
                await process(summary)
        if not control.paused:
            await hh_client.check_messages(telegram.notify)
    except Exception as exc:
        state = "failed"
        stats.record_error(type(exc).__name__)
        failure = exc

    if circuit_reason:
        notification = (
            f"Поиск приостановлен: {circuit_reason}. "
            "Проверьте /diagnostics и выполните /resume после устранения причины."
        )
    elif state == "completed" and stats.telegram_cards == 0:
        reasons = (stats.rejection_reasons + stats.error_reasons).most_common(3)
        reason_text = ", ".join(f"{name}={count}" for name, count in reasons)
        notification = (
            f"Цикл завершён: найдено {stats.found_results}, "
            f"новых {stats.new_vacancies}, карточек: 0."
            + (f" Причины: {reason_text}." if reason_text else "")
        )
    else:
        notification = ""

    run_id = database.save_search_run(
        started_at=started_at,
        finished_at=now(),
        state=state,
        query_count=stats.query_count,
        found_results=stats.found_results,
        new_vacancies=stats.new_vacancies,
        duplicates=stats.duplicates,
        rejected_by_filter=stats.rejected_by_filter,
        rejected_by_llm=stats.rejected_by_llm,
        telegram_cards=stats.telegram_cards,
        error_count=stats.error_count,
        rejection_reasons=dict(stats.rejection_reasons),
        error_reasons=dict(stats.error_reasons),
        last_safe_error=stats.last_safe_error,
        circuit_reason=circuit_reason,
    )
    if failure is not None:
        raise failure
    if notification:
        try:
            await telegram.notify(notification)
        except TelegramAPIError:
            database.record_search_run_error(run_id, "telegram_error")
    result = database.latest_search_run()
    if result is None:
        raise RuntimeError("search run was not saved")
    logger.info(
        "search_cycle_finished state=%s cards=%s errors=%s",
        result.state,
        result.telegram_cards,
        result.error_count,
    )
    return result


async def retry_due_analyses(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()
    # ponytail: one worker owns this lifecycle; add per-row claims if
    # multi-worker deployments matter.
    database.requeue_due_analysis_failures(
        now, max_attempts=ANALYSIS_MAX_ATTEMPTS
    )
    for vacancy in database.unprocessed_discovered():
        if control.paused:
            break
        await process_vacancy(
            VacancySummary(
                vacancy.id,
                vacancy.title,
                vacancy.url,
                vacancy.search_query,
            ),
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            now_factory=clock,
        )


async def agent_loop(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
) -> None:
    while True:
        if not control.paused:
            run = await run_search_cycle(
                settings, database, hh_client, analyzer, telegram, control
            )
            control.next_run_at = (
                None
                if control.paused
                else datetime.fromisoformat(run.finished_at)
                + timedelta(minutes=settings.check_interval_minutes)
            )
        else:
            control.next_run_at = None
        try:
            await asyncio.wait_for(
                control.wake_event.wait(), settings.check_interval_minutes * 60
            )
        except TimeoutError:
            pass
        finally:
            control.wake_event.clear()
            control.next_run_at = None


async def run(settings: Settings) -> None:
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    llm_provider = create_llm_provider(settings, database)
    api_keys = (
        llm_provider if isinstance(llm_provider, ApiKeyManager) else None
    )
    telegram: TelegramService | None = None
    try:
        context = await backend.start()
        guard = ApprovalGuard(settings, database)
        hh_client = HHClient(context, settings, database, guard)
        if not await hh_client.ensure_login():
            raise RuntimeError(
                "HH.ru login is required. Run with BROWSER_HEADLESS=false and sign in manually."
            )
        control = AgentControl()
        approval_service = ApprovalService(settings, database, hh_client)
        telegram = TelegramService(
            settings, database, approval_service, control, api_keys=api_keys
        )
        analyzer = VacancyAnalyzer(settings, llm_provider)
        if api_keys is not None:
            api_keys.set_notifier(telegram.notify)
        if hasattr(telegram, "check_updates"):
            asyncio.create_task(telegram.check_updates(notify=True))
        await asyncio.gather(
            telegram.start_polling(),
            agent_loop(settings, database, hh_client, analyzer, telegram, control),
        )
    finally:
        try:
            if telegram is not None:
                await telegram.stop()
        finally:
            try:
                await llm_provider.close()
            finally:
                await backend.close()


async def check_llm(
    settings: Settings,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> None:
    database = Database(settings.database_path)
    database.init()
    provider = provider_factory(settings, database)
    try:
        response = await provider.generate_text(
            LLMRequest(
                system_instructions="Return only the word OK.",
                user_content="Reply OK.",
                model=settings.llm.model,
                temperature=0,
                # Reasoning models (e.g. gpt-oss) may spend many tokens before
                # producing visible content; keep headroom for healthcheck.
                max_output_tokens=min(settings.llm.max_output_tokens, 256),
                timeout_seconds=settings.llm.timeout_seconds,
                operation="healthcheck",
            )
        )
        print(
            f"LLM check: provider={response.provider} model={response.model} "
            f"latency_ms={response.latency_ms} success=true"
        )
    finally:
        await provider.close()


def cli(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> int:
    parser = argparse.ArgumentParser(description="Safe personal HH assistant")
    parser.add_argument(
        "--version",
        action="version",
        version=f"HH Agent v{__version__}",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-llm", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file, args.profile)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.check_config:
        print(
            f"Configuration valid: mode={settings.app_mode}, "
            f"browser={settings.browser_backend}"
        )
        return 0
    if args.check_llm:
        try:
            asyncio.run(check_llm(settings, provider_factory))
        except LLMError as exc:
            print(
                f"LLM check failed: provider={settings.llm.provider} "
                f"error_type={exc.category}",
                file=sys.stderr,
            )
            return 1
        return 0
    configure_logging(settings.log_path)
    try:
        asyncio.run(run(settings))
    except (BrowserLaunchError, RuntimeError) as exc:
        logger.error("startup_failed error=%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("agent_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
