import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from database import Database, VacancyStatus


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


def make_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "agent.db")
    database.init()
    return database


def discover(database: Database, job_id: str = "job-1", letter: str = "Letter") -> None:
    assert database.discover(
        job_id=job_id,
        title="Python developer",
        company="Example",
        url=f"https://example.com/vacancy/{job_id}",
        description_hash="abc123",
        search_query="Python",
        discovered_at=NOW,
    )
    if letter:
        assert database.request_approval(
            job_id=job_id,
            cover_letter=letter,
            llm_decision=True,
            llm_reason="Profile matches",
            confidence=0.8,
            now=NOW,
        )


def approve(database: Database, job_id: str = "job-1", letter: str = "Letter") -> str:
    discover(database, job_id, letter)
    token = database.approve(
        job_id=job_id,
        telegram_user_id=42,
        expected_user_id=42,
        now=NOW + timedelta(minutes=1),
    )
    assert token
    return token


def claim(database: Database, job_id: str, token: str, **overrides: object):
    arguments = {
        "job_id": job_id,
        "permit": token,
        "telegram_user_id": 42,
        "expected_user_id": 42,
        "app_mode": "approval",
        "enable_real_apply": True,
        "daily_limit": 5,
        "now": NOW + timedelta(minutes=2),
    }
    arguments.update(overrides)
    return database.claim_application(**arguments)


def test_status_transitions_are_conditional_and_terminal(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database, letter="")

    assert database.get("job-1").status is VacancyStatus.DISCOVERED
    assert database.transition(
        "job-1", VacancyStatus.DISCOVERED, VacancyStatus.REJECTED_BY_FILTER
    )
    assert not database.transition(
        "job-1", VacancyStatus.DISCOVERED, VacancyStatus.PENDING_APPROVAL
    )
    assert database.get("job-1").status is VacancyStatus.REJECTED_BY_FILTER


def test_duplicate_discovery_is_rejected(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database, letter="")

    assert not database.discover(
        job_id="job-1",
        title="Duplicate",
        company="Other",
        url="https://example.com/duplicate",
        description_hash="different",
        search_query="Other",
        discovered_at=NOW,
    )
    assert database.get("job-1").title == "Python developer"


def test_unprocessed_discovered_returns_only_interrupted_rows(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database, "interrupted", letter="")
    discover(database, "previewed", letter="")
    assert database.store_analysis(
        "previewed",
        cover_letter="Letter",
        llm_decision=True,
        llm_reason="Profile matches",
        confidence=0.8,
    )
    discover(database, "rejected", letter="")
    assert database.transition(
        "rejected", VacancyStatus.DISCOVERED, VacancyStatus.REJECTED_BY_FILTER
    )

    assert [row.id for row in database.unprocessed_discovered()] == ["interrupted"]


def test_company_and_fit_summary_are_persisted(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    assert database.discover(
        job_id="job-1",
        title="Python developer",
        company="Example",
        company_url="https://hh.ru/employer/123",
        url="https://hh.ru/vacancy/1",
        description_hash="abc123",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.store_company_details("job-1", rating=4.7, reviews_count=128)
    assert database.store_analysis(
        "job-1",
        cover_letter="Letter",
        llm_decision=True,
        llm_reason="Relevant",
        confidence=0.8,
        fit_summary="Навыки: Python",
    )

    vacancy = database.get("job-1")
    assert vacancy.company_url == "https://hh.ru/employer/123"
    assert (vacancy.company_rating, vacancy.company_reviews_count) == (4.7, 128)
    assert vacancy.fit_summary == "Навыки: Python"


def test_search_run_survives_reopen_with_aggregated_reasons(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    database = Database(path)
    database.init()

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
    )

    run = Database(path).latest_search_run()

    assert run is not None
    assert run.state == "completed"
    assert run.found_results == 9
    assert run.rejection_reasons == {"title": 2}
    assert run.error_reasons == {"network_error": 1}


def test_search_run_can_record_notification_error(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    run_id = database.save_search_run(
        started_at=NOW,
        finished_at=NOW,
        state="completed",
        query_count=1,
        found_results=0,
        new_vacancies=0,
        duplicates=0,
        rejected_by_filter=0,
        rejected_by_llm=0,
        telegram_cards=0,
        error_count=0,
        rejection_reasons={},
        error_reasons={},
    )

    assert database.record_search_run_error(run_id, "telegram_error")
    run = database.latest_search_run()
    assert run is not None
    assert run.error_count == 1
    assert run.error_reasons == {"telegram_error": 1}
    assert run.last_safe_error == "telegram_error"


def test_pending_card_does_not_expire_before_user_click(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)

    assert database.get("job-1").approval_expires_at is None
    assert database.expire_approved(NOW + timedelta(days=1)) == 0
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_approval_click_starts_permission_ttl(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)

    token = database.approve(
        "job-1", 42, 42, NOW + timedelta(hours=8), ttl_minutes=30
    )

    assert token
    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.APPROVED
    assert vacancy.approval_expires_at == (
        NOW + timedelta(hours=8, minutes=30)
    ).isoformat()


def test_analysis_failure_retry_is_due_once_per_cycle_and_stops_after_three(
    tmp_path: Path,
) -> None:
    database = make_database(tmp_path)
    discover(database, letter="")
    retry_delay = timedelta(minutes=30)

    assert database.mark_analysis_failed(
        "job-1",
        error_type="timeout",
        now=NOW,
        retry_after=retry_delay,
        max_attempts=3,
    )
    first_failure = database.get("job-1")
    assert first_failure.status is VacancyStatus.ANALYSIS_FAILED
    assert first_failure.llm_decision is None
    assert first_failure.error_text == "timeout"
    assert first_failure.analysis_retry_count == 1
    assert first_failure.analysis_next_retry_at == (
        NOW + timedelta(minutes=30)
    ).isoformat()
    assert database.requeue_due_analysis_failures(
        NOW + timedelta(minutes=29), max_attempts=3
    ) == 0

    assert database.requeue_due_analysis_failures(
        NOW + timedelta(minutes=30), max_attempts=3
    ) == 1
    assert database.get("job-1").status is VacancyStatus.DISCOVERED

    assert database.mark_analysis_failed(
        "job-1",
        error_type="transient_server",
        now=NOW + timedelta(minutes=30),
        retry_after=retry_delay,
        max_attempts=3,
    )
    assert database.get("job-1").analysis_retry_count == 2
    assert database.requeue_due_analysis_failures(
        NOW + timedelta(minutes=60), max_attempts=3
    ) == 1

    assert database.mark_analysis_failed(
        "job-1",
        error_type="timeout",
        now=NOW + timedelta(minutes=60),
        retry_after=retry_delay,
        max_attempts=3,
    )
    exhausted = database.get("job-1")
    assert exhausted.status is VacancyStatus.ANALYSIS_FAILED
    assert exhausted.analysis_retry_count == 3
    assert exhausted.analysis_next_retry_at is None
    assert database.requeue_due_analysis_failures(
        NOW + timedelta(days=1), max_attempts=3
    ) == 0


def test_status_migration_preserves_vacancy_and_adds_retry_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    statuses = ", ".join(
        f"'{status.value}'"
        for status in VacancyStatus
        if status is not VacancyStatus.ANALYSIS_FAILED
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"""
            CREATE TABLE vacancies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL,
                description_hash TEXT NOT NULL DEFAULT '',
                search_query TEXT NOT NULL DEFAULT '',
                llm_decision INTEGER,
                llm_reason TEXT NOT NULL DEFAULT '',
                confidence REAL,
                cover_letter TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ({statuses})),
                discovered_at TEXT NOT NULL,
                approval_requested_at TEXT,
                approval_expires_at TEXT,
                approved_at TEXT,
                applied_at TEXT,
                approver_id INTEGER,
                permit_hash TEXT,
                error_text TEXT NOT NULL DEFAULT '',
                applying_at TEXT,
                submit_attempted_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vacancies (
                id, title, company, url, description_hash, search_query,
                status, discovered_at, approval_requested_at,
                approval_expires_at, approved_at, applied_at, approver_id,
                permit_hash, error_text, applying_at, submit_attempted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-job",
                "Legacy developer",
                "Legacy Co",
                "https://example.com/legacy",
                "legacy-hash",
                "Python",
                VacancyStatus.DISCOVERED.value,
                "2026-07-20T09:00:00+00:00",
                "requested",
                "expires",
                "approved",
                "applied",
                42,
                "permit",
                "legacy error",
                "applying",
                "submitted",
            ),
        )

    database = Database(path)
    database.init()

    vacancy = database.get("legacy-job")
    assert vacancy.title == "Legacy developer"
    assert vacancy.approval_requested_at == "requested"
    assert vacancy.approval_expires_at == "expires"
    assert vacancy.approved_at == "approved"
    assert vacancy.applying_at == "applying"
    assert vacancy.submit_attempted_at == "submitted"
    assert vacancy.applied_at == "applied"
    assert vacancy.approver_id == 42
    assert vacancy.error_text == "legacy error"
    assert vacancy.analysis_retry_count == 0
    assert vacancy.analysis_next_retry_at is None


def test_existing_analysis_failure_becomes_due_after_retry_migration(
    tmp_path: Path,
) -> None:
    database = make_database(tmp_path)
    discover(database, letter="")
    assert database.transition(
        "job-1",
        VacancyStatus.DISCOVERED,
        VacancyStatus.ANALYSIS_FAILED,
        error_text="timeout",
    )

    database.init()

    vacancy = database.get("job-1")
    assert vacancy.analysis_retry_count == 1
    assert vacancy.analysis_next_retry_at == vacancy.discovered_at
    assert database.requeue_due_analysis_failures(NOW, max_attempts=3) == 1


def test_foreign_user_cannot_approve_or_skip(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    discover(database)

    assert (
        database.approve(
            job_id="job-1",
            telegram_user_id=99,
            expected_user_id=42,
            now=NOW + timedelta(minutes=1),
        )
        is None
    )
    assert not database.skip("job-1", telegram_user_id=99, expected_user_id=42)
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_claim_rechecks_mode_feature_flag_and_user(tmp_path: Path) -> None:
    for name, override in (
        ("dry", {"app_mode": "dry_run"}),
        ("disabled", {"enable_real_apply": False}),
        ("foreign", {"telegram_user_id": 99}),
    ):
        database = Database(tmp_path / f"{name}.db")
        database.init()
        token = approve(database)

        result = claim(database, "job-1", token, **override)

        assert not result.allowed
        assert database.get("job-1").status is VacancyStatus.APPROVED


def test_invalid_or_reused_permit_cannot_claim_twice(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)

    assert not claim(database, "job-1", "forged").allowed
    assert claim(database, "job-1", token).allowed
    assert not claim(database, "job-1", token).allowed
    assert database.get("job-1").status is VacancyStatus.APPLYING


def test_empty_letter_and_expired_permission_are_blocked(tmp_path: Path) -> None:
    empty_database = Database(tmp_path / "empty.db")
    empty_database.init()
    discover(empty_database, letter="")
    assert empty_database.request_approval(
        job_id="job-1",
        cover_letter="   ",
        llm_decision=True,
        llm_reason="Profile matches",
        confidence=0.8,
        now=NOW,
    )
    empty_token = empty_database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert empty_token
    assert not claim(empty_database, "job-1", empty_token).allowed

    expired_database = Database(tmp_path / "expired.db")
    expired_database.init()
    token = approve(expired_database)
    result = claim(
        expired_database,
        "job-1",
        token,
        now=NOW + timedelta(minutes=31),
    )
    assert not result.allowed
    assert expired_database.get("job-1").status is VacancyStatus.EXPIRED


def test_daily_limit_uses_persisted_applied_rows(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    database = Database(path)
    database.init()
    first_token = approve(database, "job-1")
    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    assert database.complete_application(
        "job-1", first_token, success=True, now=NOW + timedelta(minutes=3)
    )

    reopened = Database(path)
    second_token = approve(reopened, "job-2")
    second_claim = claim(reopened, "job-2", second_token, daily_limit=1)

    assert reopened.applied_today(NOW + timedelta(minutes=4)) == 1
    assert not second_claim.allowed
    assert reopened.get("job-2").status is VacancyStatus.APPROVED


def test_daily_limit_atomically_counts_in_progress_claims(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    first_token = approve(database, "job-1")
    second_token = approve(database, "job-2")

    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    second_claim = claim(database, "job-2", second_token, daily_limit=1)

    assert not second_claim.allowed
    assert second_claim.reason == "daily_limit"
    assert database.get("job-2").status is VacancyStatus.APPROVED


def test_only_the_claimant_permit_can_complete_application(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)
    assert claim(database, "job-1", token).allowed

    assert not database.complete_application(
        "job-1", "forged", success=True, now=NOW + timedelta(minutes=3)
    )
    assert database.get("job-1").status is VacancyStatus.APPLYING
    assert database.complete_application(
        "job-1", token, success=True, now=NOW + timedelta(minutes=3)
    )


def test_failed_final_submit_attempt_keeps_daily_reservation(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    first_token = approve(database, "job-1")
    second_token = approve(database, "job-2")
    assert claim(database, "job-1", first_token, daily_limit=1).allowed
    assert database.mark_submit_attempt(
        "job-1", first_token, now=NOW + timedelta(minutes=3), daily_limit=1
    )
    assert database.complete_application(
        "job-1",
        first_token,
        success=False,
        now=NOW + timedelta(minutes=3),
        error_text="success signal missing",
    )

    assert not claim(database, "job-2", second_token, daily_limit=1).allowed


def test_concurrent_claim_has_one_winner(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    token = approve(database)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(database, "job-1", token), range(2)))

    assert [result.allowed for result in results].count(True) == 1
    assert database.get("job-1").status is VacancyStatus.APPLYING


def test_legacy_applied_jobs_is_preserved_and_not_imported(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE applied_jobs (id TEXT PRIMARY KEY, title TEXT, url TEXT, applied_at TEXT)"
        )
        connection.execute(
            "INSERT INTO applied_jobs VALUES (?, ?, ?, ?)",
            ("legacy", "Rejected in old version", "https://example.com/legacy", "2026-07-20"),
        )

    database = Database(path)
    database.init()

    assert database.get("legacy") is None
    assert database.applied_today(NOW) == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT title FROM applied_jobs").fetchone()[0] == "Rejected in old version"


def test_legacy_vacancy_survives_status_and_ux_column_migration(tmp_path: Path) -> None:
    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE vacancies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL,
                description_hash TEXT NOT NULL DEFAULT '',
                search_query TEXT NOT NULL DEFAULT '',
                llm_decision INTEGER,
                llm_reason TEXT NOT NULL DEFAULT '',
                confidence REAL,
                cover_letter TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ('discovered', 'pending_approval')),
                discovered_at TEXT NOT NULL,
                approval_requested_at TEXT,
                approval_expires_at TEXT,
                approved_at TEXT,
                applied_at TEXT,
                approver_id INTEGER,
                permit_hash TEXT,
                error_text TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vacancies (
                id, title, company, url, description_hash, search_query,
                llm_reason, cover_letter, status, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "Legacy Python role",
                "Example",
                "https://hh.ru/vacancy/1",
                "hash",
                "Python",
                "Relevant",
                "Letter",
                "pending_approval",
                NOW.isoformat(),
            ),
        )

    database = Database(path)
    database.init()

    vacancy = database.get("job-1")
    assert vacancy.title == "Legacy Python role"
    assert vacancy.status is VacancyStatus.PENDING_APPROVAL
    assert vacancy.company_url == ""
    assert vacancy.fit_summary == ""
