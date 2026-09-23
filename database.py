from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable


class VacancyStatus(str, Enum):
    DISCOVERED = "discovered"
    REJECTED_BY_FILTER = "rejected_by_filter"
    REJECTED_BY_LLM = "rejected_by_llm"
    ANALYSIS_FAILED = "analysis_failed"  # LLM timeout / invalid_response при анализе
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    APPLYING = "applying"
    SKIPPED = "skipped"
    APPLIED = "applied"
    APPLY_FAILED = "apply_failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Vacancy:
    id: str
    title: str
    company: str
    company_url: str
    company_rating: float | None
    company_reviews_count: int | None
    url: str
    description_hash: str
    search_query: str
    llm_decision: bool | None
    llm_reason: str
    fit_summary: str
    confidence: float | None
    cover_letter: str
    status: VacancyStatus
    discovered_at: str
    approval_requested_at: str | None
    approval_expires_at: str | None
    approved_at: str | None
    applying_at: str | None
    submit_attempted_at: str | None
    applied_at: str | None
    approver_id: int | None
    error_text: str
    analysis_retry_count: int
    analysis_next_retry_at: str | None
    description: str = ""


@dataclass(frozen=True)
class SearchRun:
    id: int
    started_at: str
    finished_at: str
    state: str
    query_count: int
    found_results: int
    new_vacancies: int
    duplicates: int
    rejected_by_filter: int
    rejected_by_llm: int
    telegram_cards: int
    error_count: int
    rejection_reasons: dict[str, int]
    error_reasons: dict[str, int]
    last_safe_error: str
    circuit_reason: str


@dataclass(frozen=True)
class MistralKeyRow:
    id: int
    encrypted_key: str = field(repr=False)
    key_hmac: str = field(repr=False)
    suffix: str
    status: str
    cooldown_until: str | None
    last_checked_at: str | None
    last_error_type: str
    created_at: str
    updated_at: str
    provider: str = "mistral"


# Alias for provider-agnostic call sites.
ApiKeyRow = MistralKeyRow


@dataclass(frozen=True)
class ClaimResult:
    allowed: bool
    reason: str
    vacancy: Vacancy | None = None


ALLOWED_TRANSITIONS = {
    VacancyStatus.DISCOVERED: {
        VacancyStatus.REJECTED_BY_FILTER,
        VacancyStatus.REJECTED_BY_LLM,
        VacancyStatus.ANALYSIS_FAILED,
        VacancyStatus.PENDING_APPROVAL,
        VacancyStatus.APPLY_FAILED,
    },
    VacancyStatus.PENDING_APPROVAL: {
        VacancyStatus.APPROVED,
        VacancyStatus.SKIPPED,
        VacancyStatus.EXPIRED,
    },
    VacancyStatus.APPROVED: {
        VacancyStatus.APPLYING,
        VacancyStatus.EXPIRED,
        VacancyStatus.APPLY_FAILED,
    },
    VacancyStatus.APPLYING: {
        VacancyStatus.APPLIED,
        VacancyStatus.APPLY_FAILED,
    },
}


def _as_utc(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        statuses = ", ".join(f"'{status.value}'" for status in VacancyStatus)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS applied_jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    msg_id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS vacancies (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL DEFAULT '',
                    company_url TEXT NOT NULL DEFAULT '',
                    company_rating REAL,
                    company_reviews_count INTEGER,
                    url TEXT NOT NULL,
                    description_hash TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    search_query TEXT NOT NULL DEFAULT '',
                    llm_decision INTEGER,
                    llm_reason TEXT NOT NULL DEFAULT '',
                    fit_summary TEXT NOT NULL DEFAULT '',
                    confidence REAL,
                    cover_letter TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (status IN ({statuses})),
                    discovered_at TEXT NOT NULL,
                    approval_requested_at TEXT,
                    approval_expires_at TEXT,
                    approved_at TEXT,
                    applying_at TEXT,
                    submit_attempted_at TEXT,
                    applied_at TEXT,
                    approver_id INTEGER,
                    permit_hash TEXT,
                    error_text TEXT NOT NULL DEFAULT '',
                    analysis_retry_count INTEGER NOT NULL DEFAULT 0,
                    analysis_next_retry_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vacancies_status_idx ON vacancies(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vacancies_applied_at_idx ON vacancies(applied_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    success INTEGER CHECK (success IN (0, 1)),
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms INTEGER,
                    provider_request_id TEXT,
                    error_type TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS llm_requests_started_at_idx ON llm_requests(started_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS llm_requests_provider_idx ON llm_requests(provider)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('completed', 'failed', 'paused_by_circuit_breaker')
                    ),
                    query_count INTEGER NOT NULL DEFAULT 0,
                    found_results INTEGER NOT NULL DEFAULT 0,
                    new_vacancies INTEGER NOT NULL DEFAULT 0,
                    duplicates INTEGER NOT NULL DEFAULT 0,
                    rejected_by_filter INTEGER NOT NULL DEFAULT 0,
                    rejected_by_llm INTEGER NOT NULL DEFAULT 0,
                    telegram_cards INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    rejection_reasons_json TEXT NOT NULL DEFAULT '{}',
                    error_reasons_json TEXT NOT NULL DEFAULT '{}',
                    last_safe_error TEXT NOT NULL DEFAULT '',
                    circuit_reason TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mistral_api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encrypted_key TEXT NOT NULL,
                    key_hmac TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ready', 'cooldown', 'disabled')),
                    cooldown_until TEXT,
                    last_checked_at TEXT,
                    last_error_type TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'mistral'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mistral_key_store_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    legacy_import_completed INTEGER NOT NULL CHECK (legacy_import_completed IN (0, 1))
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO mistral_key_store_meta(singleton, legacy_import_completed)
                VALUES (1, 0)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_key_legacy_import (
                    provider TEXT PRIMARY KEY,
                    completed INTEGER NOT NULL CHECK (completed IN (0, 1))
                )
                """
            )
            key_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(mistral_api_keys)")
            }
            if "provider" not in key_columns:
                connection.execute(
                    "ALTER TABLE mistral_api_keys "
                    "ADD COLUMN provider TEXT NOT NULL DEFAULT 'mistral'"
                )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mistral_api_keys_provider_hmac
                ON mistral_api_keys(provider, key_hmac)
                """
            )
            # Seed per-provider legacy flags from the original singleton meta row.
            legacy_completed = connection.execute(
                """
                SELECT legacy_import_completed
                FROM mistral_key_store_meta
                WHERE singleton = 1
                """
            ).fetchone()
            if legacy_completed is not None and int(legacy_completed[0]) == 1:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO llm_key_legacy_import(provider, completed)
                    VALUES ('mistral', 1)
                    """
                )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(vacancies)")
            }
            migrations = {
                "applying_at": "TEXT",
                "submit_attempted_at": "TEXT",
                "company_url": "TEXT NOT NULL DEFAULT ''",
                "company_rating": "REAL",
                "company_reviews_count": "INTEGER",
                "fit_summary": "TEXT NOT NULL DEFAULT ''",
                "analysis_retry_count": "INTEGER NOT NULL DEFAULT 0",
                "analysis_next_retry_at": "TEXT",
                "description": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE vacancies ADD COLUMN {name} {definition}"
                    )
            self._migrate_status_check(connection)
            connection.execute(
                """
                UPDATE vacancies
                SET analysis_retry_count = 1,
                    analysis_next_retry_at = discovered_at
                WHERE status = ?
                  AND analysis_retry_count = 0
                  AND analysis_next_retry_at IS NULL
                """,
                (VacancyStatus.ANALYSIS_FAILED.value,),
            )

    def _migrate_status_check(self, connection: sqlite3.Connection) -> None:
        """Пересоздать таблицу vacancies если CHECK-ограничение не включает новые статусы.

        SQLite не позволяет изменить CHECK-ограничение через ALTER TABLE —
        поэтому при появлении новых значений VacancyStatus нужна полная пересборка таблицы.
        """
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vacancies'"
        ).fetchone()
        if row is None:
            return
        table_sql: str = row["sql"] or ""
        missing = [s.value for s in VacancyStatus if f"'{s.value}'" not in table_sql]
        if not missing:
            return
        import logging as _logging
        _logging.getLogger(__name__).info(
            "db_migrate_status_check missing=%s", missing
        )
        statuses = ", ".join(f"'{s.value}'" for s in VacancyStatus)
        connection.execute("ALTER TABLE vacancies RENAME TO _vacancies_old")
        connection.execute(
            f"""
            CREATE TABLE vacancies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                company_url TEXT NOT NULL DEFAULT '',
                company_rating REAL,
                company_reviews_count INTEGER,
                url TEXT NOT NULL,
                description_hash TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                search_query TEXT NOT NULL DEFAULT '',
                llm_decision INTEGER,
                llm_reason TEXT NOT NULL DEFAULT '',
                fit_summary TEXT NOT NULL DEFAULT '',
                confidence REAL,
                cover_letter TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ({statuses})),
                discovered_at TEXT NOT NULL,
                approval_requested_at TEXT,
                approval_expires_at TEXT,
                approved_at TEXT,
                applying_at TEXT,
                submit_attempted_at TEXT,
                applied_at TEXT,
                approver_id INTEGER,
                permit_hash TEXT,
                error_text TEXT NOT NULL DEFAULT '',
                analysis_retry_count INTEGER NOT NULL DEFAULT 0,
                analysis_next_retry_at TEXT
            )
            """
        )
        columns = (
            "id, title, company, url, description_hash, description, search_query, "
            "llm_decision, llm_reason, confidence, cover_letter, status, "
            "discovered_at, approval_requested_at, approval_expires_at, "
            "approved_at, applied_at, approver_id, permit_hash, error_text, "
            "applying_at, submit_attempted_at, company_url, company_rating, "
            "company_reviews_count, fit_summary, analysis_retry_count, analysis_next_retry_at"
        )
        connection.execute(
            f"INSERT INTO vacancies ({columns}) SELECT {columns} FROM _vacancies_old"
        )
        connection.execute("DROP TABLE _vacancies_old")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_status_idx ON vacancies(status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_applied_at_idx ON vacancies(applied_at)"
        )

    def update_cover_letter(self, job_id: str, cover_letter: str) -> bool:
        """Обновить сопроводительное письмо для вакансии в статусе pending_approval."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE vacancies SET cover_letter = ? WHERE id = ? AND status = ?",
                (cover_letter, job_id, VacancyStatus.PENDING_APPROVAL.value),
            )
            return cursor.rowcount == 1

    @staticmethod

    def _vacancy(row: sqlite3.Row | None) -> Vacancy | None:
        if row is None:
            return None
        return Vacancy(
            id=row["id"],
            title=row["title"],
            company=row["company"],
            company_url=row["company_url"],
            company_rating=row["company_rating"],
            company_reviews_count=row["company_reviews_count"],
            url=row["url"],
            description_hash=row["description_hash"],
            search_query=row["search_query"],
            llm_decision=None if row["llm_decision"] is None else bool(row["llm_decision"]),
            llm_reason=row["llm_reason"],
            fit_summary=row["fit_summary"],
            confidence=row["confidence"],
            cover_letter=row["cover_letter"],
            status=VacancyStatus(row["status"]),
            discovered_at=row["discovered_at"],
            approval_requested_at=row["approval_requested_at"],
            approval_expires_at=row["approval_expires_at"],
            approved_at=row["approved_at"],
            applying_at=row["applying_at"],
            submit_attempted_at=row["submit_attempted_at"],
            applied_at=row["applied_at"],
            approver_id=row["approver_id"],
            error_text=row["error_text"],
            analysis_retry_count=row["analysis_retry_count"],
            analysis_next_retry_at=row["analysis_next_retry_at"],
            description=row["description"],
        )

    def discover(
        self,
        *,
        job_id: str,
        title: str,
        company: str,
        company_url: str = "",
        url: str,
        description_hash: str,
        search_query: str,
        discovered_at: datetime,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO vacancies (
                    id, title, company, company_url, url, description_hash, search_query,
                    status, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    title,
                    company,
                    company_url,
                    url,
                    description_hash,
                    search_query,
                    VacancyStatus.DISCOVERED.value,
                    _iso(discovered_at),
                ),
            )
            return cursor.rowcount == 1

    def store_description(self, job_id: str, description: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE vacancies SET description = ? WHERE id = ?",
                (description, job_id),
            )

    def store_company_details(
        self, job_id: str, *, rating: float | None, reviews_count: int | None
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vacancies
                SET company_rating = ?, company_reviews_count = ?
                WHERE id = ?
                """,
                (rating, reviews_count, job_id),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str) -> Vacancy | None:
        with self._connect() as connection:
            return self._vacancy(
                connection.execute("SELECT * FROM vacancies WHERE id = ?", (job_id,)).fetchone()
            )

    def unprocessed_discovered(self) -> list[Vacancy]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM vacancies
                WHERE status = ? AND llm_decision IS NULL
                ORDER BY discovered_at
                """,
                (VacancyStatus.DISCOVERED.value,),
            ).fetchall()
            return [self._vacancy(row) for row in rows]

    def mark_analysis_failed(
        self,
        job_id: str,
        *,
        error_type: str,
        now: datetime,
        retry_after: timedelta,
        max_attempts: int,
    ) -> bool:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vacancies SET
                    status = ?, llm_decision = NULL, error_text = ?,
                    analysis_retry_count = analysis_retry_count + 1,
                    analysis_next_retry_at = CASE
                        WHEN analysis_retry_count + 1 < ? THEN ?
                        ELSE NULL
                    END
                WHERE id = ? AND status = ? AND analysis_retry_count < ?
                """,
                (
                    VacancyStatus.ANALYSIS_FAILED.value,
                    error_type,
                    max_attempts,
                    _iso(now + retry_after),
                    job_id,
                    VacancyStatus.DISCOVERED.value,
                    max_attempts,
                ),
            )
            return cursor.rowcount == 1

    def requeue_due_analysis_failures(
        self, now: datetime, *, max_attempts: int
    ) -> int:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vacancies
                SET status = ?, analysis_next_retry_at = NULL
                WHERE status = ?
                  AND analysis_retry_count < ?
                  AND analysis_next_retry_at IS NOT NULL
                  AND analysis_next_retry_at <= ?
                """,
                (
                    VacancyStatus.DISCOVERED.value,
                    VacancyStatus.ANALYSIS_FAILED.value,
                    max_attempts,
                    _iso(now),
                ),
            )
            return cursor.rowcount
    def transition(
        self,
        job_id: str,
        expected: VacancyStatus | Iterable[VacancyStatus],
        target: VacancyStatus,
        **fields: object,
    ) -> bool:
        expected_statuses = (
            (expected,) if isinstance(expected, VacancyStatus) else tuple(expected)
        )
        if not expected_statuses or any(
            target not in ALLOWED_TRANSITIONS.get(status, set()) for status in expected_statuses
        ):
            return False
        allowed_fields = {"llm_decision", "llm_reason", "confidence", "cover_letter", "error_text"}
        if unknown := set(fields) - allowed_fields:
            raise ValueError(f"Unsupported transition fields: {', '.join(sorted(unknown))}")
        assignments = ["status = ?", *(f"{name} = ?" for name in fields)]
        values = [target.value, *fields.values(), job_id, *(status.value for status in expected_statuses)]
        placeholders = ", ".join("?" for _ in expected_statuses)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE vacancies SET {', '.join(assignments)} WHERE id = ? AND status IN ({placeholders})",
                values,
            )
            return cursor.rowcount == 1

    def request_approval(
        self,
        *,
        job_id: str,
        cover_letter: str,
        llm_decision: bool,
        llm_reason: str,
        confidence: float | None = None,
        fit_summary: str = "",
        now: datetime,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vacancies SET
                    status = ?, cover_letter = ?, llm_decision = ?,
                    llm_reason = ?, fit_summary = ?, confidence = ?, approval_requested_at = ?,
                    approval_expires_at = NULL, error_text = ''
                WHERE id = ? AND status = ?
                """,
                (
                    VacancyStatus.PENDING_APPROVAL.value,
                    cover_letter,
                    int(llm_decision),
                    llm_reason,
                    fit_summary,
                    confidence,
                    _iso(now),
                    job_id,
                    VacancyStatus.DISCOVERED.value,
                ),
            )
            return cursor.rowcount == 1

    def store_analysis(
        self,
        job_id: str,
        *,
        cover_letter: str,
        llm_decision: bool,
        llm_reason: str,
        confidence: float | None = None,
        fit_summary: str = "",
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vacancies SET cover_letter = ?, llm_decision = ?,
                    llm_reason = ?, fit_summary = ?, confidence = ?, error_text = '',
                    analysis_next_retry_at = NULL
                WHERE id = ? AND status = ?
                """,
                (
                    cover_letter,
                    int(llm_decision),
                    llm_reason,
                    fit_summary,
                    confidence,
                    job_id,
                    VacancyStatus.DISCOVERED.value,
                ),
            )
            return cursor.rowcount == 1

    def approve(
        self,
        job_id: str,
        telegram_user_id: int,
        expected_user_id: int,
        now: datetime,
        ttl_minutes: int = 30,
    ) -> str | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != VacancyStatus.PENDING_APPROVAL.value:
                return None
            if telegram_user_id != expected_user_id:
                return None
            permit = secrets.token_urlsafe(32)
            permit_hash = hashlib.sha256(permit.encode()).hexdigest()
            cursor = connection.execute(
                """
                UPDATE vacancies SET
                    status = ?, approved_at = ?, approval_expires_at = ?,
                    approver_id = ?, permit_hash = ?
                WHERE id = ? AND status = ?
                """,
                (
                    VacancyStatus.APPROVED.value,
                    _iso(now),
                    _iso(now + timedelta(minutes=ttl_minutes)),
                    telegram_user_id,
                    permit_hash,
                    job_id,
                    VacancyStatus.PENDING_APPROVAL.value,
                ),
            )
            return permit if cursor.rowcount == 1 else None

    def skip(self, job_id: str, telegram_user_id: int, expected_user_id: int) -> bool:
        if telegram_user_id != expected_user_id:
            return False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE vacancies SET status = ?, approver_id = ? WHERE id = ? AND status = ?",
                (
                    VacancyStatus.SKIPPED.value,
                    telegram_user_id,
                    job_id,
                    VacancyStatus.PENDING_APPROVAL.value,
                ),
            )
            return cursor.rowcount == 1

    def claim_application(
        self,
        *,
        job_id: str,
        permit: str,
        telegram_user_id: int,
        expected_user_id: int,
        app_mode: str,
        enable_real_apply: bool,
        daily_limit: int,
        now: datetime,
    ) -> ClaimResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            vacancy = self._vacancy(row)
            if app_mode != "approval":
                return ClaimResult(False, "app_mode", vacancy)
            if not enable_real_apply:
                return ClaimResult(False, "real_apply_disabled", vacancy)
            if row is None or row["status"] != VacancyStatus.APPROVED.value:
                return ClaimResult(False, "invalid_status", vacancy)
            if not row["approval_requested_at"] or not row["approved_at"]:
                return ClaimResult(False, "missing_approval", vacancy)
            if telegram_user_id != expected_user_id or row["approver_id"] != expected_user_id:
                return ClaimResult(False, "wrong_user", vacancy)
            if not row["approval_expires_at"] or row["approval_expires_at"] <= _iso(now):
                connection.execute(
                    "UPDATE vacancies SET status = ?, permit_hash = NULL WHERE id = ? AND status = ?",
                    (VacancyStatus.EXPIRED.value, job_id, VacancyStatus.APPROVED.value),
                )
                return ClaimResult(False, "expired", vacancy)
            if not row["cover_letter"].strip():
                return ClaimResult(False, "empty_letter", vacancy)
            supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
            if not secrets.compare_digest(row["permit_hash"] or "", supplied_hash):
                return ClaimResult(False, "invalid_permit", vacancy)
            if self._reserved_today(connection, now) >= daily_limit:
                return ClaimResult(False, "daily_limit", vacancy)
            cursor = connection.execute(
                "UPDATE vacancies SET status = ?, applying_at = ? WHERE id = ? AND status = ? AND permit_hash = ?",
                (
                    VacancyStatus.APPLYING.value,
                    _iso(now),
                    job_id,
                    VacancyStatus.APPROVED.value,
                    row["permit_hash"],
                ),
            )
            claimed = cursor.rowcount == 1
            return ClaimResult(
                claimed,
                "allowed" if claimed else "concurrent_claim",
                vacancy,
            )

    def complete_application(
        self,
        job_id: str,
        permit: str,
        *,
        success: bool,
        now: datetime,
        error_text: str = "",
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
            cursor = connection.execute(
                """
                UPDATE vacancies SET status = ?, applied_at = ?, error_text = ?, permit_hash = NULL
                WHERE id = ? AND status = ? AND permit_hash = ?
                """,
                (
                    (VacancyStatus.APPLIED if success else VacancyStatus.APPLY_FAILED).value,
                    _iso(now) if success else None,
                    "" if success else error_text,
                    job_id,
                    VacancyStatus.APPLYING.value,
                    supplied_hash,
                ),
            )
            return cursor.rowcount == 1

    def mark_submit_attempt(
        self,
        job_id: str,
        permit: str,
        *,
        now: datetime,
        daily_limit: int,
    ) -> bool:
        supplied_hash = hashlib.sha256(permit.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM vacancies WHERE id = ?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != VacancyStatus.APPLYING.value
                or not secrets.compare_digest(row["permit_hash"] or "", supplied_hash)
            ):
                return False
            if not row["approval_expires_at"] or row["approval_expires_at"] <= _iso(now):
                connection.execute(
                    "UPDATE vacancies SET status = ?, permit_hash = NULL WHERE id = ? AND status = ? AND permit_hash = ?",
                    (
                        VacancyStatus.EXPIRED.value,
                        job_id,
                        VacancyStatus.APPLYING.value,
                        row["permit_hash"],
                    ),
                )
                return False
            start, end = self._day_bounds(now)
            this_row_reserved = any(
                timestamp is not None and start <= timestamp < end
                for timestamp in (row["applying_at"], row["submit_attempted_at"])
            )
            if self._reserved_today(connection, now) - int(this_row_reserved) >= daily_limit:
                return False
            cursor = connection.execute(
                """
                UPDATE vacancies SET submit_attempted_at = ?
                WHERE id = ? AND status = ? AND permit_hash = ?
                    AND submit_attempted_at IS NULL
                """,
                (
                    _iso(now),
                    job_id,
                    VacancyStatus.APPLYING.value,
                    row["permit_hash"],
                ),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _day_bounds(now: datetime) -> tuple[str, str]:
        aware = now.replace(tzinfo=UTC) if now.tzinfo is None else now
        start = datetime.combine(aware.date(), time.min, tzinfo=aware.tzinfo)
        return _iso(start), _iso(start + timedelta(days=1))

    def _applied_today(self, connection: sqlite3.Connection, now: datetime) -> int:
        start, end = self._day_bounds(now)
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM vacancies WHERE status = ? AND applied_at >= ? AND applied_at < ?",
                (VacancyStatus.APPLIED.value, start, end),
            ).fetchone()[0]
        )

    def _reserved_today(self, connection: sqlite3.Connection, now: datetime) -> int:
        start, end = self._day_bounds(now)
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM vacancies
                WHERE (status = ? AND applied_at >= ? AND applied_at < ?)
                   OR (status = ? AND applying_at >= ? AND applying_at < ?)
                   OR (status IN (?, ?) AND submit_attempted_at >= ? AND submit_attempted_at < ?)
                """,
                (
                    VacancyStatus.APPLIED.value,
                    start,
                    end,
                    VacancyStatus.APPLYING.value,
                    start,
                    end,
                    VacancyStatus.APPLYING.value,
                    VacancyStatus.APPLY_FAILED.value,
                    start,
                    end,
                ),
            ).fetchone()[0]
        )

    def applied_today(self, now: datetime) -> int:
        with self._connect() as connection:
            return self._applied_today(connection, now)

    def reserve_llm_request(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        now: datetime,
        daily_limit: int,
    ) -> int | None:
        start, end = self._day_bounds(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if daily_limit > 0:
                count = connection.execute(
                    "SELECT COUNT(*) FROM llm_requests WHERE started_at >= ? AND started_at < ?",
                    (start, end),
                ).fetchone()[0]
                if count >= daily_limit:
                    return None
            cursor = connection.execute(
                """
                INSERT INTO llm_requests (provider, model, operation, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (provider, model, operation, _iso(now)),
            )
            return int(cursor.lastrowid)

    def complete_llm_request(
        self,
        request_row_id: int,
        *,
        success: bool,
        finished_at: datetime,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        provider_request_id: str | None = None,
        error_type: str = "",
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE llm_requests SET
                    finished_at = ?, success = ?, input_tokens = ?,
                    output_tokens = ?, latency_ms = ?, provider_request_id = ?,
                    error_type = ?
                WHERE id = ? AND success IS NULL
                """,
                (
                    _iso(finished_at),
                    int(success),
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    provider_request_id,
                    error_type,
                    request_row_id,
                ),
            )
            return cursor.rowcount == 1

    def llm_requests_today(self, now: datetime) -> int:
        start, end = self._day_bounds(now)
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM llm_requests WHERE started_at >= ? AND started_at < ?",
                    (start, end),
                ).fetchone()[0]
            )

    def llm_usage_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(success = 1), 0) AS successful_requests,
                    COALESCE(SUM(success = 0), 0) AS failed_requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM llm_requests
                """
            ).fetchone()
            return {key: int(row[key]) for key in row.keys()}

    def expire_approved(self, now: datetime) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE vacancies SET status = ?, permit_hash = NULL
                WHERE status = ?
                    AND approval_expires_at IS NOT NULL
                    AND approval_expires_at <= ?
                """,
                (
                    VacancyStatus.EXPIRED.value,
                    VacancyStatus.APPROVED.value,
                    _iso(now),
                ),
            )
            return cursor.rowcount

    def pending(self) -> list[Vacancy]:
        with self._connect() as connection:
            return [
                self._vacancy(row)
                for row in connection.execute(
                    "SELECT * FROM vacancies WHERE status = ? ORDER BY discovered_at",
                    (VacancyStatus.PENDING_APPROVAL.value,),
                ).fetchall()
            ]

    def recent_applied(self, limit: int = 15) -> list[Vacancy]:
        with self._connect() as connection:
            return [
                self._vacancy(row)
                for row in connection.execute(
                    """
                    SELECT * FROM vacancies
                    WHERE status = ?
                    ORDER BY applied_at DESC, id DESC
                    LIMIT ?
                    """,
                    (VacancyStatus.APPLIED.value, limit),
                ).fetchall()
            ]

    def stats(self) -> dict[str, int]:
        counts = {status.value: 0 for status in VacancyStatus}
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM vacancies GROUP BY status"
            ):
                counts[row["status"]] = row["count"]
        return counts

    def save_search_run(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        state: str,
        query_count: int,
        found_results: int,
        new_vacancies: int,
        duplicates: int,
        rejected_by_filter: int,
        rejected_by_llm: int,
        telegram_cards: int,
        error_count: int,
        rejection_reasons: dict[str, int],
        error_reasons: dict[str, int],
        last_safe_error: str = "",
        circuit_reason: str = "",
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO search_runs (
                    started_at, finished_at, state, query_count, found_results,
                    new_vacancies, duplicates, rejected_by_filter, rejected_by_llm,
                    telegram_cards, error_count, rejection_reasons_json,
                    error_reasons_json, last_safe_error, circuit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(started_at),
                    _iso(finished_at),
                    state,
                    query_count,
                    found_results,
                    new_vacancies,
                    duplicates,
                    rejected_by_filter,
                    rejected_by_llm,
                    telegram_cards,
                    error_count,
                    json.dumps(rejection_reasons, ensure_ascii=False, sort_keys=True),
                    json.dumps(error_reasons, ensure_ascii=False, sort_keys=True),
                    last_safe_error,
                    circuit_reason,
                ),
            )
            return int(cursor.lastrowid)

    def latest_search_run(self) -> SearchRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM search_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return SearchRun(
            id=row["id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            state=row["state"],
            query_count=row["query_count"],
            found_results=row["found_results"],
            new_vacancies=row["new_vacancies"],
            duplicates=row["duplicates"],
            rejected_by_filter=row["rejected_by_filter"],
            rejected_by_llm=row["rejected_by_llm"],
            telegram_cards=row["telegram_cards"],
            error_count=row["error_count"],
            rejection_reasons=json.loads(row["rejection_reasons_json"]),
            error_reasons=json.loads(row["error_reasons_json"]),
            last_safe_error=row["last_safe_error"],
            circuit_reason=row["circuit_reason"],
        )

    def record_search_run_error(self, run_id: int, reason: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT error_reasons_json FROM search_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return False
            reasons = json.loads(row["error_reasons_json"])
            reasons[reason] = reasons.get(reason, 0) + 1
            cursor = connection.execute(
                """
                UPDATE search_runs
                SET error_count = error_count + 1,
                    error_reasons_json = ?, last_safe_error = ?
                WHERE id = ?
                """,
                (json.dumps(reasons, ensure_ascii=False, sort_keys=True), reason, run_id),
            )
            return cursor.rowcount == 1

    def is_message_processed(self, message_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM chat_messages WHERE msg_id = ?", (message_id,)
                ).fetchone()
                is not None
            )

    def add_processed_message(self, message_id: str, chat_id: str, text_value: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO chat_messages (msg_id, chat_id, text) VALUES (?, ?, ?)",
                (message_id, chat_id, text_value),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _mistral_key_row(row: sqlite3.Row) -> MistralKeyRow:
        keys = row.keys()
        return MistralKeyRow(
            id=row["id"],
            encrypted_key=row["encrypted_key"],
            key_hmac=row["key_hmac"],
            suffix=row["suffix"],
            status=row["status"],
            cooldown_until=row["cooldown_until"],
            last_checked_at=row["last_checked_at"],
            last_error_type=row["last_error_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            provider=row["provider"] if "provider" in keys else "mistral",
        )

    @staticmethod
    def _release_expired_mistral_keys(
        connection: sqlite3.Connection,
        now: datetime,
        *,
        provider: str | None = None,
    ) -> None:
        if provider is None:
            connection.execute(
                """
                UPDATE mistral_api_keys
                SET status = 'ready', cooldown_until = NULL,
                    last_error_type = '', updated_at = ?
                WHERE status = 'cooldown' AND cooldown_until <= ?
                """,
                (_iso(now), _iso(now)),
            )
            return
        connection.execute(
            """
            UPDATE mistral_api_keys
            SET status = 'ready', cooldown_until = NULL,
                last_error_type = '', updated_at = ?
            WHERE provider = ? AND status = 'cooldown' AND cooldown_until <= ?
            """,
            (_iso(now), provider, _iso(now)),
        )

    def import_legacy_api_key(
        self,
        *,
        provider: str,
        encrypted_key: str | None,
        key_hmac: str | None,
        suffix: str | None,
        now: datetime,
    ) -> int | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            completed_row = connection.execute(
                """
                SELECT completed FROM llm_key_legacy_import WHERE provider = ?
                """,
                (provider,),
            ).fetchone()
            if completed_row is not None and int(completed_row[0]) == 1:
                return None
            # Backward-compat: singleton mistral meta may already mark import done.
            if provider == "mistral" and completed_row is None:
                singleton = connection.execute(
                    """
                    SELECT legacy_import_completed
                    FROM mistral_key_store_meta
                    WHERE singleton = 1
                    """
                ).fetchone()
                if singleton is not None and int(singleton[0]) == 1:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO llm_key_legacy_import(provider, completed)
                        VALUES ('mistral', 1)
                        """
                    )
                    return None
            key_id = None
            if encrypted_key is not None and key_hmac is not None and suffix is not None:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO mistral_api_keys (
                        encrypted_key, key_hmac, suffix, status, created_at, updated_at,
                        provider
                    ) VALUES (?, ?, ?, 'ready', ?, ?, ?)
                    """,
                    (
                        encrypted_key,
                        key_hmac,
                        suffix,
                        _iso(now),
                        _iso(now),
                        provider,
                    ),
                )
                if cursor.rowcount == 1:
                    key_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_key_legacy_import(provider, completed)
                VALUES (?, 1)
                """,
                (provider,),
            )
            if provider == "mistral":
                connection.execute(
                    """
                    UPDATE mistral_key_store_meta
                    SET legacy_import_completed = 1
                    WHERE singleton = 1
                    """
                )
            return key_id

    def import_legacy_mistral_key(
        self,
        *,
        encrypted_key: str | None,
        key_hmac: str | None,
        suffix: str | None,
        now: datetime,
    ) -> int | None:
        return self.import_legacy_api_key(
            provider="mistral",
            encrypted_key=encrypted_key,
            key_hmac=key_hmac,
            suffix=suffix,
            now=now,
        )

    def add_api_key(
        self,
        *,
        provider: str,
        encrypted_key: str,
        key_hmac: str,
        suffix: str,
        now: datetime,
    ) -> int | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO mistral_api_keys (
                    encrypted_key, key_hmac, suffix, status, created_at, updated_at,
                    provider
                ) VALUES (?, ?, ?, 'ready', ?, ?, ?)
                """,
                (
                    encrypted_key,
                    key_hmac,
                    suffix,
                    _iso(now),
                    _iso(now),
                    provider,
                ),
            )
            return int(cursor.lastrowid) if cursor.rowcount == 1 else None

    def add_mistral_key(
        self,
        *,
        encrypted_key: str,
        key_hmac: str,
        suffix: str,
        now: datetime,
    ) -> int | None:
        return self.add_api_key(
            provider="mistral",
            encrypted_key=encrypted_key,
            key_hmac=key_hmac,
            suffix=suffix,
            now=now,
        )

    def api_encrypted_keys(self, provider: str) -> tuple[str, ...]:
        with self._connect() as connection:
            return tuple(
                row["encrypted_key"]
                for row in connection.execute(
                    """
                    SELECT encrypted_key FROM mistral_api_keys
                    WHERE provider = ? ORDER BY id
                    """,
                    (provider,),
                )
            )

    def mistral_encrypted_keys(self) -> tuple[str, ...]:
        return self.api_encrypted_keys("mistral")

    def api_key(
        self, key_id: int, now: datetime, *, provider: str | None = None
    ) -> MistralKeyRow | None:
        with self._connect() as connection:
            self._release_expired_mistral_keys(connection, now, provider=provider)
            if provider is None:
                row = connection.execute(
                    "SELECT * FROM mistral_api_keys WHERE id = ?", (key_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM mistral_api_keys
                    WHERE id = ? AND provider = ?
                    """,
                    (key_id, provider),
                ).fetchone()
            return None if row is None else self._mistral_key_row(row)

    def mistral_key(self, key_id: int, now: datetime) -> MistralKeyRow | None:
        return self.api_key(key_id, now, provider="mistral")

    def api_key_by_hmac(
        self, key_hmac: str, now: datetime, *, provider: str
    ) -> MistralKeyRow | None:
        with self._connect() as connection:
            self._release_expired_mistral_keys(connection, now, provider=provider)
            row = connection.execute(
                """
                SELECT * FROM mistral_api_keys
                WHERE key_hmac = ? AND provider = ?
                """,
                (key_hmac, provider),
            ).fetchone()
            return None if row is None else self._mistral_key_row(row)

    def mistral_key_by_hmac(
        self, key_hmac: str, now: datetime
    ) -> MistralKeyRow | None:
        return self.api_key_by_hmac(key_hmac, now, provider="mistral")

    def api_keys(self, now: datetime, *, provider: str) -> list[MistralKeyRow]:
        with self._connect() as connection:
            self._release_expired_mistral_keys(connection, now, provider=provider)
            rows = connection.execute(
                """
                SELECT * FROM mistral_api_keys
                WHERE provider = ? ORDER BY id
                """,
                (provider,),
            ).fetchall()
            return [self._mistral_key_row(row) for row in rows]

    def mistral_keys(self, now: datetime) -> list[MistralKeyRow]:
        return self.api_keys(now, provider="mistral")

    def update_api_key_state(
        self,
        key_id: int,
        *,
        status: str,
        cooldown_until: datetime | None,
        last_checked_at: datetime | None,
        last_error_type: str,
        now: datetime,
        provider: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if provider is None:
                cursor = connection.execute(
                    """
                    UPDATE mistral_api_keys
                    SET status = ?, cooldown_until = ?, last_checked_at = ?,
                        last_error_type = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        None if cooldown_until is None else _iso(cooldown_until),
                        None if last_checked_at is None else _iso(last_checked_at),
                        last_error_type,
                        _iso(now),
                        key_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE mistral_api_keys
                    SET status = ?, cooldown_until = ?, last_checked_at = ?,
                        last_error_type = ?, updated_at = ?
                    WHERE id = ? AND provider = ?
                    """,
                    (
                        status,
                        None if cooldown_until is None else _iso(cooldown_until),
                        None if last_checked_at is None else _iso(last_checked_at),
                        last_error_type,
                        _iso(now),
                        key_id,
                        provider,
                    ),
                )
            return cursor.rowcount == 1

    def update_mistral_key_state(
        self,
        key_id: int,
        *,
        status: str,
        cooldown_until: datetime | None,
        last_checked_at: datetime | None,
        last_error_type: str,
        now: datetime,
    ) -> bool:
        return self.update_api_key_state(
            key_id,
            status=status,
            cooldown_until=cooldown_until,
            last_checked_at=last_checked_at,
            last_error_type=last_error_type,
            now=now,
            provider="mistral",
        )

    def delete_api_key(self, key_id: int, *, provider: str | None = None) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if provider is None:
                cursor = connection.execute(
                    "DELETE FROM mistral_api_keys WHERE id = ?", (key_id,)
                )
            else:
                cursor = connection.execute(
                    "DELETE FROM mistral_api_keys WHERE id = ? AND provider = ?",
                    (key_id, provider),
                )
            return cursor.rowcount == 1

    def delete_mistral_key(self, key_id: int) -> bool:
        return self.delete_api_key(key_id, provider="mistral")
