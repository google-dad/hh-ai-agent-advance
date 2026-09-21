import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pydantic import BaseModel

from config import load_settings
from database import Database
from llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMDailyLimitError,
    LLMError,
    LLMInvalidResponseError,
    LLMNoAvailableKeysError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
    MistralKeyDuplicateError,
    MistralKeyInputError,
)
from llm.mistral_keys import (
    MistralKeyCheckResult,
    MistralKeyCipher,
    MistralKeyManager,
    MistralKeyView,
)
from llm.providers.fake import FakeProvider
from llm.types import LLMRequest, LLMResponse
from tests.test_config import VALID_ENV, write_profile


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def mistral_settings(tmp_path, master_key: str, legacy_key: str = ""):
    return load_settings(
        profile_path=write_profile(tmp_path),
        environ={
            **VALID_ENV,
            "LLM_PROVIDER": "mistral",
            "LLM_MODEL": "mistral-small-latest",
            "MISTRAL_KEYS_MASTER_KEY": master_key,
            "MISTRAL_API_KEY": legacy_key,
        },
    ).llm


def no_adapter(_api_key: str):
    raise AssertionError("legacy migration must not use the network")


def key_store_snapshot(database: Database):
    with sqlite3.connect(database.path) as connection:
        return (
            connection.execute(
                "SELECT * FROM mistral_api_keys ORDER BY id"
            ).fetchall(),
            connection.execute(
                "SELECT * FROM mistral_key_store_meta ORDER BY singleton"
            ).fetchall(),
        )


class StructuredAnswer(BaseModel):
    value: str


class AdapterFactory:
    def __init__(self, outcomes: dict[str, list[LLMResponse | Exception]]):
        self.outcomes = {key: iter(values) for key, values in outcomes.items()}
        self.adapters: dict[str, list[FakeProvider]] = {}
        self.created_keys: list[str] = []
        self.delay_seconds = 0.0
        self.close_errors: dict[str, Exception] = {}

    def __call__(self, api_key: str) -> FakeProvider:
        adapter = FakeProvider(
            self.outcomes[api_key], delay_seconds=self.delay_seconds
        )
        close_error = self.close_errors.get(api_key)
        if close_error is not None:
            async def fail_close() -> None:
                raise close_error

            adapter.close = fail_close
        self.adapters.setdefault(api_key, []).append(adapter)
        self.created_keys.append(api_key)
        return adapter


def request(*, structured: bool = False) -> LLMRequest:
    return LLMRequest(
        system_instructions="Return a short answer.",
        user_content="Test request.",
        model="mistral-small-latest",
        temperature=0,
        max_output_tokens=8,
        timeout_seconds=10,
        operation="vacancy_analysis",
        json_schema=StructuredAnswer.model_json_schema() if structured else None,
    )


def response(text: str) -> LLMResponse:
    return LLMResponse(text=text, provider="mistral", model="mistral-small-latest")


def make_manager(
    tmp_path: Path,
    outcomes: dict[str, list[LLMResponse | Exception]],
    *,
    keys: tuple[str, ...],
    now_factory=lambda: NOW,
    max_retries: int = 1,
    max_requests_per_day: int = 100,
):
    master_key = Fernet.generate_key().decode()
    settings = load_settings(
        profile_path=write_profile(tmp_path),
        environ={
            **VALID_ENV,
            "LLM_PROVIDER": "mistral",
            "LLM_MODEL": "mistral-small-latest",
            "LLM_MAX_RETRIES": str(max_retries),
            "LLM_MAX_REQUESTS_PER_DAY": str(max_requests_per_day),
            "MISTRAL_KEYS_MASTER_KEY": master_key,
        },
    )
    database = Database(tmp_path / "agent.db")
    database.init()
    cipher = MistralKeyCipher(master_key)
    for raw_key in keys:
        protected = cipher.protect(raw_key)
        assert database.add_mistral_key(
            encrypted_key=protected.encrypted_key,
            key_hmac=protected.key_hmac,
            suffix=protected.suffix,
            now=NOW,
        ) is not None
    factory = AdapterFactory(outcomes)

    async def no_sleep(_seconds: float) -> None:
        return None

    manager = MistralKeyManager(
        settings.llm,
        database,
        adapter_factory=factory,
        sleep=no_sleep,
        now_factory=now_factory,
    )
    return manager, factory, database


def test_cipher_round_trip_never_stores_plaintext(tmp_path) -> None:
    raw = "mistral-super-secret-key"
    cipher = MistralKeyCipher(Fernet.generate_key().decode())
    protected = cipher.protect(raw)

    assert cipher.reveal(protected.encrypted_key) == raw
    assert raw not in repr(protected)
    assert raw not in protected.encrypted_key
    assert protected.suffix == "-key"


def test_wrong_master_key_fails_without_exposing_ciphertext() -> None:
    first = MistralKeyCipher(Fernet.generate_key().decode())
    second = MistralKeyCipher(Fernet.generate_key().decode())
    encrypted = first.protect("mistral-super-secret-key").encrypted_key

    with pytest.raises(LLMConfigurationError) as raised:
        second.reveal(encrypted)

    assert encrypted not in str(raised.value)
    assert "MISTRAL_KEYS_MASTER_KEY" in str(raised.value)


def test_manager_rejects_wrong_master_before_network_or_database_changes(
    tmp_path,
) -> None:
    original_master = Fernet.generate_key().decode()
    database = Database(tmp_path / "agent.db")
    database.init()
    protected = MistralKeyCipher(original_master).protect(
        "mistral-existing-super-secret"
    )
    key_id = database.add_mistral_key(
        encrypted_key=protected.encrypted_key,
        key_hmac=protected.key_hmac,
        suffix=protected.suffix,
        now=NOW,
    )
    assert key_id is not None
    assert database.update_mistral_key_state(
        key_id,
        status="cooldown",
        cooldown_until=NOW - timedelta(minutes=1),
        last_checked_at=NOW,
        last_error_type="rate_limit",
        now=NOW,
    )
    before = key_store_snapshot(database)
    adapter_calls: list[str] = []

    def adapter_factory(api_key: str):
        adapter_calls.append(api_key)
        raise AssertionError("adapter must not be created")

    with pytest.raises(LLMConfigurationError) as raised:
        MistralKeyManager(
            mistral_settings(
                tmp_path,
                Fernet.generate_key().decode(),
                "mistral-new-legacy-secret",
            ),
            database,
            adapter_factory=adapter_factory,
            now_factory=lambda: NOW,
        )

    assert "MISTRAL_KEYS_MASTER_KEY" in str(raised.value)
    assert protected.encrypted_key not in str(raised.value)
    assert "mistral-existing-super-secret" not in str(raised.value)
    assert adapter_calls == []
    assert key_store_snapshot(database) == before


def test_manager_accepts_existing_ciphertext_with_correct_master(tmp_path) -> None:
    master_key = Fernet.generate_key().decode()
    database = Database(tmp_path / "agent.db")
    database.init()
    protected = MistralKeyCipher(master_key).protect("mistral-existing-key")
    assert database.add_mistral_key(
        encrypted_key=protected.encrypted_key,
        key_hmac=protected.key_hmac,
        suffix=protected.suffix,
        now=NOW,
    ) is not None

    manager = MistralKeyManager(
        mistral_settings(tmp_path, master_key),
        database,
        adapter_factory=no_adapter,
        now_factory=lambda: NOW,
    )

    assert isinstance(manager, MistralKeyManager)


def test_manager_imports_legacy_key_once_without_plaintext_in_sqlite(tmp_path) -> None:
    raw = "mistral-legacy-super-secret"
    master_key = Fernet.generate_key().decode()
    settings = mistral_settings(tmp_path, master_key, raw)
    database = Database(tmp_path / "agent.db")
    database.init()

    MistralKeyManager(
        settings,
        database,
        adapter_factory=no_adapter,
        now_factory=lambda: NOW,
    )
    MistralKeyManager(
        settings,
        database,
        adapter_factory=no_adapter,
        now_factory=lambda: NOW,
    )

    with sqlite3.connect(database.path) as connection:
        rows = connection.execute("SELECT * FROM mistral_api_keys").fetchall()
    assert len(rows) == 1
    assert raw not in repr(rows)
    assert raw not in database.path.read_bytes().decode(errors="ignore")


def test_empty_legacy_key_closes_one_shot_migration(tmp_path) -> None:
    master_key = Fernet.generate_key().decode()
    database = Database(tmp_path / "agent.db")
    database.init()

    MistralKeyManager(
        mistral_settings(tmp_path, master_key),
        database,
        adapter_factory=no_adapter,
        now_factory=lambda: NOW,
    )
    MistralKeyManager(
        mistral_settings(tmp_path, master_key, "mistral-too-late-secret"),
        database,
        adapter_factory=no_adapter,
        now_factory=lambda: NOW,
    )

    assert database.mistral_keys(NOW) == []


def test_key_views_and_check_results_contain_no_secret_fields() -> None:
    view = MistralKeyView(1, "aB7x", "ready", None, NOW, "", True)
    result = MistralKeyCheckResult(view, True, "")

    assert result.key is view
    assert "encrypted_key" not in view.__dict__
    assert "key_hmac" not in view.__dict__


@pytest.mark.parametrize(
    ("error_type", "category"),
    [
        (LLMNoAvailableKeysError, "no_available_keys"),
        (MistralKeyInputError, "invalid_key"),
        (MistralKeyDuplicateError, "duplicate_key"),
    ],
)
def test_key_errors_are_typed_and_safe(error_type, category: str) -> None:
    error = error_type()

    assert isinstance(error, LLMError)
    assert error.category == category


def test_auth_error_disables_key_and_replays_on_next_key(tmp_path: Path) -> None:
    manager, factory, database = make_manager(
        tmp_path,
        {
            "first-mistral-key": [LLMAuthenticationError()],
            "second-mistral-key": [response("ok"), response("still ok")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    assert asyncio.run(manager.generate_text(request())).text == "ok"
    assert asyncio.run(manager.generate_text(request())).text == "still ok"

    assert database.mistral_keys(NOW)[0].status == "disabled"
    assert manager.list_keys()[1].is_current is True
    assert database.llm_requests_today(NOW) == 3
    assert factory.adapters["first-mistral-key"][0].closed is True
    assert len(factory.adapters["second-mistral-key"]) == 1


def test_permission_error_disables_key_and_replays_on_next_key(tmp_path: Path) -> None:
    manager, _, database = make_manager(
        tmp_path,
        {
            "first-mistral-key": [LLMPermissionError()],
            "second-mistral-key": [response("ok")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    assert asyncio.run(manager.generate_text(request())).text == "ok"
    assert database.mistral_keys(NOW)[0].last_error_type == "permission"


def test_second_rate_limit_cools_key_and_rotates(tmp_path: Path) -> None:
    manager, _, database = make_manager(
        tmp_path,
        {
            "first-mistral-key": [LLMRateLimitError(), LLMRateLimitError()],
            "second-mistral-key": [response("ok")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
        max_retries=5,
    )

    assert asyncio.run(manager.generate_text(request())).text == "ok"
    first = database.mistral_keys(NOW)[0]
    assert first.status == "cooldown"
    assert datetime.fromisoformat(first.cooldown_until) == NOW + timedelta(hours=1)
    assert database.llm_requests_today(NOW) == 3


def test_rate_limit_retry_after_controls_cooldown(tmp_path: Path) -> None:
    manager, _, database = make_manager(
        tmp_path,
        {
            "first-mistral-key": [LLMRateLimitError(120), LLMRateLimitError(120)],
            "second-mistral-key": [response("ok")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    assert asyncio.run(manager.generate_text(request())).text == "ok"
    first = database.mistral_keys(NOW)[0]
    assert datetime.fromisoformat(first.cooldown_until) == NOW + timedelta(seconds=120)


@pytest.mark.parametrize(
    "error",
    [LLMTimeoutError(), LLMTransientError(), LLMInvalidResponseError()],
)
def test_transient_errors_do_not_rotate_keys(tmp_path: Path, error: LLMError) -> None:
    manager, factory, database = make_manager(
        tmp_path,
        {
            "first-mistral-key": [error, error],
            "second-mistral-key": [response("must not run")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    with pytest.raises(type(error)):
        asyncio.run(manager.generate_text(request()))

    assert manager.list_keys()[0].is_current is True
    assert database.mistral_keys(NOW)[0].status == "ready"
    assert "second-mistral-key" not in factory.adapters


def test_daily_limit_does_not_rotate_keys(tmp_path: Path) -> None:
    manager, factory, _ = make_manager(
        tmp_path,
        {
            "first-mistral-key": [response("first"), response("must not run")],
            "second-mistral-key": [response("must not run")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
        max_requests_per_day=1,
    )
    assert asyncio.run(manager.generate_text(request())).text == "first"

    with pytest.raises(LLMDailyLimitError):
        asyncio.run(manager.generate_text(request()))

    assert manager.list_keys()[0].is_current is True
    assert "second-mistral-key" not in factory.adapters


def test_structured_request_replays_after_rotation(tmp_path: Path) -> None:
    manager, _, _ = make_manager(
        tmp_path,
        {
            "first-mistral-key": [LLMAuthenticationError()],
            "second-mistral-key": [response('{"value":"ok"}')],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    raw, parsed = asyncio.run(
        manager.generate_structured(request(structured=True), StructuredAnswer)
    )

    assert raw.text == '{"value":"ok"}'
    assert parsed == StructuredAnswer(value="ok")


@pytest.mark.parametrize(
    "raw_key",
    ["too-short", "x" * 513, "mistral key with whitespace"],
)
def test_add_key_rejects_invalid_input_before_network(
    tmp_path: Path, raw_key: str
) -> None:
    manager, factory, database = make_manager(tmp_path, {}, keys=())

    with pytest.raises(MistralKeyInputError):
        asyncio.run(manager.add_key(raw_key))

    assert factory.adapters == {}
    assert database.mistral_keys(NOW) == []


def test_add_key_rejects_duplicate_before_network(tmp_path: Path) -> None:
    raw_key = "duplicate-mistral-key"
    manager, factory, database = make_manager(tmp_path, {}, keys=(raw_key,))

    with pytest.raises(MistralKeyDuplicateError):
        asyncio.run(manager.add_key(raw_key))

    assert factory.adapters == {}
    assert len(database.mistral_keys(NOW)) == 1


def test_add_key_health_checks_then_saves(tmp_path: Path) -> None:
    raw_key = "new-mistral-api-key"
    manager, factory, database = make_manager(
        tmp_path, {raw_key: [response("OK")]}, keys=()
    )

    view = asyncio.run(manager.add_key(raw_key))

    health_request = factory.adapters[raw_key][0].requests[0]
    assert health_request.operation == "api_key_healthcheck"
    assert health_request.max_output_tokens <= 256
    assert factory.adapters[raw_key][0].closed is True
    assert view.suffix == "-key"
    assert len(database.mistral_keys(NOW)) == 1


def test_failed_add_does_not_save_key(tmp_path: Path) -> None:
    raw_key = "bad-mistral-api-key"
    manager, _, database = make_manager(
        tmp_path, {raw_key: [LLMAuthenticationError()]}, keys=()
    )

    with pytest.raises(LLMAuthenticationError):
        asyncio.run(manager.add_key(raw_key))

    assert database.mistral_keys(NOW) == []


@pytest.mark.parametrize("initial_status", ["disabled", "cooldown"])
def test_successful_manual_check_returns_key_to_ready(
    tmp_path: Path, initial_status: str
) -> None:
    raw_key = "recoverable-mistral-key"
    manager, _, database = make_manager(
        tmp_path, {raw_key: [response("OK")]}, keys=(raw_key,)
    )
    cooldown = NOW + timedelta(hours=1) if initial_status == "cooldown" else None
    assert database.update_mistral_key_state(
        1,
        status=initial_status,
        cooldown_until=cooldown,
        last_checked_at=NOW,
        last_error_type="authentication",
        now=NOW,
    )

    checked = asyncio.run(manager.check_key(1))

    assert checked is not None
    assert checked.success is True
    assert checked.key.status == "ready"
    assert checked.error_type == ""


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (LLMAuthenticationError(), "disabled"),
        (LLMPermissionError(), "disabled"),
        (LLMRateLimitError(), "cooldown"),
        (LLMTransientError(), "ready"),
    ],
)
def test_manual_check_updates_state_by_error(
    tmp_path: Path, error: LLMError, expected_status: str
) -> None:
    raw_key = "checked-mistral-key"
    manager, _, database = make_manager(
        tmp_path, {raw_key: [error]}, keys=(raw_key,)
    )

    checked = asyncio.run(manager.check_key(1))

    assert checked is not None
    assert checked.success is False
    assert checked.key.status == expected_status
    assert checked.error_type == error.category
    assert database.llm_requests_today(NOW) == 0


def test_healthcheck_does_not_consume_production_daily_quota(tmp_path: Path) -> None:
    raw_key = "quota-safe-mistral-key"
    manager, _, database = make_manager(
        tmp_path,
        {raw_key: [response("OK"), response("production")]},
        keys=(raw_key,),
        max_requests_per_day=1,
    )

    assert asyncio.run(manager.check_key(1)).success is True
    assert database.llm_requests_today(NOW) == 0
    assert asyncio.run(manager.generate_text(request())).text == "production"
    assert database.llm_requests_today(NOW) == 1


def test_check_all_runs_health_checks_sequentially(tmp_path: Path) -> None:
    manager, factory, _ = make_manager(
        tmp_path,
        {
            "first-mistral-key": [response("OK")],
            "second-mistral-key": [response("OK")],
        },
        keys=("first-mistral-key", "second-mistral-key"),
    )

    checked = asyncio.run(manager.check_all())

    assert [result.success for result in checked] == [True, True]
    assert factory.created_keys == ["first-mistral-key", "second-mistral-key"]


def test_check_all_does_not_hold_generation_lock_during_network_io(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class CoordinatedAdapter:
        name = "mistral"

        async def complete(self, llm_request: LLMRequest) -> LLMResponse:
            if llm_request.operation == "api_key_healthcheck":
                started.set()
                await release.wait()
                return response("OK")
            return response("production")

        async def close(self) -> None:
            return None

    manager, _, _ = make_manager(
        tmp_path,
        {},
        keys=("first-mistral-key",),
    )
    manager._adapter_factory = lambda _raw: CoordinatedAdapter()

    async def scenario() -> None:
        checking = asyncio.create_task(manager.check_all())
        await asyncio.wait_for(started.wait(), 0.2)
        generated = await asyncio.wait_for(manager.generate_text(request()), 0.2)
        assert generated.text == "production"
        release.set()
        await checking

    asyncio.run(scenario())


def test_concurrent_state_change_wins_over_stale_healthcheck(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    raw_key = "concurrent-mistral-key"
    manager, _, database = make_manager(tmp_path, {}, keys=(raw_key,))

    class DelayedHealthAdapter:
        name = "mistral"

        async def complete(self, _request: LLMRequest) -> LLMResponse:
            started.set()
            await release.wait()
            return response("OK")

        async def close(self) -> None:
            return None

    manager._adapter_factory = lambda _raw: DelayedHealthAdapter()

    async def scenario() -> None:
        checking = asyncio.create_task(manager.check_key(1))
        await started.wait()
        assert database.update_mistral_key_state(
            1,
            status="disabled",
            cooldown_until=None,
            last_checked_at=NOW,
            last_error_type="authentication",
            now=NOW,
        )
        release.set()
        result = await checking
        assert result is not None
        assert result.key.status == "disabled"
        assert result.error_type == "authentication"

    asyncio.run(scenario())


def test_delete_current_closes_provider_and_clears_current(tmp_path: Path) -> None:
    raw_key = "current-mistral-key"
    manager, factory, database = make_manager(
        tmp_path, {raw_key: [response("ok")]}, keys=(raw_key,)
    )
    assert asyncio.run(manager.generate_text(request())).text == "ok"
    adapter = factory.adapters[raw_key][0]

    assert asyncio.run(manager.delete_key(1)) is True

    assert adapter.closed is True
    assert database.mistral_keys(NOW) == []
    assert all(not key.is_current for key in manager.list_keys())


def test_delete_current_waits_for_in_flight_request(tmp_path: Path) -> None:
    raw_key = "delayed-mistral-key"
    manager, factory, _ = make_manager(
        tmp_path, {raw_key: [response("ok")]}, keys=(raw_key,)
    )
    factory.delay_seconds = 0.02

    async def scenario() -> None:
        generated = asyncio.create_task(manager.generate_text(request()))
        await asyncio.sleep(0.005)
        adapter = factory.adapters[raw_key][0]
        deleted = asyncio.create_task(manager.delete_key(1))
        await asyncio.sleep(0)
        assert adapter.closed is False
        assert (await generated).text == "ok"
        assert await deleted is True
        assert adapter.closed is True

    asyncio.run(scenario())


def test_close_error_is_safe_and_does_not_restore_disabled_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_key = "close-error-mistral-key"
    manager, factory, database = make_manager(
        tmp_path, {raw_key: [LLMAuthenticationError()]}, keys=(raw_key,)
    )
    factory.close_errors[raw_key] = RuntimeError(raw_key)

    with caplog.at_level("WARNING"):
        checked = asyncio.run(manager.check_key(1))

    assert checked is not None
    assert checked.key.status == "disabled"
    assert database.mistral_keys(NOW)[0].status == "disabled"
    assert raw_key not in caplog.text


def test_empty_pool_notifies_once_and_successful_add_resets_latch(
    tmp_path: Path,
) -> None:
    raw_key = "notifier-mistral-key"
    manager, _, _ = make_manager(
        tmp_path, {raw_key: [response("OK")]}, keys=()
    )
    notifications: list[str] = []

    async def notify(message: str) -> None:
        notifications.append(message)

    manager.set_notifier(notify)
    for _ in range(2):
        with pytest.raises(LLMNoAvailableKeysError):
            asyncio.run(manager.generate_text(request()))
    assert len(notifications) == 1

    added = asyncio.run(manager.add_key(raw_key))
    assert asyncio.run(manager.delete_key(added.id)) is True
    with pytest.raises(LLMNoAvailableKeysError):
        asyncio.run(manager.generate_text(request()))
    assert len(notifications) == 2


def test_successful_check_and_expired_cooldown_reset_notification_latch(
    tmp_path: Path,
) -> None:
    current = [NOW]
    first = "checked-reset-mistral-key"
    second = "cooldown-reset-mistral-key"
    manager, _, database = make_manager(
        tmp_path,
        {first: [response("OK")], second: [response("available")]},
        keys=(first, second),
        now_factory=lambda: current[0],
    )
    for key_id in (1, 2):
        assert database.update_mistral_key_state(
            key_id,
            status="cooldown" if key_id == 2 else "disabled",
            cooldown_until=NOW + timedelta(minutes=1) if key_id == 2 else None,
            last_checked_at=NOW,
            last_error_type="rate_limit",
            now=NOW,
        )
    notifications: list[str] = []

    async def notify(message: str) -> None:
        notifications.append(message)

    manager.set_notifier(notify)
    with pytest.raises(LLMNoAvailableKeysError):
        asyncio.run(manager.generate_text(request()))
    assert asyncio.run(manager.check_key(1)).success is True
    assert asyncio.run(manager.delete_key(1)) is True
    with pytest.raises(LLMNoAvailableKeysError):
        asyncio.run(manager.generate_text(request()))
    current[0] = NOW + timedelta(minutes=2)
    assert asyncio.run(manager.generate_text(request())).text == "available"
    assert len(notifications) == 2


def test_logs_and_sqlite_do_not_expose_key_material(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_key = "secret-mistral-test-key"
    manager, _, database = make_manager(
        tmp_path, {raw_key: [LLMAuthenticationError()]}, keys=(raw_key,)
    )
    row = database.mistral_keys(NOW)[0]

    with caplog.at_level("WARNING"):
        checked = asyncio.run(manager.check_key(row.id))

    assert checked is not None
    sqlite_dump = database.path.read_bytes().decode(errors="ignore")
    assert raw_key not in sqlite_dump
    for secret in (raw_key, row.encrypted_key, row.key_hmac):
        assert secret not in caplog.text
