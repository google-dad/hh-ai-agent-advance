import sqlite3
from datetime import UTC, datetime, timedelta

from database import Database


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def make_database(tmp_path):
    database = Database(tmp_path / "agent.db")
    database.init()
    return database


def add_key(database: Database, token: str = "cipher-one") -> int:
    key_id = database.add_mistral_key(
        encrypted_key=token,
        key_hmac=f"hmac-{token}",
        suffix="aB7x",
        now=NOW,
    )
    assert key_id is not None
    return key_id


def test_key_crud_stores_only_protected_fields(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)

    row = database.mistral_key(key_id, NOW)

    assert row is not None
    assert row.encrypted_key == "cipher-one"
    assert row.suffix == "aB7x"
    assert row.status == "ready"
    assert "api_key" not in row.__dict__
    assert "cipher-one" not in repr(row)


def test_duplicate_hmac_is_rejected(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)

    duplicate = database.add_mistral_key(
        encrypted_key="different-ciphertext",
        key_hmac="hmac-cipher-one",
        suffix="z9Yx",
        now=NOW,
    )

    assert duplicate is None
    assert database.mistral_key_by_hmac("hmac-cipher-one", NOW).id == key_id
    assert len(database.mistral_keys(NOW)) == 1


def test_key_state_persists_disabled_and_audit_fields(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)
    checked_at = NOW + timedelta(minutes=1)

    assert database.update_mistral_key_state(
        key_id,
        status="disabled",
        cooldown_until=None,
        last_checked_at=checked_at,
        last_error_type="authentication",
        now=checked_at,
    )

    row = database.mistral_key(key_id, checked_at)
    assert row.status == "disabled"
    assert row.cooldown_until is None
    assert row.last_checked_at == checked_at.isoformat()
    assert row.last_error_type == "authentication"


def test_expired_cooldown_returns_to_ready_atomically(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)
    cooldown_until = NOW + timedelta(minutes=10)
    database.update_mistral_key_state(
        key_id,
        status="cooldown",
        cooldown_until=cooldown_until,
        last_checked_at=NOW,
        last_error_type="rate_limit",
        now=NOW,
    )

    cooling = database.mistral_key(key_id, NOW)
    assert cooling.status == "cooldown"
    assert cooling.cooldown_until == cooldown_until.isoformat()
    ready = database.mistral_key(key_id, NOW + timedelta(minutes=11))
    assert ready.status == "ready"
    assert ready.cooldown_until is None
    assert ready.last_error_type == ""


def test_encrypted_key_read_does_not_release_expired_cooldown(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)
    database.update_mistral_key_state(
        key_id,
        status="cooldown",
        cooldown_until=NOW - timedelta(minutes=1),
        last_checked_at=NOW,
        last_error_type="rate_limit",
        now=NOW,
    )
    with sqlite3.connect(database.path) as connection:
        before = connection.execute(
            "SELECT * FROM mistral_api_keys WHERE id = ?", (key_id,)
        ).fetchone()

    assert database.mistral_encrypted_keys() == ("cipher-one",)

    with sqlite3.connect(database.path) as connection:
        after = connection.execute(
            "SELECT * FROM mistral_api_keys WHERE id = ?", (key_id,)
        ).fetchone()
    assert after == before


def test_legacy_import_runs_once_even_after_delete(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = database.import_legacy_mistral_key(
        encrypted_key="cipher-legacy",
        key_hmac="hmac-legacy",
        suffix="old1",
        now=NOW,
    )
    assert key_id is not None
    assert database.delete_mistral_key(key_id)

    assert database.import_legacy_mistral_key(
        encrypted_key="cipher-legacy",
        key_hmac="hmac-legacy",
        suffix="old1",
        now=NOW,
    ) is None
    assert database.mistral_keys(NOW) == []


def test_missing_legacy_key_still_closes_import(tmp_path) -> None:
    database = make_database(tmp_path)

    assert database.import_legacy_mistral_key(
        encrypted_key=None,
        key_hmac=None,
        suffix=None,
        now=NOW,
    ) is None
    assert database.import_legacy_mistral_key(
        encrypted_key="cipher-late",
        key_hmac="hmac-late",
        suffix="late",
        now=NOW,
    ) is None
    assert database.mistral_keys(NOW) == []


def test_delete_physically_removes_key_row(tmp_path) -> None:
    database = make_database(tmp_path)
    key_id = add_key(database)

    assert database.delete_mistral_key(key_id)
    assert database.mistral_key(key_id, NOW) is None
    with sqlite3.connect(database.path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM mistral_api_keys WHERE id = ?", (key_id,)
        ).fetchone()[0]
    assert count == 0


def test_api_keys_are_isolated_by_provider(tmp_path) -> None:
    database = make_database(tmp_path)
    mistral_id = database.add_api_key(
        provider="mistral",
        encrypted_key="cipher-mistral",
        key_hmac="hmac-mistral",
        suffix="mstr",
        now=NOW,
    )
    compatible_id = database.add_api_key(
        provider="openai_compatible",
        encrypted_key="cipher-compat",
        key_hmac="hmac-compat",
        suffix="cmpa",
        now=NOW,
    )
    assert mistral_id is not None
    assert compatible_id is not None

    assert [row.id for row in database.api_keys(NOW, provider="mistral")] == [mistral_id]
    assert [row.id for row in database.api_keys(NOW, provider="openai_compatible")] == [
        compatible_id
    ]
    assert database.api_key(mistral_id, NOW, provider="openai_compatible") is None
    assert database.api_key_by_hmac("hmac-mistral", NOW, provider="openai_compatible") is None
