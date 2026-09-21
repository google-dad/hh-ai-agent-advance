import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from config import load_settings
from database import Database
from llm.errors import LLMConfigurationError
from llm.factory import create_llm_provider
from llm.api_keys import ApiKeyManager
from llm.mistral_keys import MistralKeyCipher, MistralKeyManager
from llm.providers.ollama import OllamaProvider
import main as main_module
from tests.test_config import VALID_ENV, write_profile


def configured(tmp_path: Path, **overrides: str):
    return load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, **overrides},
    )


@pytest.mark.parametrize(
    ("overrides", "adapter_type"),
    [
        ({}, OllamaProvider),
    ],
)
def test_factory_builds_exactly_one_selected_adapter(
    tmp_path: Path, overrides: dict[str, str], adapter_type: type
) -> None:
    settings = configured(tmp_path, **overrides)
    database = Database(tmp_path / "agent.db")
    database.init()

    provider = create_llm_provider(settings, database)

    assert isinstance(provider.adapter, adapter_type)
    assert not hasattr(provider, "fallbacks")


def test_factory_builds_openai_compatible_key_manager(tmp_path: Path) -> None:
    settings = configured(
        tmp_path,
        LLM_PROVIDER="openai_compatible",
        LLM_MODEL="custom-model",
        OPENAI_COMPATIBLE_BASE_URL="https://provider.example/v1",
        OPENAI_COMPATIBLE_API_KEY="compatible-key",
        MISTRAL_KEYS_MASTER_KEY=Fernet.generate_key().decode(),
    )
    database = Database(tmp_path / "agent.db")
    database.init()

    provider = create_llm_provider(settings, database)

    assert isinstance(provider, ApiKeyManager)
    assert provider.provider == "openai_compatible"
    assert len(provider.list_keys()) == 1


def test_factory_builds_mistral_key_manager(tmp_path: Path) -> None:
    settings = configured(
        tmp_path,
        LLM_PROVIDER="mistral",
        LLM_MODEL="mistral-small-latest",
        MISTRAL_KEYS_MASTER_KEY=Fernet.generate_key().decode(),
    )
    database = Database(tmp_path / "agent.db")
    database.init()

    provider = create_llm_provider(settings, database)

    assert isinstance(provider, MistralKeyManager)
    assert provider.list_keys() == ()


def test_factory_rejects_wrong_master_for_existing_keys(tmp_path: Path) -> None:
    original_master = Fernet.generate_key().decode()
    database = Database(tmp_path / "agent.db")
    database.init()
    protected = MistralKeyCipher(original_master).protect("mistral-existing-key")
    assert database.add_mistral_key(
        encrypted_key=protected.encrypted_key,
        key_hmac=protected.key_hmac,
        suffix=protected.suffix,
        now=datetime(2026, 8, 9, tzinfo=UTC),
    ) is not None
    settings = configured(
        tmp_path,
        LLM_PROVIDER="mistral",
        LLM_MODEL="mistral-small-latest",
        MISTRAL_KEYS_MASTER_KEY=Fernet.generate_key().decode(),
    )

    with pytest.raises(LLMConfigurationError) as raised:
        create_llm_provider(settings, database)

    assert "MISTRAL_KEYS_MASTER_KEY" in str(raised.value)
    assert protected.encrypted_key not in str(raised.value)


def test_main_does_not_start_backend_when_existing_keys_use_another_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_master = Fernet.generate_key().decode()
    database_path = tmp_path / "agent.db"
    database = Database(database_path)
    database.init()
    protected = MistralKeyCipher(original_master).protect("mistral-existing-key")
    assert database.add_mistral_key(
        encrypted_key=protected.encrypted_key,
        key_hmac=protected.key_hmac,
        suffix=protected.suffix,
        now=datetime(2026, 8, 9, tzinfo=UTC),
    ) is not None
    settings = configured(
        tmp_path,
        LLM_PROVIDER="mistral",
        LLM_MODEL="mistral-small-latest",
        MISTRAL_KEYS_MASTER_KEY=Fernet.generate_key().decode(),
    )
    settings = replace(settings, database_path=database_path)

    class BackendMustNotStart:
        async def start(self):
            raise AssertionError("backend must not start")

        async def close(self):
            return None

    monkeypatch.setattr(
        main_module,
        "create_browser_backend",
        lambda _settings: BackendMustNotStart(),
    )

    with pytest.raises(LLMConfigurationError):
        asyncio.run(main_module.run(settings))


def test_factory_rejects_unknown_provider_even_if_validation_is_bypassed(
    tmp_path: Path,
) -> None:
    settings = configured(tmp_path)
    settings = replace(settings, llm=replace(settings.llm, provider="unknown"))
    database = Database(tmp_path / "agent.db")
    database.init()

    with pytest.raises(LLMConfigurationError):
        create_llm_provider(settings, database)
