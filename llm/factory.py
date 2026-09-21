from config import Settings
from database import Database
from llm.api_keys import API_KEY_PROVIDERS, ApiKeyManager
from llm.base import LLMProvider
from llm.errors import LLMConfigurationError
from llm.managed import ManagedLLMProvider
from llm.providers.ollama import OllamaProvider


def create_llm_provider(settings: Settings, database: Database) -> LLMProvider:
    config = settings.llm
    if config.provider == "ollama":
        adapter = OllamaProvider(config.ollama_url)
        return ManagedLLMProvider(
            adapter,
            database,
            max_retries=config.max_retries,
            max_requests_per_day=config.max_requests_per_day,
        )
    if config.provider in API_KEY_PROVIDERS:
        return ApiKeyManager(config, database)
    raise LLMConfigurationError()
