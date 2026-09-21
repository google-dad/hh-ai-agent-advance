#!/usr/bin/env python3
"""
HH Agent — интерактивный мастер настройки.

Запуск: python setup_wizard.py
Редактирование существующих настроек: python setup_wizard.py --edit
"""
from __future__ import annotations

import argparse
import getpass
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import urllib.error
from pathlib import Path

from version import __version__

# ---------------------------------------------------------------------------
# ANSI-цвета (stdlib, без rich)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or (
    os.name == "nt" and os.environ.get("TERM_PROGRAM") in {"vscode", "mintty", "xterm"}
)

# На Windows включаем ANSI через kernel32 если возможно
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _USE_COLOR = True
    except Exception:
        pass


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:       return _c("1", t)
def green(t: str) -> str:      return _c("32", t)
def yellow(t: str) -> str:     return _c("33", t)
def red(t: str) -> str:        return _c("31", t)
def cyan(t: str) -> str:       return _c("36", t)
def dim(t: str) -> str:        return _c("2", t)
def blue(t: str) -> str:       return _c("34", t)


def ok(msg: str) -> None:
    print(f"  {green('✓')} {msg}")


def warn(msg: str) -> None:
    print(f"  {yellow('!')} {msg}")


def err(msg: str) -> None:
    print(f"  {red('✗')} {msg}")


def info(msg: str) -> None:
    print(f"  {dim('·')} {msg}")


def header(title: str) -> None:
    width = 60
    print()
    print(cyan("─" * width))
    print(bold(cyan(f"  {title}")))
    print(cyan("─" * width))


def section(title: str) -> None:
    print()
    print(bold(f"▶ {title}"))
    print()


# ---------------------------------------------------------------------------
# Утилиты ввода
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "", required: bool = False, secret: bool = False) -> str:
    """Запросить строку у пользователя."""
    hint = f" [{dim(default)}]" if default and not secret else ""
    full_prompt = f"  {bold(prompt)}{hint}: "
    while True:
        try:
            if secret:
                value = getpass.getpass(full_prompt)
            else:
                value = input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(yellow("\nВыход из мастера настройки."))
            sys.exit(0)
        if not value and default:
            return default
        if value:
            return value
        if not required:
            return ""
        warn("Это поле обязательно. Пожалуйста, введите значение.")


def ask_list(prompt: str, default: list[str] | None = None, required: bool = False) -> list[str]:
    """Запросить список через запятую."""
    hint_str = ", ".join(default) if default else ""
    raw = ask(prompt, default=hint_str, required=required)
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


def ask_choice(prompt: str, options: list[tuple[str, str]], default: int = 1) -> int:
    """Показать нумерованный список и вернуть индекс (1-based)."""
    print(f"  {bold(prompt)}")
    for i, (label, description) in enumerate(options, 1):
        marker = green("→") if i == default else " "
        print(f"    {marker} {bold(str(i))}) {label}")
        if description:
            print(f"         {dim(description)}")
    while True:
        raw = ask(f"Ваш выбор", default=str(default))
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        warn(f"Введите число от 1 до {len(options)}.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Да/нет вопрос."""
    hint = "[Y/n]" if default else "[y/N]"
    raw = ask(f"{prompt} {hint}", default="y" if default else "n").lower()
    return raw in ("y", "yes", "да", "д", "")


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
PROFILE_PATH = BASE_DIR / "profile.yaml"


# ---------------------------------------------------------------------------
# Проверки окружения
# ---------------------------------------------------------------------------

def detect_python() -> str:
    """Вернуть путь к текущему python."""
    return sys.executable


def is_in_venv() -> bool:
    return sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV") is not None


def find_venv() -> Path | None:
    """Ищем .venv или venv в папке проекта."""
    for name in (".venv", "venv", "env"):
        path = BASE_DIR / name
        if path.is_dir():
            return path
    return None


def get_venv_python(venv_path: Path) -> Path:
    """Вернуть путь к python внутри venv."""
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def check_deps_installed() -> bool:
    """Проверить что основные зависимости установлены."""
    try:
        import aiogram  # noqa: F401
        import yaml     # noqa: F401
        import dotenv   # noqa: F401
        return True
    except ImportError:
        return False


def step_environment() -> None:
    """Шаг 0: Проверка Python и зависимостей."""
    section("Проверка окружения Python")

    py = detect_python()
    ver = platform.python_version()
    info(f"Python: {py} (версия {ver})")

    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        err(f"Требуется Python 3.11+, у вас {ver}")
        print(f"\n  Установите Python 3.13: {cyan('https://www.python.org/downloads/')}")
        sys.exit(1)
    ok(f"Python {ver} подходит")

    in_venv = is_in_venv()
    venv_path = find_venv()

    if in_venv:
        ok("Виртуальное окружение активно")
    else:
        warn("Виртуальное окружение не активно")
        if venv_path:
            info(f"Найдено: {venv_path}")
            if ask_yes_no("Хотите продолжить без активированного venv (не рекомендуется)?", default=False):
                warn("Продолжаем без venv...")
            else:
                if os.name == "nt":
                    activate = venv_path / "Scripts" / "activate"
                    print(f"\n  Активируйте venv командой:\n  {cyan(str(activate))}")
                else:
                    activate = venv_path / "bin" / "activate"
                    print(f"\n  Активируйте venv командой:\n  {cyan(f'source {activate}')}")
                print(f"\n  Затем снова запустите: {cyan('python setup_wizard.py')}")
                sys.exit(0)
        else:
            print(f"\n  {yellow('Создать виртуальное окружение?')}")
            if ask_yes_no("Создать .venv прямо сейчас?", default=True):
                _create_venv()
                return  # перезапуск с инструкциями
            else:
                warn("Устанавливаем зависимости в системный Python (не рекомендуется)")

    if check_deps_installed():
        ok("Зависимости установлены")
    else:
        warn("Зависимости не установлены")
        req_file = BASE_DIR / "requirements.txt"
        if not req_file.exists():
            err(f"Файл {req_file} не найден!")
            sys.exit(1)
        if ask_yes_no("Установить зависимости из requirements.txt?", default=True):
            _install_deps(req_file)
        else:
            warn("Пропускаем установку — некоторые функции могут не работать")


def _create_venv() -> None:
    venv_path = BASE_DIR / ".venv"
    print(f"\n  Создаём {venv_path} ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        ok("Виртуальное окружение создано!")
    except subprocess.CalledProcessError as e:
        err(f"Не удалось создать venv: {e}")
        sys.exit(1)

    venv_py = get_venv_python(venv_path)
    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        if ask_yes_no("Установить зависимости в новый venv?", default=True):
            _install_deps(req_file, python=venv_py)

    print()
    if os.name == "nt":
        activate_cmd = str(venv_path / "Scripts" / "activate")
    else:
        activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
    print(f"  {yellow('Теперь активируйте venv и перезапустите wizard:')}")
    print(f"  {cyan(activate_cmd)}")
    print(f"  {cyan('python setup_wizard.py')}")
    sys.exit(0)


def _install_deps(req_file: Path, python: Path | None = None) -> None:
    py = str(python) if python else sys.executable
    print(f"\n  Устанавливаем зависимости...")
    try:
        subprocess.run(
            [py, "-m", "pip", "install", "--upgrade", "pip", "-q"],
            check=True,
        )
        result = subprocess.run(
            [py, "-m", "pip", "install", "-r", str(req_file)],
            capture_output=False,
        )
        if result.returncode == 0:
            ok("Зависимости успешно установлены!")
        else:
            err("Установка завершилась с ошибкой")
            if ask_yes_no("Продолжить anyway?", default=False):
                pass
            else:
                sys.exit(1)
    except Exception as e:
        err(f"Ошибка при установке: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def step_telegram(current: dict[str, str]) -> dict[str, str]:
    section("Telegram Bot")

    print(textwrap.dedent(f"""
    {bold('Что нужно:')}
    1. Откройте Telegram и найдите {cyan('@BotFather')}
    2. Отправьте команду {cyan('/newbot')} и следуйте инструкциям
    3. Скопируйте выданный токен (выглядит как {dim('1234567890:ABCdef...')})

    Ваш Telegram User ID можно узнать через {cyan('@userinfobot')} — напишите ему что угодно.
    """))

    token = ask(
        "Токен бота (TG_BOT_TOKEN)",
        default=current.get("TG_BOT_TOKEN", ""),
        required=True,
        secret=bool(not current.get("TG_BOT_TOKEN")),
    )

    user_id = ask(
        "Ваш Telegram User ID (числовой)",
        default=current.get("TG_USER_ID", ""),
        required=True,
    )
    try:
        int(user_id)
    except ValueError:
        err("User ID должен быть числом")
        return step_telegram(current)

    print()
    ok(f"Telegram: токен сохранён, User ID = {user_id}")
    return {"TG_BOT_TOKEN": token, "TG_USER_ID": user_id}


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

def _check_ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _check_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def step_llm(current: dict[str, str]) -> dict[str, str]:
    section("AI-провайдер (LLM)")

    print(textwrap.dedent(f"""
    {bold('Провайдер анализирует вакансии и генерирует сопроводительные письма.')}

    {green('Ollama')} — работает локально, бесплатно, данные никуда не уходят.
    {yellow('Mistral API')} — облачный сервис, нужен API ключ, данные уходят к Mistral.
    {dim('OpenAI-compatible')} — любой совместимый /chat/completions endpoint.
    """))

    current_provider = current.get("LLM_PROVIDER", "ollama")
    default_choice = {"ollama": 1, "mistral": 2, "openai_compatible": 3}.get(current_provider, 1)

    choice = ask_choice(
        "Выберите провайдер:",
        options=[
            ("Ollama (локально)", "рекомендуется — бесплатно и приватно"),
            ("Mistral API", "облачный, нужен API ключ"),
            ("OpenAI-compatible API", "для других совместимых сервисов"),
        ],
        default=default_choice,
    )

    result: dict[str, str] = {}

    if choice == 1:
        result = _setup_ollama(current)
    elif choice == 2:
        result = _setup_mistral(current)
    else:
        result = _setup_openai_compat(current)

    return result


def _setup_ollama(current: dict[str, str]) -> dict[str, str]:
    print()
    installed = _check_ollama_installed()
    running = _check_ollama_running()

    if installed:
        ok("Ollama установлена")
    else:
        warn("Ollama не найдена в системе")
        print(textwrap.dedent(f"""
    {bold('Установка Ollama:')}
      macOS/Linux: {cyan('curl -fsSL https://ollama.com/install.sh | sh')}
      Windows:     {cyan('https://ollama.com/download')}

    После установки запустите: {cyan('ollama serve')}
    Или откройте приложение Ollama.
        """))
        if not ask_yes_no("Ollama уже установлена (я только что её поставил)?", default=False):
            warn("Продолжаем без Ollama — измените настройки позже через setup_wizard.py --edit")

    if running:
        ok("Ollama запущена (localhost:11434)")
    else:
        if installed:
            warn("Ollama не отвечает на localhost:11434")
            print(f"  Запустите: {cyan('ollama serve')} или откройте приложение Ollama")
            input(f"  {dim('Нажмите Enter после запуска Ollama...')}")
            if _check_ollama_running():
                ok("Ollama теперь доступна!")
            else:
                warn("Ollama по-прежнему недоступна — проверьте позже")

    # Показать доступные модели
    default_model = current.get("LLM_MODEL", "llama3")
    if running or _check_ollama_running():
        try:
            import urllib.request as req
            import json
            with req.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
                data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                print(f"\n  {bold('Установленные модели:')} {', '.join(cyan(m) for m in models)}")
                if default_model not in models and models:
                    default_model = models[0]
            else:
                warn("Моделей нет. Рекомендуем: ollama pull llama3")
                print(f"  Или: {cyan('ollama pull mistral')}, {cyan('ollama pull gemma2')}")
        except Exception:
            pass

    model = ask("Название модели (LLM_MODEL)", default=default_model, required=True)

    print()
    ok(f"LLM: Ollama / {model}")
    return {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": model,
        "OLLAMA_URL": current.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
    }


def _setup_mistral(current: dict[str, str]) -> dict[str, str]:
    from cryptography.fernet import Fernet

    print()
    print(textwrap.dedent(f"""
    {bold('Получение API ключа Mistral:')}
    1. Зарегистрируйтесь на {cyan('https://console.mistral.ai/')}
    2. Перейдите в API Keys и создайте новый ключ
    3. {yellow('Сохраните ключ — он показывается только один раз')}

    {dim('Данные (текст вакансий + ваш профиль) уходят к Mistral.')}
    """))

    api_key = ask(
        "Mistral API Key",
        default=current.get("MISTRAL_API_KEY", ""),
        required=True,
        secret=bool(not current.get("MISTRAL_API_KEY")),
    )
    model = ask(
        "Модель",
        default=current.get("LLM_MODEL", "mistral-small-latest"),
        required=True,
    )

    master_key = (
        current.get("MISTRAL_KEYS_MASTER_KEY")
        or current.get("LLM_KEYS_MASTER_KEY")
        or Fernet.generate_key().decode()
    )

    print()
    ok(f"LLM: Mistral / {model}")
    return {
        "LLM_PROVIDER": "mistral",
        "LLM_MODEL": model,
        "MISTRAL_API_KEY": api_key,
        "MISTRAL_KEYS_MASTER_KEY": master_key,
        "LLM_KEYS_MASTER_KEY": master_key,
        "MISTRAL_BASE_URL": current.get("MISTRAL_BASE_URL", ""),
    }


def _setup_openai_compat(current: dict[str, str]) -> dict[str, str]:
    from cryptography.fernet import Fernet

    print()
    print(textwrap.dedent(f"""
    {bold('OpenAI-compatible API:')}
    Любой сервис с эндпоинтом {cyan('/chat/completions')}.
    Примеры: LocalAI, LM Studio, Together AI, Groq и т.п.

    {dim('HTTPS обязателен (HTTP только для localhost).')}
    {dim('Ключи хранятся зашифрованно; управлять пулом можно командой /keys в Telegram.')}
    """))

    base_url = ask(
        "Base URL (например https://api.together.xyz/v1)",
        default=current.get("OPENAI_COMPATIBLE_BASE_URL", ""),
        required=True,
    )
    api_key = ask(
        "API Key",
        default=current.get("OPENAI_COMPATIBLE_API_KEY", ""),
        required=True,
        secret=bool(not current.get("OPENAI_COMPATIBLE_API_KEY")),
    )
    model = ask(
        "Модель",
        default=current.get("LLM_MODEL", ""),
        required=True,
    )
    json_mode_raw = current.get("OPENAI_COMPATIBLE_JSON_MODE", "true")
    json_mode = ask_yes_no("Включить JSON mode (если сервис поддерживает)?", default=json_mode_raw == "true")

    master_key = (
        current.get("LLM_KEYS_MASTER_KEY")
        or current.get("MISTRAL_KEYS_MASTER_KEY")
        or Fernet.generate_key().decode()
    )

    print()
    ok(f"LLM: OpenAI-compatible / {model}")
    return {
        "LLM_PROVIDER": "openai_compatible",
        "LLM_MODEL": model,
        "OPENAI_COMPATIBLE_BASE_URL": base_url,
        "OPENAI_COMPATIBLE_API_KEY": api_key,
        "OPENAI_COMPATIBLE_JSON_MODE": "true" if json_mode else "false",
        "LLM_KEYS_MASTER_KEY": master_key,
        "MISTRAL_KEYS_MASTER_KEY": master_key,
    }


# ---------------------------------------------------------------------------
# Профиль кандидата
# ---------------------------------------------------------------------------

def step_profile_required(current_profile: dict) -> dict:
    section("Ваш профиль — обязательные поля")

    print(textwrap.dedent(f"""
    {bold('Эти поля используются для:')}
    · Анализа подходящих вакансий
    · Генерации сопроводительных писем
    · Поиска на HH.ru

    {yellow('Заполняйте только реальными данными.')}
    """))

    cand = current_profile.get("candidate", {})
    hh = current_profile.get("hh", {})

    name = ask("Ваше имя (для письма)", default=cand.get("name", ""), required=True)
    experience = ask(
        "Краткое описание опыта (1-3 предложения)",
        default=cand.get("experience_summary", ""),
        required=True,
    )
    positions = ask_list(
        "Желаемые должности через запятую",
        default=list(cand.get("desired_positions", [])),
        required=True,
    )

    print(f"\n  {dim('Точное название резюме на HH.ru — должно совпадать символ в символ.')}")
    resume_name = ask(
        "Название резюме на HH.ru",
        default=hh.get("resume_name", ""),
        required=True,
    )
    queries = ask_list(
        "Поисковые запросы на HH.ru через запятую",
        default=list(hh.get("search_queries", [])),
        required=True,
    )

    return {
        "name": name,
        "experience_summary": experience,
        "desired_positions": positions,
        "resume_name": resume_name,
        "search_queries": queries,
    }


def step_profile_optional(current_profile: dict, required_data: dict) -> dict:
    section("Профиль кандидата — дополнительные поля")

    print(textwrap.dedent(f"""
    {dim('Эти поля улучшают качество анализа и писем.')}
    {dim('Нажмите Enter чтобы пропустить любое поле.')}
    """))

    cand = current_profile.get("candidate", {})
    hh = current_profile.get("hh", {})

    location = ask("Город/локация (например: Москва)", default=cand.get("location", ""))
    education = ask("Образование (кратко)", default=cand.get("education", ""))
    technologies = ask_list(
        "Технологии через запятую (Python, FastAPI, ...)",
        default=list(cand.get("technologies", [])),
    )
    salary = ask("Ожидаемая зарплата (например: от 150 000 руб.)", default=cand.get("salary_expectation", ""))

    print(f"\n  {dim('Форматы работы: remote, office, hybrid')}")
    work_format = ask_list(
        "Формат работы через запятую",
        default=list(cand.get("work_format", [])),
    )
    excluded = ask_list(
        "Исключённые должности (junior, intern, ...)",
        default=list(cand.get("excluded_positions", [])),
    )

    print(f"\n  {dim('Регионы HH.ru: 1=Москва, 2=Санкт-Петербург, 113=Россия. Пусто = все.')}")
    areas = ask_list("Регионы HH.ru через запятую (ID или пусто)", default=list(hh.get("areas", [])))

    print(f"\n  {dim('Опыт: noExperience, between1And3, between3And6, moreThan6')}")
    exp_filters = ask_list(
        "Фильтр опыта через запятую (или пусто)",
        default=list(hh.get("experience_filters", [])),
    )

    additional = ask("Дополнительная информация о себе", default=cand.get("additional_information", ""))
    github = ask("GitHub URL (или пусто)", default=cand.get("github_url", ""))

    print(f"\n  {dim('Язык письма: ru или en')}")
    cover_lang = ask("Язык сопроводительного письма", default=current_profile.get("cover_letter", {}).get("language", "ru"))
    cover_style = ask("Стиль письма (professional / friendly / concise)", default=current_profile.get("cover_letter", {}).get("style", "professional"))

    return {
        "location": location,
        "education": education,
        "technologies": technologies,
        "salary_expectation": salary,
        "work_format": work_format,
        "excluded_positions": excluded,
        "areas": areas,
        "experience_filters": exp_filters,
        "additional_information": additional,
        "github_url": github,
        "cover_language": cover_lang,
        "cover_style": cover_style,
    }


# ---------------------------------------------------------------------------
# Режим запуска
# ---------------------------------------------------------------------------

def step_app_mode(current: dict[str, str]) -> dict[str, str]:
    section("Режим запуска")

    print(textwrap.dedent(f"""
    {bold('dry_run')} {green('(рекомендуется для начала)')}
      · Бот ищет и анализирует вакансии
      · Присылает превью в Telegram
      · {green('Никаких реальных откликов — безопасно')}

    {bold('approval')} (после успешного dry_run)
      · Бот шлёт вакансию в Telegram с кнопкой «Откликнуться»
      · {yellow('Отклик происходит только после вашего нажатия')}
      · Одноразовое подтверждение, истекает через 30 минут
    """))

    current_mode = current.get("APP_MODE", "dry_run")
    default = 1 if current_mode == "dry_run" else 2

    choice = ask_choice(
        "Режим:",
        options=[
            ("dry_run", "безопасный старт — никаких реальных откликов"),
            ("approval", "реальные отклики по вашему подтверждению"),
        ],
        default=default,
    )

    mode = "dry_run" if choice == 1 else "approval"
    real_apply = "true" if choice == 2 else "false"

    if choice == 2:
        print()
        warn("Убедитесь что dry_run работал корректно перед включением approval!")
        if not ask_yes_no("Вы действительно хотите включить режим с реальными откликами?", default=False):
            mode = "dry_run"
            real_apply = "false"
            ok("Переключено на dry_run — вы всегда сможете изменить позже")

    print()
    ok(f"Режим: {bold(mode)}")
    return {"APP_MODE": mode, "ENABLE_REAL_APPLY": real_apply}


# ---------------------------------------------------------------------------
# Запись файлов
# ---------------------------------------------------------------------------

def _write_env(content: str) -> None:
    ENV_PATH.write_text(content, encoding="utf-8")
    if os.name != "nt":
        ENV_PATH.chmod(0o600)


def build_env(
    telegram: dict[str, str],
    llm: dict[str, str],
    mode: dict[str, str],
    limits: dict[str, str] | None = None,
) -> str:
    """Собрать содержимое .env файла."""
    limits = limits or {}
    lines = [
        "# HH Agent — конфигурация (создано setup_wizard.py)",
        "",
        "# === Telegram ===",
        f"TG_BOT_TOKEN={telegram.get('TG_BOT_TOKEN', '')}",
        f"TG_USER_ID={telegram.get('TG_USER_ID', '')}",
        "",
        "# === LLM Provider ===",
        f"LLM_PROVIDER={llm.get('LLM_PROVIDER', 'ollama')}",
        f"LLM_MODEL={llm.get('LLM_MODEL', 'llama3')}",
        f"LLM_TIMEOUT_SECONDS={limits.get('LLM_TIMEOUT_SECONDS', '30')}",
        f"LLM_MAX_RETRIES={limits.get('LLM_MAX_RETRIES', '1')}",
        f"LLM_TEMPERATURE={limits.get('LLM_TEMPERATURE', '0')}",
        f"LLM_MAX_OUTPUT_TOKENS={limits.get('LLM_MAX_OUTPUT_TOKENS', '1200')}",
        f"LLM_MAX_REQUESTS_PER_DAY={limits.get('LLM_MAX_REQUESTS_PER_DAY', '100')}",
        "",
        "# Ollama",
        f"OLLAMA_URL={llm.get('OLLAMA_URL', 'http://localhost:11434/api/generate')}",
        "",
        "# Mistral / shared API key pool",
        f"MISTRAL_API_KEY={llm.get('MISTRAL_API_KEY', '')}",
        f"LLM_KEYS_MASTER_KEY={llm.get('LLM_KEYS_MASTER_KEY') or llm.get('MISTRAL_KEYS_MASTER_KEY', '')}",
        f"MISTRAL_KEYS_MASTER_KEY={llm.get('MISTRAL_KEYS_MASTER_KEY') or llm.get('LLM_KEYS_MASTER_KEY', '')}",
        f"MISTRAL_BASE_URL={llm.get('MISTRAL_BASE_URL', '')}",
        "",
        "# OpenAI-compatible",
        f"OPENAI_COMPATIBLE_BASE_URL={llm.get('OPENAI_COMPATIBLE_BASE_URL', '')}",
        f"OPENAI_COMPATIBLE_API_KEY={llm.get('OPENAI_COMPATIBLE_API_KEY', '')}",
        f"OPENAI_COMPATIBLE_JSON_MODE={llm.get('OPENAI_COMPATIBLE_JSON_MODE', 'true')}",
        "",
        "# === Режим работы ===",
        f"APP_MODE={mode.get('APP_MODE', 'dry_run')}",
        f"ENABLE_REAL_APPLY={mode.get('ENABLE_REAL_APPLY', 'false')}",
        "",
        "# === Браузер ===",
        "BROWSER_BACKEND=cloakbrowser",
        "BROWSER_HEADLESS=false",
        "BROWSER_PROFILE_DIR=.browser-profile",
        "",
        "# === Лимиты и интервалы ===",
        f"CHECK_INTERVAL_MINUTES={limits.get('CHECK_INTERVAL_MINUTES', '30')}",
        f"MAX_APPLICATIONS_PER_DAY={limits.get('MAX_APPLICATIONS_PER_DAY', '5')}",
        f"MAX_VACANCIES_PER_QUERY={limits.get('MAX_VACANCIES_PER_QUERY', '20')}",
        f"MAX_PAGES_PER_QUERY={limits.get('MAX_PAGES_PER_QUERY', '2')}",
        f"MIN_SECONDS_BETWEEN_ACTIONS={limits.get('MIN_SECONDS_BETWEEN_ACTIONS', '5')}",
        f"APPROVAL_TTL_MINUTES={limits.get('APPROVAL_TTL_MINUTES', '30')}",
        f"CAPTCHA_TIMEOUT_SECONDS={limits.get('CAPTCHA_TIMEOUT_SECONDS', '120')}",
        f"CAPTCHA_MAX_ATTEMPTS={limits.get('CAPTCHA_MAX_ATTEMPTS', '2')}",
        "",
        "# === Circuit Breaker ===",
        f"CIRCUIT_BREAKER_MIN_SAMPLE={limits.get('CIRCUIT_BREAKER_MIN_SAMPLE', '5')}",
        f"CIRCUIT_BREAKER_UNKNOWN_RATIO={limits.get('CIRCUIT_BREAKER_UNKNOWN_RATIO', '0.8')}",
        f"CIRCUIT_BREAKER_PAGE_ERRORS={limits.get('CIRCUIT_BREAKER_PAGE_ERRORS', '3')}",
    ]
    return "\n".join(lines) + "\n"


def _yaml_str(value: str) -> str:
    """Экранировать строку для YAML."""
    if not value:
        return '""'
    # Если содержит спецсимволы — обернуть в кавычки
    if any(c in value for c in ':#{}[]|>&!%@`,"\''):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return f'"{value}"'


def _yaml_list(items: list[str]) -> str:
    """Сформировать YAML список. Возвращает строку с ведущим пробелом для inline [] или
    строку начинающуюся с \n для блочного списка (чтобы использовать как key:{val})."""
    if not items:
        return " []"  # ведущий пробел: key: [] → валидный YAML
    return "\n" + "\n".join(f'    - "{item}"' for item in items)


def build_profile(required: dict, optional: dict) -> str:
    """Собрать содержимое profile.yaml."""
    positions = _yaml_list(required.get("desired_positions", []))
    technologies = _yaml_list(optional.get("technologies", []))
    work_format = _yaml_list(optional.get("work_format", []))
    excluded = _yaml_list(optional.get("excluded_positions", []))
    queries = _yaml_list(required.get("search_queries", []))
    areas = _yaml_list(optional.get("areas", []))
    exp_filters = _yaml_list(optional.get("experience_filters", []))
    projects: list[str] = []
    projects_yaml = _yaml_list(projects)

    lines = [
        "# HH Agent — профиль кандидата (создано setup_wizard.py)",
        "candidate:",
        f"  name: {_yaml_str(required.get('name', ''))}",
        f"  location: {_yaml_str(optional.get('location', ''))}",
        f"  desired_positions:{positions}",
        f"  experience_summary: {_yaml_str(required.get('experience_summary', ''))}",
        f"  education: {_yaml_str(optional.get('education', ''))}",
        f"  technologies:{technologies}",
        f"  projects:{projects_yaml}",
        f"  github_url: {_yaml_str(optional.get('github_url', ''))}",
        f"  salary_expectation: {_yaml_str(optional.get('salary_expectation', ''))}",
        f"  work_format:{work_format}",
        f"  excluded_positions:{excluded}",
        f"  additional_information: {_yaml_str(optional.get('additional_information', ''))}",
        "",
        "hh:",
        f"  resume_name: {_yaml_str(required.get('resume_name', ''))}",
        f"  search_queries:{queries}",
        f"  areas:{areas}",
        f"  experience_filters:{exp_filters}",
        "",
        "cover_letter:",
        f"  language: {_yaml_str(optional.get('cover_language', 'ru'))}",
        "  max_length: 1800",
        f"  style: {_yaml_str(optional.get('cover_style', 'professional'))}",
    ]
    return "\n".join(lines) + "\n"


def write_files(env_content: str, profile_content: str) -> None:
    """Записать .env и profile.yaml."""
    section("Сохранение конфигурации")

    _write_env(env_content)
    ok(f"Записан {bold('.env')}")

    PROFILE_PATH.write_text(profile_content, encoding="utf-8")
    ok(f"Записан {bold('profile.yaml')}")


# ---------------------------------------------------------------------------
# Валидация через main.py --check-config
# ---------------------------------------------------------------------------

def run_check_config() -> bool:
    section("Проверка конфигурации")
    print("  Запускаем: python main.py --check-config ...")
    print()
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "main.py"), "--check-config"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ok(green(result.stdout.strip()))
            return True
        else:
            err("Ошибка конфигурации:")
            for line in result.stderr.strip().splitlines():
                print(f"    {red(line)}")
            return False
    except Exception as e:
        err(f"Не удалось запустить проверку: {e}")
        return False


# ---------------------------------------------------------------------------
# Загрузка существующего конфига
# ---------------------------------------------------------------------------

def load_existing_env() -> dict[str, str]:
    """Прочитать существующий .env файл."""
    if not ENV_PATH.exists():
        return {}
    result: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def load_existing_profile() -> dict:
    """Прочитать существующий profile.yaml."""
    if not PROFILE_PATH.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def show_current_config(env: dict[str, str], profile: dict) -> None:
    """Показать сводку текущих настроек."""
    print()
    print(bold("  Текущая конфигурация:"))

    tg_token = env.get("TG_BOT_TOKEN", "")
    tg_user = env.get("TG_USER_ID", "")
    if tg_token:
        masked = tg_token[:8] + "..." if len(tg_token) > 8 else tg_token
        ok(f"Telegram: токен={masked}, User ID={tg_user}")
    else:
        err("Telegram: не настроен")

    provider = env.get("LLM_PROVIDER", "")
    model = env.get("LLM_MODEL", "")
    if provider:
        ok(f"LLM: {provider} / {model}")
    else:
        err("LLM: не настроен")

    cand = profile.get("candidate", {})
    name = cand.get("name", "")
    positions = cand.get("desired_positions", [])
    if name:
        pos_str = ", ".join(positions[:2]) if positions else "—"
        ok(f"Профиль: {name} ({pos_str})")
    else:
        err("Профиль: не заполнен")

    mode = env.get("APP_MODE", "—")
    real_apply = env.get("ENABLE_REAL_APPLY", "false")
    mode_display = f"{mode}" + (" + real apply" if real_apply == "true" else "")
    ok(f"Режим: {mode_display}")


# ---------------------------------------------------------------------------
# Расширенные настройки
# ---------------------------------------------------------------------------

def step_advanced(current: dict[str, str]) -> dict[str, str]:
    section("Расширенные настройки")
    print(f"  {dim('Нажмите Enter для сохранения текущего значения')}\n")

    def get(key: str, default: str) -> str:
        return current.get(key, default)

    result = {}
    result["CHECK_INTERVAL_MINUTES"] = ask(
        "Интервал проверки вакансий (минуты)",
        default=get("CHECK_INTERVAL_MINUTES", "30"),
    )
    result["MAX_APPLICATIONS_PER_DAY"] = ask(
        "Максимум откликов в день",
        default=get("MAX_APPLICATIONS_PER_DAY", "5"),
    )
    result["MAX_VACANCIES_PER_QUERY"] = ask(
        "Максимум вакансий на запрос",
        default=get("MAX_VACANCIES_PER_QUERY", "20"),
    )
    result["LLM_MAX_REQUESTS_PER_DAY"] = ask(
        "Лимит запросов к LLM в день",
        default=get("LLM_MAX_REQUESTS_PER_DAY", "100"),
    )
    result["APPROVAL_TTL_MINUTES"] = ask(
        "Время жизни подтверждения (минуты)",
        default=get("APPROVAL_TTL_MINUTES", "30"),
    )
    result["CIRCUIT_BREAKER_PAGE_ERRORS"] = ask(
        "Количество ошибок подряд для срабатывания защиты (circuit breaker)",
        default=get("CIRCUIT_BREAKER_PAGE_ERRORS", "3"),
    )
    return result


# ---------------------------------------------------------------------------
# Итоговые инструкции
# ---------------------------------------------------------------------------

def show_next_steps(mode: str, env: dict[str, str]) -> None:
    header("🎉 Настройка завершена!")
    print()

    provider = env.get("LLM_PROVIDER", "ollama")
    real_apply = env.get("ENABLE_REAL_APPLY", "false") == "true"

    steps = []

    if provider == "ollama":
        steps.append(f"Убедитесь что Ollama запущена: {cyan('ollama serve')}")
        model = env.get("LLM_MODEL", "llama3")
        steps.append(f"Загрузите модель если нужно: {cyan(f'ollama pull {model}')}")

    steps.append(f"Проверьте конфиг: {cyan('python main.py --check-config')}")
    steps.append(f"Проверьте LLM: {cyan('python main.py --check-llm')}")
    steps.append(f"Запустите агента: {cyan('python main.py')}")
    steps.append(f"В браузере {bold('войдите на HH.ru')} и нажмите Enter в терминале")

    if mode == "dry_run":
        steps.append(f"В Telegram проверьте {cyan('/status')} — режим должен быть {bold('dry_run')}")
        steps.append(f"Дождитесь превью вакансий — {green('кнопки отклика не будет')} (это нормально)")
    else:
        steps.append(f"В Telegram нажимайте {bold('«Откликнуться»')} только осознанно!")

    for i, step in enumerate(steps, 1):
        print(f"  {bold(str(i) + '.')} {step}")

    print()
    print(f"  {dim('Изменить настройки позже:')} {cyan('python setup_wizard.py --edit')}")
    print(f"  {dim('Документация:')} {cyan('README.md')}")
    print()


# ---------------------------------------------------------------------------
# Главный поток
# ---------------------------------------------------------------------------

def wizard_new(existing_env: dict[str, str], existing_profile: dict) -> None:
    """Полный wizard для первичной настройки."""
    header(f"HH Agent v{__version__} — Мастер настройки")
    print(f"""
  Этот мастер поможет вам настроить агента за несколько минут.
  Агент будет искать вакансии на HH.ru, анализировать их с помощью AI
  и присылать подходящие в Telegram.

  {dim('Ctrl+C в любой момент для выхода')}
""")

    # 0. Окружение
    step_environment()

    # 1. Telegram
    tg = step_telegram(existing_env)

    # 2. LLM
    llm = step_llm(existing_env)

    # 3. Профиль — обязательное
    required = step_profile_required(existing_profile)

    # 4. Профиль — опциональное
    if ask_yes_no("\nЗаполнить дополнительные поля профиля (улучшает качество)?", default=True):
        optional = step_profile_optional(existing_profile, required)
    else:
        optional = {}

    # 5. Режим
    mode = step_app_mode(existing_env)

    # 6. Запись файлов
    all_env = {**existing_env, **tg, **llm, **mode}
    env_content = build_env(tg, llm, mode, all_env)
    profile_content = build_profile(required, optional)
    write_files(env_content, profile_content)

    # 7. Проверка
    config_ok = run_check_config()
    if not config_ok:
        print(f"\n  {yellow('Запустите wizard снова для исправления:')} {cyan('python setup_wizard.py --edit')}")

    # 8. Инструкции
    show_next_steps(mode.get("APP_MODE", "dry_run"), all_env)


def wizard_edit(existing_env: dict[str, str], existing_profile: dict) -> None:
    """Режим редактирования существующей конфигурации."""
    header(f"HH Agent v{__version__} — Редактирование конфигурации")
    show_current_config(existing_env, existing_profile)

    while True:
        print()
        choice = ask_choice(
            "Что изменить?",
            options=[
                ("Telegram настройки", "токен бота и User ID"),
                ("LLM провайдер/модель", "Ollama, Mistral или OpenAI-compatible"),
                ("Профиль кандидата", "имя, опыт, должности, запросы"),
                ("Режим запуска", "dry_run или approval"),
                ("Расширенные настройки", "лимиты, интервалы, тайм-ауты"),
                ("Проверить конфигурацию", "запустить --check-config"),
                ("Выйти", ""),
            ],
            default=7,
        )

        if choice == 1:
            tg = step_telegram(existing_env)
            existing_env.update(tg)
            _save_env(existing_env, existing_profile)

        elif choice == 2:
            llm = step_llm(existing_env)
            existing_env.update(llm)
            _save_env(existing_env, existing_profile)

        elif choice == 3:
            required = step_profile_required(existing_profile)
            if ask_yes_no("Также редактировать дополнительные поля?", default=False):
                optional = step_profile_optional(existing_profile, required)
            else:
                optional = {
                    "location": existing_profile.get("candidate", {}).get("location", ""),
                    "education": existing_profile.get("candidate", {}).get("education", ""),
                    "technologies": list(existing_profile.get("candidate", {}).get("technologies", [])),
                    "salary_expectation": existing_profile.get("candidate", {}).get("salary_expectation", ""),
                    "work_format": list(existing_profile.get("candidate", {}).get("work_format", [])),
                    "excluded_positions": list(existing_profile.get("candidate", {}).get("excluded_positions", [])),
                    "areas": list(existing_profile.get("hh", {}).get("areas", [])),
                    "experience_filters": list(existing_profile.get("hh", {}).get("experience_filters", [])),
                    "additional_information": existing_profile.get("candidate", {}).get("additional_information", ""),
                    "github_url": existing_profile.get("candidate", {}).get("github_url", ""),
                    "cover_language": existing_profile.get("cover_letter", {}).get("language", "ru"),
                    "cover_style": existing_profile.get("cover_letter", {}).get("style", "professional"),
                }
            profile_content = build_profile(required, optional)
            PROFILE_PATH.write_text(profile_content, encoding="utf-8")
            ok(f"Профиль сохранён в {bold('profile.yaml')}")

        elif choice == 4:
            mode = step_app_mode(existing_env)
            existing_env.update(mode)
            _save_env(existing_env, existing_profile)

        elif choice == 5:
            advanced = step_advanced(existing_env)
            existing_env.update(advanced)
            _save_env(existing_env, existing_profile)

        elif choice == 6:
            run_check_config()

        else:
            break

    show_current_config(existing_env, load_existing_profile())
    print()
    ok("Готово!")


def _save_env(env: dict[str, str], profile: dict) -> None:
    """Пересобрать и сохранить .env из текущего словаря."""
    tg = {k: env.get(k, "") for k in ("TG_BOT_TOKEN", "TG_USER_ID")}
    llm_keys = (
        "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_URL",
        "MISTRAL_API_KEY", "LLM_KEYS_MASTER_KEY", "MISTRAL_KEYS_MASTER_KEY", "MISTRAL_BASE_URL",
        "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_JSON_MODE",
    )
    llm = {k: env.get(k, "") for k in llm_keys}
    mode = {k: env.get(k, "") for k in ("APP_MODE", "ENABLE_REAL_APPLY")}
    content = build_env(tg, llm, mode, env)
    _write_env(content)
    ok(f"Настройки сохранены в {bold('.env')}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HH Agent — интерактивный мастер настройки",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Примеры:
              python setup_wizard.py          # первичная настройка
              python setup_wizard.py --edit   # редактирование настроек
        """),
    )
    parser.add_argument("--edit", action="store_true", help="Редактировать существующую конфигурацию")
    args = parser.parse_args()

    existing_env = load_existing_env()
    existing_profile = load_existing_profile()
    has_config = ENV_PATH.exists() and PROFILE_PATH.exists()

    if args.edit:
        if not has_config:
            warn("Конфигурация не найдена. Запускаем первичную настройку.")
            wizard_new(existing_env, existing_profile)
        else:
            wizard_edit(existing_env, existing_profile)
        return

    if has_config:
        header(f"HH Agent v{__version__} — Конфигурация найдена")
        show_current_config(existing_env, existing_profile)
        print()
        choice = ask_choice(
            "Что делать?",
            options=[
                ("Редактировать настройки", "изменить отдельные параметры"),
                ("Пересоздать конфигурацию", "заполнить заново с нуля"),
                ("Выйти", ""),
            ],
            default=1,
        )
        if choice == 1:
            wizard_edit(existing_env, existing_profile)
        elif choice == 2:
            if ask_yes_no(yellow("Создать конфигурацию заново? (старые файлы будут перезаписаны)"), default=False):
                wizard_new({}, {})
        # choice 3 = exit
    else:
        wizard_new(existing_env, existing_profile)


if __name__ == "__main__":
    main()
