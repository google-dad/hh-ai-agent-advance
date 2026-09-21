# HH AI Agent (advance)

Форк [fikstt2/hh-ai-agent](https://github.com/fikstt2/hh-ai-agent): агент ищет вакансии на HH.ru, оценивает через LLM, пишет сопроводительные и шлёт подходящие в Telegram.

Репозиторий: [google-dad/hh-ai-agent-advance](https://github.com/google-dad/hh-ai-agent-advance).

### Что добавлено в этом форке

- Универсальный пул API-ключей и команда `/keys` для **Mistral** и **OpenAI-compatible** (Together, Groq и т.п.); `/mistral_keys` — алиас
- Пример и рекомендации для [Together AI](https://www.together.ai/) / reasoning-моделей (`gpt-oss`)
- Если в `work_format` есть «удалённо» / `remote`, поиск HH идёт с `schedule=remote`
- Более мягкий LLM-отбор под SEO / linkbuilding / technical SEO / lead-роли
- Полные сопроводительные: запас `max_output_tokens`, обрезка по границе предложения (не на полуслове)

Секреты (`.env`, `profile.yaml`) в Git не входят — настраиваются локально.

## Быстрый старт

```bash
python setup_wizard.py
```

Wizard спросит:
1. Токен Telegram-бота и твой User ID
2. AI-провайдер (Ollama, Mistral или OpenAI-compatible)
3. Данные профиля для анализа вакансий
4. Режим работы

Создаст `.env` и `profile.yaml`, проверит конфигурацию.

**Изменить настройки позже:**

```bash
python setup_wizard.py --edit
```

На Windows удобнее `py -3.12 -m venv .venv`, затем `.venv\Scripts\activate` и `pip install -r requirements.txt`.

Проверки без полного цикла:

```bash
python main.py --check-config
python main.py --check-llm
python main.py
```

---

## Требования

- Python 3.11+ (рекомендуется 3.12)
- Telegram Bot ([@BotFather](https://t.me/BotFather))
- Один из LLM-провайдеров (ниже)
- [CloakBrowser](https://cloakbrowser.com/) (ставится через wizard)

---

## Режимы работы

| Режим | Описание |
|---|---|
| `dry_run` | Поиск и анализ, превью в Telegram — **без реальных откликов** |
| `approval` | Карточка с кнопкой «Откликнуться» — отклик только после твоего нажатия |

Начинай с `dry_run`. В Telegram: краткое обоснование, рейтинг компании и сворачиваемое письмо. Если rich messages недоступны — обычная HTML-карточка.

---

## Профиль и поиск HH

Ключевые поля в `profile.yaml` (локально, не в репозитории):

| Поле | Зачем |
|---|---|
| `search_queries` / `desired_positions` | Запросы и целевые роли для поиска и LLM |
| `areas` | Регионы HH — **числовые ID** (`1` = Москва, `2` = СПб). Пустой список = без фильтра по региону |
| `experience_filters` | Коды HH: `noExperience`, `between1And3`, `between3And6`, `moreThan6`. Пустой список = любой опыт |
| `work_format` | Если есть «удалённо» / `remote` → в URL поиска добавляется `schedule=remote` |
| `cover_letter.max_length` | Лимит символов письма (у HH поле до ~10 000; разумный рабочий диапазон 1500–8000) |

Поиск идёт по свежим публикациям (`order_by=publication_time`), глубина — `MAX_PAGES_PER_QUERY` / `MAX_VACANCIES_PER_QUERY` из `.env`.

---

## LLM-провайдеры

### Ollama (локально)

1. Установи [Ollama](https://ollama.com/download)
2. `ollama pull llama3`
3. В wizard выбери **Ollama**

`/keys` для Ollama недоступен — ключи не используются.

### Mistral API

1. Ключ на [console.mistral.ai](https://console.mistral.ai/)
2. В wizard — **Mistral API**

Нужен `LLM_KEYS_MASTER_KEY` (синоним `MISTRAL_KEYS_MASTER_KEY`) — Fernet-ключ для локального шифрования пула. Сохрани бэкап: без него старые ключи не расшифровать. Управление: `/keys`.

> ⚠️ Текст вакансий и профиль уходят во внешний API.

### OpenAI-compatible (Together, Groq, LM Studio, …)

Любой `/chat/completions`. Ключи тоже в зашифрованном пуле — `/keys`.

Пример Together AI:

```ini
LLM_PROVIDER=openai_compatible
LLM_MODEL=openai/gpt-oss-120b
OPENAI_COMPATIBLE_BASE_URL=https://api.together.xyz/v1
OPENAI_COMPATIBLE_API_KEY=your_key
LLM_KEYS_MASTER_KEY=your_fernet_master_key
LLM_TIMEOUT_SECONDS=90
LLM_MAX_OUTPUT_TOKENS=4000
```

У reasoning-моделей (`gpt-oss` и аналоги) часть `max_tokens` уходит во внутренние рассуждения — держи `LLM_MAX_OUTPUT_TOKENS` с запасом (для писем удобно ≥ 3500–4000). Healthcheck (`python main.py --check-llm`) выделяет до 256 токенов.

> ⚠️ Данные профиля и вакансий уходят к выбранному провайдеру.

---

## Сопроводительные письма

- Промпт просит **законченное** короткое письмо в пределах `cover_letter.max_length`
- Если ответ всё же длиннее лимита — обрезка по концу предложения / слова, без обрыва на «Ожидаемая…»
- Перед откликом всегда читай текст в Telegram

---

## Telegram-команды

| Команда | Описание |
|---|---|
| `/start` | Краткая справка |
| `/status` | Режим, состояние, статистика |
| `/pause` / `/resume` | Пауза / продолжение поиска |
| `/pending` | Вакансии, ожидающие решения |
| `/stats` | Статистика по статусам |
| `/diagnostics` | Последний цикл и circuit breaker |
| `/keys` | Пул API-ключей (Mistral / OpenAI-compatible); алиас `/mistral_keys` |
| `/cancel` | Отменить ввод CAPTCHA |

---

## Архитектура

| Файл | Ответственность |
|---|---|
| `config.py` | Валидация `.env` и `profile.yaml` |
| `browser_backend.py` | CloakBrowser / Playwright |
| `hh_client.py` | Поиск, страницы, отклики (`schedule=remote` при удалёнке) |
| `llm/` | Провайдеры, пул ключей, retry, квота |
| `ai_analyzer.py` | Оценка вакансий, генерация писем |
| `database.py` | SQLite, лимиты, статусы |
| `approval.py` | Единственный путь к реальному отклику |
| `tg_bot.py` | Команды, превью, inline-кнопки |
| `main.py` | Цикл агента |
| `setup_wizard.py` | Мастер настройки |
| `vacancy_filter.py` | Предфильтр по заголовку (stop-words) |

---

## Безопасность

- Реальный отклик только при **трёх** условиях: `APP_MODE=approval` + `ENABLE_REAL_APPLY=true` + кнопка твоим Telegram ID (permit ~30 минут)
- Массового авто-отклика нет
- `.env`, `profile.yaml`, `.browser-profile/` в `.gitignore`
- Токены и cookies в логи не пишутся

---

## Типичные ошибки

| Ошибка | Решение |
|---|---|
| `Configuration error` | `python setup_wizard.py --edit` |
| `CloakBrowser failed to start` | `python -m cloakbrowser info` или `BROWSER_BACKEND=playwright` |
| `HH.ru login is required` | `BROWSER_HEADLESS=false`, войти вручную |
| `LLM check failed` / пустой healthcheck | Проверь URL/ключ; для reasoning увеличь `LLM_MAX_OUTPUT_TOKENS` |
| `Invalid model response` | Вакансия пропускается без отклика |
| В Telegram тишина | Смотри логи: stop-words, `suitable=false`, уже обработанные в SQLite |

Сброс локальной истории вакансий (осторожно): останови агент, удали/переименуй `agent.db` (или сделай backup), запусти снова.

---

## Разработка

```bash
python -m compileall .
pytest -q
```

Тесты не ходят в HH.ru, Telegram и внешние LLM.

---

## Ограничения

- Автоматизация может нарушать правила HH.ru — риск на пользователе
- CloakBrowser не гарантирует отсутствие детекта / CAPTCHA
- Нет proxy, GeoIP-ротации и внешних CAPTCHA-сервисов
- Один владелец, одна SQLite-база
- Письмо нужно читать перед откликом

---

## Благодарности

Основа: **[fikstt2/hh-ai-agent](https://github.com/fikstt2/hh-ai-agent)**.

Спасибо **[kkonstantin08](https://github.com/kkonstantin08)** за архитектуру пайплайна и approval с permit-токенами, и **[danscMax](https://github.com/danscMax)** за проверки и валидацию конфигурации.

---

## Контакты

Вопросы по апстриму: **@fikstt3 (telegram)**

## Disclaimer

Автоматизация HH.ru нарушает пользовательское соглашение — используйте на свой страх и риск. Автор форка не несёт ответственности за ограничения аккаунта.
