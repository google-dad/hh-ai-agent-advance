import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from approval import ApprovalGuard
from config import load_settings
from database import Database
from hh_client import HHClient, PageState, VacancySummary, classify_page
from tests.test_config import VALID_ENV, write_profile
from vacancy_filter import title_rejection_reason


class FakeLocator:
    def __init__(self, visible: bool):
        self.visible = visible
        self.first = self

    async def is_visible(self) -> bool:
        return self.visible

    async def inner_text(self) -> str:
        return ""


class FakePage:
    def __init__(self, visible_selector: str = "", broken: bool = False):
        self.visible_selector = visible_selector
        self.broken = broken

    def locator(self, selector: str) -> FakeLocator:
        if self.broken:
            raise RuntimeError("page closed")
        return FakeLocator(selector == self.visible_selector)

    async def goto(self, url: str, **kwargs) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeContext:
    def __init__(self, page: FakePage):
        self.page = page

    async def new_page(self) -> FakePage:
        return self.page


class SearchCard:
    def __init__(self, href: str, title: str):
        self.href = href
        self.title = title

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None

    async def inner_text(self) -> str:
        return self.title


class TextLocator(FakeLocator):
    def __init__(self, text: str = "", href: str | None = None):
        super().__init__(bool(text))
        self.text = text
        self.href = href

    async def inner_text(self) -> str:
        return self.text

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None


class SearchLocator:
    def __init__(self, cards: list[SearchCard]):
        self.cards = cards

    async def all(self) -> list[SearchCard]:
        return self.cards


class SearchPage(FakePage):
    def __init__(self, cards: list[SearchCard] | None = None):
        super().__init__()
        self.cards = cards or []

    def locator(self, selector: str) -> SearchLocator:
        return SearchLocator(self.cards)


class VacancyPage(FakePage):
    def locator(self, selector: str) -> TextLocator:
        values = {
            '[data-qa="vacancy-description"]': "Описание вакансии.",
            '[data-qa="vacancy-company-name"]': "Компания",
        }
        return TextLocator(
            values.get(selector, ""),
            "/employer/123?hhtmFrom=vacancy"
            if selector == '[data-qa="vacancy-company-name"]'
            else None,
        )


class EmployerPage(FakePage):
    def __init__(self, rating: str = "", reviews: str = ""):
        super().__init__()
        self.values = {
            '[data-qa="employer-review-small-widget-total-rating"]': rating,
            '[data-qa="employer-review-small-widget-review-count-action"]': reviews,
        }

    def locator(self, selector: str) -> TextLocator:
        return TextLocator(self.values.get(selector, ""))


class FailingContext:
    async def new_page(self):
        raise RuntimeError("page creation failed")


class RetryLoginPage(FakePage):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def goto(self, url: str, **kwargs) -> None:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("temporary navigation failure")


class CaptchaLocator:
    def __init__(self, page: "CaptchaPage"):
        self.page = page
        self.first = self

    async def is_visible(self) -> bool:
        return True

    async def fill(self, value: str) -> None:
        self.page.solution = value

    async def press(self, key: str) -> None:
        self.page.actions.append(f"press:{key}")

    async def click(self) -> None:
        self.page.actions.append("click:submit")
        self.page.solved = True


class CaptchaPage:
    def __init__(self):
        self.actions: list[str] = []
        self.solution = ""
        self.solved = False

    def locator(self, selector: str):
        if selector == 'input[type="text"]' or "button" in selector:
            return CaptchaLocator(self)
        return FakeLocator(
            (self.solved and selector == '[data-qa="vacancy-description"]')
            or (not self.solved and selector == 'form[action*="captcha"]')
        )

    async def screenshot(self, *, path: Path) -> None:
        path.touch()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ('[data-qa="vacancy-description"]', PageState.VACANCY_LOADED),
        ('form[action*="captcha"]', PageState.CAPTCHA_DETECTED),
        ('[data-qa="access-denied"]', PageState.ACCESS_DENIED),
        ('[data-qa="vacancy-removed"]', PageState.VACANCY_REMOVED),
        ("", PageState.PAGE_STRUCTURE_CHANGED),
    ],
)
def test_page_state_uses_explicit_signals(selector: str, expected: PageState) -> None:
    assert asyncio.run(classify_page(FakePage(selector))) is expected


def test_page_state_reports_network_error() -> None:
    assert asyncio.run(classify_page(FakePage(broken=True))) is PageState.NETWORK_ERROR


def test_visible_but_empty_description_reports_structure_change(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(FakePage('[data-qa="vacancy-description"]')),
        settings,
        database,
        ApprovalGuard(settings, database),
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.PAGE_STRUCTURE_CHANGED


def test_page_creation_failure_reports_network_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database)
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.NETWORK_ERROR
    assert "page creation failed" in result.error


def test_search_page_creation_failure_returns_safe_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database)
    )

    result = asyncio.run(client.search_vacancies("Python", (), ()))

    assert result.summaries == []
    assert result.error_reason == "search_error:RuntimeError"


def test_search_ignores_ad_redirects_without_numeric_vacancy_id(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    page = SearchPage(
        [
            SearchCard(
                "https://adsrv.hh.ru/click?clickType=link_to_vacancy",
                "Ad",
            ),
            SearchCard("https://hh.ru/vacancy/135006927?from=search", "DevOps"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.title) for item in result.summaries] == [
        ("135006927", "DevOps")
    ]


def test_search_counts_found_new_and_existing_results(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    assert database.discover(
        job_id="135006927",
        title="Existing",
        company="Example",
        url="https://hh.ru/vacancy/135006927",
        description_hash="hash",
        search_query="DevOps",
        discovered_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Existing"),
            SearchCard("https://hh.ru/vacancy/135006928", "New"),
            SearchCard("https://example.com/ad", "Ad"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert result.found_results == 2
    assert result.duplicates == 1
    assert [item.id for item in result.summaries] == ["135006928"]


def test_search_returns_only_pending_existing_vacancies_as_repeats(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    now = datetime(2026, 7, 30, tzinfo=UTC)
    for job_id in ("135006927", "135006928"):
        assert database.discover(
            job_id=job_id,
            title="Existing",
            company="Example",
            url=f"https://hh.ru/vacancy/{job_id}",
            description_hash="hash",
            search_query="DevOps",
            discovered_at=now,
        )
        assert database.request_approval(
            job_id=job_id,
            cover_letter="Letter",
            llm_decision=True,
            llm_reason="Relevant",
            confidence=0.9,
            now=now,
        )
    assert database.skip("135006928", 42, 42)
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Pending"),
            SearchCard("https://hh.ru/vacancy/135006928", "Skipped"),
            SearchCard("https://hh.ru/vacancy/135006929", "New"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.previously_sent) for item in result.summaries] == [
        ("135006927", True),
        ("135006929", False),
    ]
    assert result.duplicates == 2


def test_read_vacancy_collects_safe_company_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(VacancyPage()),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )

    result = asyncio.run(
        client.read_vacancy(
            VacancySummary("job-1", "Developer", "https://hh.ru/vacancy/1", "Python")
        )
    )

    assert result.company_url == "https://hh.ru/employer/123"


def test_read_company_details_parses_rating_and_review_count(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("3,4", "14 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )

    result = asyncio.run(client.read_company_details("https://hh.ru/employer/123"))

    assert result.rating == 3.4
    assert result.reviews_count == 14


def test_read_company_details_rejects_non_hh_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("5", "100 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
    )

    result = asyncio.run(client.read_company_details("https://example.com/employer/1"))

    assert result.rating is None
    assert result.reviews_count is None


def test_login_check_retries_temporary_navigation_errors(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = RetryLoginPage()

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    assert asyncio.run(client.ensure_login())
    assert page.attempts == 3


def test_captcha_solution_uses_submit_button(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = CaptchaPage()
    client = HHClient(
        FakeContext(page), settings, database, ApprovalGuard(settings, database)
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    async def solve(*_args):
        return "abcd"

    result = asyncio.run(client._solve_captcha(page, summary, solve))

    assert result is PageState.VACANCY_LOADED
    assert page.solution == "abcd"
    assert page.actions == ["click:submit"]


@pytest.mark.parametrize(
    ("title", "excluded", "expected"),
    [
        ("Intern Python developer", (), "intern"),
        ("Python sales engineer", ("sales",), "sales"),
        ("Senior SEO Team Lead", (), None),
        ("Python developer", (), "no-role"),
        ("Менеджер по продажам", (), "продаж"),
        ("Facebook маркетолог", (), "facebook"),
        ("Интернет-маркетолог", (), None),
        ("маркетолог по продажам", (), "продаж"),
        ("линкбилдер", (), None),
    ],
)
def test_title_filter_returns_the_matched_reason(
    title: str, excluded: tuple[str, ...], expected: str | None
) -> None:
    assert title_rejection_reason(title, excluded) == expected
