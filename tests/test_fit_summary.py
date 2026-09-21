import pytest

from fit_summary import FIT_SUMMARY_FALLBACK, normalize_fit_summary


def test_valid_points_are_rendered() -> None:
    assert normalize_fit_summary(
        [
            {"category": "Опыт", "text": "backend-сервисы."},
            {"category": "Навыки", "text": "Python и REST API."},
        ]
    ) == "Опыт: backend-сервисы.\nНавыки: Python и REST API."


@pytest.mark.parametrize("raw", [None, "text", {}, [None]])
def test_invalid_points_use_neutral_fallback(raw: object) -> None:
    assert normalize_fit_summary(raw) == FIT_SUMMARY_FALLBACK


def test_untrusted_markdown_duplicates_and_long_points_are_dropped() -> None:
    assert normalize_fit_summary(
        [
            {"category": "Опыт", "text": "backend-сервисы"},
            {"category": "Опыт", "text": "duplicate"},
            {"category": "Навыки", "text": "**Python**"},
            {"category": "Задачи", "text": "x" * 141},
            {"category": {}, "text": "unhashable input"},
        ]
    ) == f"Опыт: backend-сервисы\n{FIT_SUMMARY_FALLBACK}"
