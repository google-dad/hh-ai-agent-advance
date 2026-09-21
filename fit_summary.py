from __future__ import annotations

import re


FIT_CATEGORIES = frozenset(("Опыт", "Навыки", "Задачи", "Формат", "Локация"))
FIT_SUMMARY_FALLBACK = "Цель: вакансия соответствует моему направлению развития."
_MARKDOWN = re.compile(
    r"```|`|!?\[[^\]]*\]\([^)]+\)|"
    r"^\s{0,3}(?:#{1,6}|>|[-+*]|\d+[.)])\s+|"
    r"(?:\*{1,3}|_{1,3}|~{2})\S(?:.*?\S)?(?:\*{1,3}|_{1,3}|~{2})",
    re.MULTILINE,
)


def normalize_fit_summary(raw: object) -> str:
    if not isinstance(raw, list):
        return FIT_SUMMARY_FALLBACK
    lines: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category, text = item.get("category"), item.get("text")
        if (
            not isinstance(category, str)
            or category not in FIT_CATEGORIES
            or category in seen
            or not isinstance(text, str)
            or _MARKDOWN.search(text)
        ):
            continue
        text = " ".join(text.split())
        if not text or len(text) > 140:
            continue
        lines.append(f"{category}: {text}")
        seen.add(category)
        if len(lines) == 4:
            break
    if not lines:
        return FIT_SUMMARY_FALLBACK
    if len(lines) == 1:
        lines.append(FIT_SUMMARY_FALLBACK)
    return "\n".join(lines)
