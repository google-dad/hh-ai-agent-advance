from __future__ import annotations

import re

__version__ = "1.2.6"


def parse_version(version_str: str) -> tuple[int, ...]:
    """Разбирает строку версии вида 'v1.2.5' или '1.2.6-beta' в кортеж чисел (1, 2, 6)."""
    clean = re.sub(r"^[^\d]*", "", version_str.strip())
    parts: list[int] = []
    for part in re.split(r"[^\d]+", clean):
        if part:
            try:
                parts.append(int(part))
            except ValueError:
                break
    return tuple(parts) if parts else (0,)


def is_newer_version(latest_str: str, current_str: str = __version__) -> bool:
    """Проверяет, является ли latest_str более новой версией, чем current_str."""
    latest = parse_version(latest_str)
    current = parse_version(current_str)
    return latest > current
