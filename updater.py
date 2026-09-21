from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from version import __version__, is_newer_version

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO = "fikstt2/hh-ai-agent"


@dataclass(frozen=True)
class ReleaseInfo:
    tag_name: str
    name: str
    html_url: str
    published_at: str
    body: str


async def check_github_release(
    *,
    repo: str = DEFAULT_GITHUB_REPO,
    current_version: str = __version__,
    session: Any | None = None,
    timeout_seconds: float = 5.0,
) -> ReleaseInfo | None:
    """Асинхронно проверяет наличие свежего релиза на GitHub."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": f"HH-Agent/{current_version}",
    }
    owns_session = session is None
    http_session = session or aiohttp.ClientSession()
    try:
        async with http_session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            if response.status != 200:
                logger.debug("github_release_check_status status=%s", response.status)
                return None
            data = await response.json()
            if not isinstance(data, dict):
                return None
            tag_name = str(data.get("tag_name", "")).strip()
            if not tag_name or not is_newer_version(tag_name, current_version):
                return None
            return ReleaseInfo(
                tag_name=tag_name,
                name=str(data.get("name", "")).strip(),
                html_url=str(data.get("html_url", "")).strip(),
                published_at=str(data.get("published_at", "")).strip(),
                body=str(data.get("body", "")).strip(),
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
        logger.debug("github_release_check_failed error=%s", exc)
        return None
    except Exception as exc:
        logger.warning("github_release_check_unexpected_error error=%s", exc)
        return None
    finally:
        if owns_session:
            await http_session.close()
