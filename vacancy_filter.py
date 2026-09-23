DEFAULT_STOP_WORDS = (
    "стажер",
    "intern",
    "trainee",
    "дизайнер",
    "риелтор",
)

ROLE_KEYWORDS = (
    "seo",
    "сео",
    "линкбилд",
    "linkbuild",
    "pbn",
    "xrumer",
    "хрумер",
    "gsa",
    "дорве",
    "organic",
    "органик",
    "оптимизац",
    "поисков",
    "маркетолог",
)

BLOCK_WORDS = (
    "продаж",
    "facebook",
    "фейсбук",
    "instagram",
    "инстаграм",
    "tiktok",
    "тикток",
    "smm",
    "таргет",
    "контекст",
    "копирайт",
    "рекрутер",
    "бухгалтер",
    "юрист",
)

NO_ROLE_REASON = "no-role"


def title_rejection_reason(
    title: str, excluded_positions: tuple[str, ...]
) -> str | None:
    normalized = title.casefold()
    blocked = next(
        (
            word
            for word in (*DEFAULT_STOP_WORDS, *BLOCK_WORDS, *excluded_positions)
            if word.casefold() in normalized
        ),
        None,
    )
    if blocked:
        return blocked
    if any(word.casefold() in normalized for word in ROLE_KEYWORDS):
        return None
    return NO_ROLE_REASON
