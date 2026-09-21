DEFAULT_STOP_WORDS = (
    "стажер",
    "intern",
    "trainee",
    "дизайнер",
    "риелтор",
)


def title_rejection_reason(
    title: str, excluded_positions: tuple[str, ...]
) -> str | None:
    normalized = title.casefold()
    return next(
        (
            word
            for word in (*DEFAULT_STOP_WORDS, *excluded_positions)
            if word.casefold() in normalized
        ),
        None,
    )
