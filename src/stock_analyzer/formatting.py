_SUFFIXES = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]


def format_large_number(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "N/A"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for threshold, suffix in _SUFFIXES:
        if magnitude >= threshold:
            return f"{sign}${magnitude / threshold:.2f}{suffix}"
    return f"{sign}${magnitude:.2f}"
