"""Shared season mapping utilities."""

SEASON_MAP = {
    12: "Winter (DJF)",
    1: "Winter (DJF)",
    2: "Winter (DJF)",
    3: "Spring (MAM)",
    4: "Spring (MAM)",
    5: "Spring (MAM)",
    6: "Summer (JJA)",
    7: "Summer (JJA)",
    8: "Summer (JJA)",
    9: "Fall (SON)",
    10: "Fall (SON)",
    11: "Fall (SON)",
}

SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]

SEASON_MAP_SHORT = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}


def get_season(month: int) -> str:
    """Return long season name for calendar month."""
    return SEASON_MAP[int(month)]


def get_season_months(season_name: str) -> list[int]:
    """Return list of month numbers for a long or short season label."""
    normalized = season_name.strip().lower()
    short_to_long = {
        "winter": "Winter (DJF)",
        "spring": "Spring (MAM)",
        "summer": "Summer (JJA)",
        "fall": "Fall (SON)",
    }
    if normalized in short_to_long:
        target = short_to_long[normalized]
    else:
        target = season_name

    return [month for month, season in SEASON_MAP.items() if season == target]
