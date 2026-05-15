"""Web scraping functions for ONI data."""

import pandas as pd
import requests
from bs4 import BeautifulSoup

from cno_project.config import ONI_URL


SEASON_TO_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}


def scrape_oni_with_bs4(url=ONI_URL):
    """Scrape Oceanic Niño Index data from NOAA."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    rows = []

    for line in text.splitlines():
        parts = line.strip().split()

        if len(parts) != 4:
            continue

        season, year, total, oni = parts

        if season not in SEASON_TO_MONTH:
            continue

        month = SEASON_TO_MONTH[season]

        rows.append(
            {
                "date": pd.to_datetime(f"{year}-{month:02d}-01"),
                "season": season,
                "year": int(year),
                "month": month,
                "total": float(total),
                "oni": float(oni),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)