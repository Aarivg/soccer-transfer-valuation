"""
scrape_fbref.py

Pulls current-season per-90 stats for outfield players across all Big 5
European leagues using the soccerdata library. Saves raw and processed CSVs.

Stats collected:
  - goals, assists, xG, xAG (expected goal involvement)
  - progressive passes, progressive carries
  - pressures
  - minutes played

Filters to players with >= 900 minutes played (roughly 10 full matches).
"""

import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SEASON = "2324"          # soccerdata season string; update to "2425" once data is available
MIN_MINUTES = 900

BIG5_LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga",
                "ITA-Serie A", "FRA-Ligue 1"]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# FBref stat categories we want and the columns to keep from each
STAT_TABLES = {
    "standard":     ["minutes_90s", "goals", "assists", "xg", "xg_assist"],
    "passing":      ["progressive_passes"],
    "possession":   ["progressive_carries"],
    "defense":      ["pressures"],
}

# Canonical rename map so columns are consistent regardless of soccerdata version
RENAME = {
    "minutes_90s":          "minutes_90s",
    "goals":                "goals",
    "assists":              "assists",
    "xg":                   "xg",
    "xg_assist":            "xag",
    "progressive_passes":   "progressive_passes",
    "progressive_carries":  "progressive_carries",
    "pressures":            "pressures",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _keep_cols(df: pd.DataFrame, want: list[str]) -> pd.DataFrame:
    """Return only the columns in *want* that actually exist in df."""
    present = [c for c in want if c in df.columns]
    missing = set(want) - set(present)
    if missing:
        log.warning("Columns not found in this table: %s", missing)
    return df[present]


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """soccerdata sometimes returns MultiIndex columns — flatten to strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, c))).strip("_")
                      for c in df.columns]
    return df


# ── Main scrape ───────────────────────────────────────────────────────────────

def fetch_fbref_stats(season: str = SEASON,
                      leagues: list[str] | None = None,
                      min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """
    Fetch and merge FBref stat tables for the given season and leagues.

    Returns a tidy DataFrame with one row per (league, player, team) and
    per-90 versions of every counting stat.
    """
    if leagues is None:
        leagues = BIG5_LEAGUES

    fbref = sd.FBref(leagues=leagues, seasons=season)

    frames: list[pd.DataFrame] = []

    for table_name, cols in STAT_TABLES.items():
        log.info("Fetching FBref '%s' table …", table_name)
        try:
            raw = fbref.read_player_season_stats(stat_type=table_name)
        except Exception as exc:
            log.error("Failed to fetch '%s': %s", table_name, exc)
            continue

        raw = _flatten_multiindex(raw)
        raw = raw.reset_index()  # brings league/player/team out of MultiIndex index

        present = _keep_cols(raw, cols + ["league", "player", "team"])
        frames.append(present)

    if not frames:
        raise RuntimeError("No data fetched — check league names and season string.")

    # Merge all tables on shared identity columns
    identity = ["league", "player", "team"]
    merged = frames[0]
    for frame in frames[1:]:
        on_cols = [c for c in identity if c in frame.columns]
        merged = merged.merge(frame, on=on_cols, how="outer")

    merged = merged.rename(columns=RENAME)

    # Derive minutes played from minutes_90s (FBref stores as 90-min units)
    if "minutes_90s" in merged.columns:
        merged["minutes_played"] = (merged["minutes_90s"] * 90).round().astype("Int64")

    # Filter by minimum minutes
    if "minutes_played" in merged.columns:
        before = len(merged)
        merged = merged[merged["minutes_played"] >= min_minutes].copy()
        log.info("Filtered %d → %d players (min %d min)", before, len(merged), min_minutes)

    # Convert counting stats to per-90 rates
    per90_cols = ["goals", "assists", "xg", "xag",
                  "progressive_passes", "progressive_carries", "pressures"]
    for col in per90_cols:
        if col in merged.columns and "minutes_90s" in merged.columns:
            merged[f"{col}_per90"] = (merged[col] / merged["minutes_90s"]).round(3)

    merged = merged.sort_values(["league", "player"]).reset_index(drop=True)
    return merged


# ── I/O ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"fbref_big5_{SEASON}_raw.csv"
    df.to_csv(raw_path, index=False)
    log.info("Saved raw data → %s (%d rows)", raw_path, len(df))

    per90_cols = [c for c in df.columns if c.endswith("_per90")]
    keep = ["league", "player", "team", "minutes_played"] + per90_cols
    keep = [c for c in keep if c in df.columns]
    processed_path = PROCESSED_DIR / f"fbref_big5_{SEASON}_per90.csv"
    df[keep].to_csv(processed_path, index=False)
    log.info("Saved processed (per-90) data → %s", processed_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_fbref_stats()
    save(df)
    print(df[["league", "player", "team", "minutes_played"]].head(20).to_string(index=False))
