"""
scrape_fbref.py — Pull player-level stats from FBref via soccerdata.

Fetches multiple stat tables (standard, shooting, passing, possession,
defense, misc) for the Big 5 European Leagues and merges them into a
single DataFrame. Saves to data/raw/fbref_stats_SEASON.csv.

Usage:
    python src/scrape_fbref.py                  # default: 2024-25
    python src/scrape_fbref.py --season 2324    # specific season
"""

import argparse
import os
import time
import pandas as pd
import soccerdata as sd

# ── Configuration ────────────────────────────────────────────────────
STAT_TYPES = [
    "standard",
    "shooting",
    "passing",
    "possession",
    "defense",
    "misc",
]

# Columns to keep from each stat table (per-90 where available)
COLUMNS_TO_KEEP = {
    "standard": [
        "player", "nation", "pos", "squad", "comp", "age", "born",
        "minutes_90s",       # 90-minute units played
        "goals_per90", "assists_per90",
        "goals_assists_per90",
        "goals_pens_per90",  # non-penalty goals per 90
        "xg_per90", "xg_assist_per90",  # xG and xAG per 90
        "npxg_per90",        # non-penalty xG per 90
        "minutes",
    ],
    "shooting": [
        "player", "squad",
        "shots_per90", "shots_on_target_per90",
        "goals_per_shot", "goals_per_shot_on_target",
        "npxg_per_shot",
        "xg_net_per90",      # xG overperformance
    ],
    "passing": [
        "player", "squad",
        "passes_completed", "passes", "passes_pct",
        "progressive_passes",
        "passes_into_final_third",
        "passes_into_penalty_area",
        "assisted_shots",    # key passes
    ],
    "possession": [
        "player", "squad",
        "progressive_carries",
        "carries_into_final_third",
        "carries_into_penalty_area",
        "successful_dribbles", "dribbles",
        "progressive_passes_received",
    ],
    "defense": [
        "player", "squad",
        "tackles_won", "tackles",
        "interceptions",
        "blocks",
        "clearances",
        "aerials_won", "aerials_lost",
    ],
    "misc": [
        "player", "squad",
        "fouls", "fouled",
        "offsides",
        "ball_recoveries",
    ],
}

# ── Helper: safe column select ───────────────────────────────────────
def safe_select(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Select columns that exist, silently skip missing ones."""
    available = [c for c in wanted if c in df.columns]
    missing = set(wanted) - set(available)
    if missing:
        print(f"  ⚠  Missing columns (skipped): {missing}")
    return df[available]


# ── Main scraper ─────────────────────────────────────────────────────
def scrape_fbref(season: str = "2425") -> pd.DataFrame:
    """
    Pull all stat tables for Big 5 leagues from FBref and merge.

    Parameters
    ----------
    season : str
        Season identifier, e.g. "2425" for 2024-25.

    Returns
    -------
    pd.DataFrame
        One row per player with all stats merged.
    """
    print(f"📡 Fetching FBref data for season {season}...")
    print("   (First run may take a few minutes — data is cached after.)\n")

    fbref = sd.FBref(
        leagues="Big 5 European Leagues Combined",
        seasons=season,
    )

    merged = None

    for stat in STAT_TYPES:
        print(f"  → Fetching: {stat}...")
        try:
            df = fbref.read_player_season_stats(stat_type=stat)
            df = df.reset_index()

            # Flatten multi-level column names if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(str(c) for c in col).strip("_")
                              for col in df.columns]

            # Lowercase all column names for consistency
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            print(f"    ✓ {len(df)} rows, {len(df.columns)} columns")

            if merged is None:
                merged = df
            else:
                # Merge on player + squad to avoid duplicate columns
                merge_keys = ["player", "squad"]
                # Drop columns from right df that already exist (except keys)
                existing_cols = set(merged.columns) - set(merge_keys)
                new_cols = [c for c in df.columns
                            if c not in existing_cols or c in merge_keys]
                merged = merged.merge(
                    df[new_cols],
                    on=merge_keys,
                    how="outer",
                    suffixes=("", f"_{stat}"),
                )

            # Be polite to FBref
            time.sleep(4)

        except Exception as e:
            print(f"    ✗ Error fetching {stat}: {e}")
            continue

    if merged is None:
        raise RuntimeError("Failed to fetch any stat tables from FBref.")

    # ── Filter: outfield players with ≥900 minutes ──────────────────
    min_col = None
    for candidate in ["minutes", "minutes_90s", "min"]:
        if candidate in merged.columns:
            min_col = candidate
            break

    if min_col and min_col != "minutes":
        # Convert 90s-based to total minutes
        if "90" in min_col:
            merged["minutes"] = pd.to_numeric(
                merged[min_col], errors="coerce"
            ) * 90
        else:
            merged["minutes"] = pd.to_numeric(
                merged[min_col], errors="coerce"
            )

    if "minutes" in merged.columns:
        before = len(merged)
        merged = merged[
            pd.to_numeric(merged["minutes"], errors="coerce") >= 900
        ]
        print(f"\n  🔽 Filtered ≥900 min: {before} → {len(merged)} players")

    # Drop goalkeepers
    if "pos" in merged.columns:
        before = len(merged)
        merged = merged[~merged["pos"].str.contains("GK", na=False)]
        print(f"  🔽 Dropped GKs: {before} → {len(merged)} players")

    print(f"\n✅ Final dataset: {len(merged)} players, {len(merged.columns)} features")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Scrape FBref player stats")
    parser.add_argument(
        "--season", default="2425",
        help='Season code, e.g. "2425" for 2024-25 (default: 2425)'
    )
    args = parser.parse_args()

    df = scrape_fbref(season=args.season)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fbref_stats_{args.season}.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
