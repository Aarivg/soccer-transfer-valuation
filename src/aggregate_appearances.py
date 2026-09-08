"""
aggregate_appearances.py — Aggregate per-match appearance rows into
season-level counting stats per player (games played, goals, assists,
minutes played, cards).

Source: appearances.csv from the Kaggle "player-scores" dataset
(https://www.kaggle.com/datasets/davidcariboo/player-scores) — one row
per player per game played, refreshed weekly. This is Transfermarkt data,
keyed by the same player_id used in players.csv / player_valuations.csv,
so it merges into the existing pipeline without fuzzy name matching.

NOTE: appearances.csv holds full historical data across every competition
Transfermarkt tracks — it is downloaded fresh each run but deliberately
NOT committed to git. Only this script's small per-player summary is.

Usage:
    python src/aggregate_appearances.py --season 2526
"""

import argparse
import os

import pandas as pd

# Big 5 league codes used by Transfermarkt / Kaggle
BIG5_CODES = {"GB1", "ES1", "L1", "IT1", "FR1"}

# Season start dates, used to filter appearances to the current season
SEASON_START = {
    "2526": "2025-07-01",
    "2425": "2024-07-01",
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate appearances.csv into per-player season stats"
    )
    parser.add_argument("--season", default="2526")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    raw_dir = args.data_dir or os.path.join(base, "data", "raw")
    proc_dir = os.path.join(base, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    path = os.path.join(raw_dir, "appearances.csv")
    if not os.path.exists(path):
        print(f"❌ appearances.csv not found at {path}")
        print("   Add the Kaggle download step for appearances.csv first.")
        return

    print(f"📂 Loading {path}")
    df = pd.read_csv(path)
    print(f"  → appearances.csv: {len(df)} rows")

    # ── Filter to current season ──────────────────────────────────────
    date_col = find_col(df, ["date", "appearance_date", "game_date"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        season_start = pd.Timestamp(SEASON_START.get(args.season, "2025-07-01"))
        before = len(df)
        df = df[df[date_col] >= season_start]
        print(f"  → Season filter ({season_start.date()}+): {before} → {len(df)}")
    else:
        print("  ⚠  No date column found — skipping season filter "
              "(stats will be all-time, not just this season)")

    # ── Filter to Big 5 leagues ────────────────────────────────────────
    comp_col = find_col(df, ["competition_id"])
    if comp_col:
        before = len(df)
        df = df[df[comp_col].astype(str).isin(BIG5_CODES)]
        print(f"  → Big 5 filter: {before} → {len(df)}")
    else:
        print("  ⚠  No competition_id column found — skipping Big 5 filter")

    id_col = find_col(df, ["player_id"])
    game_col = find_col(df, ["game_id"])
    if not id_col or not game_col:
        print("  ❌ Missing player_id or game_id column — cannot aggregate")
        print(f"     Columns present: {list(df.columns)}")
        return

    # ── Aggregate to one row per player ────────────────────────────────
    agg_map = {"games_played": (game_col, "count")}
    for stat, col_candidates in [
        ("goals", ["goals"]),
        ("assists", ["assists"]),
        ("minutes_played", ["minutes_played", "minutes"]),
        ("yellow_cards", ["yellow_cards"]),
        ("red_cards", ["red_cards"]),
    ]:
        col = find_col(df, col_candidates)
        if col:
            agg_map[stat] = (col, "sum")
        else:
            print(f"  ⚠  Column for '{stat}' not found — skipping")

    agg = df.groupby(id_col).agg(**agg_map).reset_index()
    agg = agg.rename(columns={id_col: "player_id"})

    if "goals" in agg.columns:
        agg["goals_per_game"] = (
            agg["goals"] / agg["games_played"]
        ).round(3)
    if "minutes_played" in agg.columns:
        agg["minutes_per_game"] = (
            agg["minutes_played"] / agg["games_played"]
        ).round(1)

    out_path = os.path.join(proc_dir, f"season_stats_{args.season}.csv")
    agg.to_csv(out_path, index=False)
    print(f"\n💾 Saved season stats for {len(agg)} players → {out_path}")
    print(f"   Columns: {list(agg.columns)}")


if __name__ == "__main__":
    main()
