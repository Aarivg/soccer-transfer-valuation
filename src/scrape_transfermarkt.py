"""
scrape_transfermarkt.py — Load Transfermarkt data from Kaggle CSVs.

Reads player_valuations.csv and players.csv from the Kaggle dataset
(https://www.kaggle.com/datasets/davidcariboo/player-scores), extracts
the most recent valuation per player, and enriches with player metadata.

Usage:
    python src/scrape_transfermarkt.py
    python src/scrape_transfermarkt.py --data-dir ./data/raw/kaggle
"""

import argparse
import os
import glob
import pandas as pd


def find_csv(data_dir: str, name: str) -> str:
    """Find a CSV file by name in the data directory (recursive)."""
    patterns = [
        os.path.join(data_dir, name),
        os.path.join(data_dir, "**", name),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find '{name}' in {data_dir}. "
        f"Download from: kaggle.com/datasets/davidcariboo/player-scores"
    )


def load_transfermarkt(data_dir: str = None) -> pd.DataFrame:
    """
    Load Transfermarkt valuations and player metadata.

    Parameters
    ----------
    data_dir : str
        Directory containing Kaggle CSVs. Searches in common locations
        if not specified.

    Returns
    -------
    pd.DataFrame
        One row per player with latest market value + metadata.
    """
    # Search common locations
    if data_dir is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
            os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kaggle"),
            os.path.join(os.path.dirname(__file__), ".."),
            ".",
        ]
        for c in candidates:
            try:
                find_csv(c, "player_valuations.csv")
                data_dir = c
                break
            except FileNotFoundError:
                continue
        if data_dir is None:
            raise FileNotFoundError(
                "Cannot find Kaggle CSVs. Provide --data-dir or place files "
                "in data/raw/"
            )

    print(f"📂 Loading Transfermarkt data from: {data_dir}")

    # ── Load valuations ──────────────────────────────────────────────
    val_path = find_csv(data_dir, "player_valuations.csv")
    vals = pd.read_csv(val_path)
    print(f"  → player_valuations.csv: {len(vals)} rows")

    # Parse dates
    if "date" in vals.columns:
        vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    elif "datetime" in vals.columns:
        vals["date"] = pd.to_datetime(vals["datetime"], errors="coerce")

    # Get the most recent valuation per player
    vals = vals.sort_values("date", ascending=False)
    latest_vals = vals.drop_duplicates(subset=["player_id"], keep="first")
    print(f"  → Latest valuations: {len(latest_vals)} unique players")

    # ── Load player metadata ─────────────────────────────────────────
    players_path = find_csv(data_dir, "players.csv")
    players = pd.read_csv(players_path)
    print(f"  → players.csv: {len(players)} rows")

    # ── Merge ────────────────────────────────────────────────────────
    # Identify the player ID column
    pid_col = "player_id"
    if pid_col not in players.columns and "id" in players.columns:
        players = players.rename(columns={"id": "player_id"})

    merged = latest_vals.merge(
        players,
        on="player_id",
        how="left",
        suffixes=("_val", "_player"),
    )

    # ── Clean up key columns ─────────────────────────────────────────
    # Market value
    value_col = None
    for candidate in ["market_value_in_eur", "market_value", "value"]:
        if candidate in merged.columns:
            value_col = candidate
            break
    if value_col and value_col != "market_value_eur":
        merged = merged.rename(columns={value_col: "market_value_eur"})

    # Player name
    name_col = None
    for candidate in ["name", "player_name", "pretty_name"]:
        if candidate in merged.columns:
            name_col = candidate
            break
    if name_col and name_col != "player_name":
        merged = merged.rename(columns={name_col: "player_name"})

    # Date of birth → age
    for dob_col in ["date_of_birth", "dob", "birth_date"]:
        if dob_col in merged.columns:
            merged[dob_col] = pd.to_datetime(merged[dob_col], errors="coerce")
            merged["age"] = (
                (pd.Timestamp.now() - merged[dob_col]).dt.days / 365.25
            ).round(1)
            break

    # Contract expiry → years remaining
    for contract_col in ["contract_expiration_date", "contract_expires"]:
        if contract_col in merged.columns:
            merged[contract_col] = pd.to_datetime(
                merged[contract_col], errors="coerce"
            )
            merged["contract_years_remaining"] = (
                (merged[contract_col] - pd.Timestamp.now()).dt.days / 365.25
            ).clip(lower=0).round(2)
            break

    # ── Filter to Big 5 leagues ──────────────────────────────────────
    big5_keywords = [
        "premier league", "la liga", "1. bundesliga", "bundesliga",
        "serie a", "ligue 1",
        "GB1", "ES1", "L1", "IT1", "FR1",  # Transfermarkt league codes
    ]
    league_col = None
    for candidate in ["current_club_domestic_competition_id",
                       "domestic_competition_id", "league", "comp"]:
        if candidate in merged.columns:
            league_col = candidate
            break

    if league_col:
        mask = merged[league_col].astype(str).str.lower().apply(
            lambda x: any(kw in x.lower() for kw in big5_keywords)
        )
        before = len(merged)
        merged = merged[mask]
        print(f"  → Big 5 filter: {before} → {len(merged)} players")

    # Drop players with no market value
    merged = merged.dropna(subset=["market_value_eur"])
    merged = merged[merged["market_value_eur"] > 0]

    print(f"\n✅ Final: {len(merged)} players with market values")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Load Transfermarkt data from Kaggle CSVs"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing player_valuations.csv and players.csv"
    )
    args = parser.parse_args()

    df = load_transfermarkt(data_dir=args.data_dir)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tm_valuations_latest.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
