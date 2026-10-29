"""
scrape_transfermarkt.py — Build the one-row-per-player Transfermarkt table.

Primary source: data/raw/tm_squads_current.csv, the live Big 5 squads
scraped by scrape_tm_squads.py (current clubs, values and contracts).
Background fields (caps, birthplace, peak value…) still come from the Kaggle
players.csv.

Fallback: the Kaggle dataset alone
(https://www.kaggle.com/datasets/davidcariboo/player-scores), which stopped
updating in July 2026 — latest valuation per player plus metadata.

Usage:
    python src/scrape_transfermarkt.py
    python src/scrape_transfermarkt.py --data-dir ./data/raw
"""

import argparse
import os
import glob
import pandas as pd


# Big 5 league codes used by Transfermarkt / Kaggle
BIG5_CODES = {"GB1", "ES1", "L1", "IT1", "FR1"}


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

    Returns one row per player with latest market value + metadata.
    """
    # Search common locations
    if data_dir is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
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
    date_col = "date" if "date" in vals.columns else "datetime"
    vals["date"] = pd.to_datetime(vals[date_col], errors="coerce")

    # Get the most recent valuation per player
    vals = vals.sort_values("date", ascending=False)
    latest_vals = vals.drop_duplicates(subset=["player_id"], keep="first")
    print(f"  → Latest valuations: {len(latest_vals)} unique players")

    # ── Load player metadata ─────────────────────────────────────────
    players_path = find_csv(data_dir, "players.csv")
    players = pd.read_csv(players_path)
    print(f"  → players.csv: {len(players)} rows")

    # ── Merge ────────────────────────────────────────────────────────
    merged = latest_vals.merge(
        players,
        on="player_id",
        how="left",
        suffixes=("_val", "_player"),
    )

    # ── Filter to Big 5 leagues ──────────────────────────────────────
    # Try multiple possible column names for league code
    league_col = None
    for candidate in [
        "current_club_domestic_competition_id",
        "player_club_domestic_competition_id",
    ]:
        if candidate in merged.columns:
            league_col = candidate
            break

    if league_col:
        before = len(merged)
        merged = merged[
            merged[league_col].astype(str).isin(BIG5_CODES)
        ]
        print(f"  → Big 5 filter: {before} → {len(merged)} players")
    else:
        print("  ⚠  No league column found — skipping Big 5 filter")

    # ── Rename market value column ───────────────────────────────────
    # Kaggle uses 'market_value_in_eur' — standardize to 'market_value_eur'
    mv_col = None
    for candidate in ["market_value_in_eur_val", "market_value_in_eur",
                       "market_value_eur"]:
        if candidate in merged.columns:
            mv_col = candidate
            break

    if mv_col and mv_col != "market_value_eur":
        merged = merged.rename(columns={mv_col: "market_value_eur"})

    # Drop players with no market value
    merged = merged.dropna(subset=["market_value_eur"])
    merged = merged[merged["market_value_eur"] > 0]

    # ── Player name ──────────────────────────────────────────────────
    if "name" in merged.columns and "player_name" not in merged.columns:
        merged = merged.rename(columns={"name": "player_name"})

    # ── Date of birth → age ──────────────────────────────────────────
    if "date_of_birth" in merged.columns:
        merged["date_of_birth"] = pd.to_datetime(
            merged["date_of_birth"], errors="coerce"
        )
        merged["age"] = (
            (pd.Timestamp.now() - merged["date_of_birth"]).dt.days / 365.25
        ).round(1)

    # ── Contract expiry → years remaining ────────────────────────────
    if "contract_expiration_date" in merged.columns:
        merged["contract_expiration_date"] = pd.to_datetime(
            merged["contract_expiration_date"], errors="coerce"
        )
        merged["contract_years_remaining"] = (
            (merged["contract_expiration_date"] - pd.Timestamp.now()).dt.days
            / 365.25
        ).clip(lower=0).round(2)

    print(f"\n✅ Final: {len(merged)} players with market values")
    return merged


# ── Live squads (scrape_tm_squads.py) ────────────────────────────────
# Fields that come from the live Transfermarkt scrape and replace the
# frozen Kaggle values. Everything else (birthplace, caps, agent, peak
# value…) is still taken from Kaggle's players.csv where available.
LIVE_FIELDS = [
    "player_name", "position", "sub_position", "date_of_birth",
    "country_of_citizenship", "height_in_cm", "foot",
    "contract_expiration_date", "image_url", "url",
    "current_club_id", "current_club_name",
    "current_club_domestic_competition_id",
]


def load_live_squads(data_dir: str) -> pd.DataFrame | None:
    """
    Build the same one-row-per-player table as load_transfermarkt(), but
    from the current Big 5 squads scraped by scrape_tm_squads.py, so new
    transfers, promoted clubs and new valuations are reflected. Returns None
    if no scrape is available.
    """
    path = os.path.join(data_dir, "tm_squads_current.csv")
    if not os.path.exists(path):
        return None
    squads = pd.read_csv(path)
    squads = squads.dropna(subset=["market_value_in_eur"])
    squads = squads[squads["market_value_in_eur"] > 0]
    as_of = squads["scraped_at"].max()
    print(f"📂 Live Transfermarkt squads: {len(squads)} valued players "
          f"(scraped {as_of})")

    # Background metadata from the (frozen) Kaggle players.csv
    try:
        players = pd.read_csv(find_csv(data_dir, "players.csv"))
        extra = [c for c in players.columns
                 if c not in LIVE_FIELDS and c not in
                 ("name", "market_value_in_eur", "last_season")]
        squads = squads.merge(players[extra], on="player_id", how="left")
    except FileNotFoundError:
        print("  ⚠  players.csv not found — using scraped fields only")

    df = squads.rename(columns={"market_value_in_eur": "market_value_eur"})
    df["date"] = df["scraped_at"]
    # Columns the rest of the pipeline / app already read
    df["current_club_name_val"] = df["current_club_name"]
    df["current_club_id_val"] = df["current_club_id"]
    df["current_club_name_player"] = df["current_club_name"]
    df["player_club_domestic_competition_id"] = \
        df["current_club_domestic_competition_id"]
    df["market_value_in_eur_player"] = df["market_value_eur"]
    hv = pd.to_numeric(df.get("highest_market_value_in_eur"), errors="coerce")
    df["highest_market_value_in_eur"] = pd.concat(
        [hv, df["market_value_eur"]], axis=1).max(axis=1)

    now = pd.Timestamp.now()
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = ((now - dob).dt.days / 365.25).round(1)
    exp = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
    df["contract_years_remaining"] = (
        (exp - now).dt.days / 365.25).clip(lower=0).round(2)

    print(f"✅ Final: {len(df)} players at Big 5 clubs, values as of {as_of}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load Transfermarkt data from Kaggle CSVs"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing player_valuations.csv and players.csv"
    )
    parser.add_argument(
        "--kaggle-only", action="store_true",
        help="Ignore tm_squads_current.csv and use the frozen Kaggle data"
    )
    args = parser.parse_args()

    raw_dir = args.data_dir or os.path.join(
        os.path.dirname(__file__), "..", "data", "raw")
    df = None if args.kaggle_only else load_live_squads(raw_dir)
    if df is None:
        print("ℹ️  No live squad scrape found — falling back to Kaggle data "
              "(frozen at June 2026)")
        df = load_transfermarkt(data_dir=args.data_dir)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tm_valuations_latest.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
