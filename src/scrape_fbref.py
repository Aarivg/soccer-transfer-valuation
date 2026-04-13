"""
scrape_fbref.py — Load FBref player stats from local CSV exports.

Reads CSV files exported from FBref's Big 5 European Leagues pages
(standard, shooting, passing, possession, defense, misc), merges them
into a single DataFrame, and saves to data/raw/fbref_stats_SEASON.csv.

How to get the CSVs:
  1. Go to each FBref Big 5 stats page in Chrome
  2. Scroll to player table → Share & Export → Get table as CSV
  3. Save to data/raw/ as fbref_STATTYPE_SEASON.csv

Usage:
    python src/scrape_fbref.py                  # default: 2526
    python src/scrape_fbref.py --season 2324    # older season
"""

import argparse
import os
import io
import pandas as pd


STAT_TYPES = ["standard", "shooting", "passing", "possession", "defense", "misc"]


def load_fbref_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single FBref CSV export, handling the multi-header format.

    FBref CSVs often have TWO header rows:
      Row 0: category groups (e.g. "Total,,,,Short,,Medium,,Long,,")
      Row 1: actual column names (e.g. "Rk,Player,Nation,Pos,Squad,...")

    This function detects that pattern and uses the correct row as header.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    # Check if first row looks like category groups (lots of empty cells)
    first_row = lines[0].strip().split(",")
    empty_count = sum(1 for v in first_row if v.strip() == "")

    if len(lines) >= 2 and empty_count > len(first_row) * 0.3:
        # First row is category groups — use second row as header
        print(f"    (detected multi-header — using row 2 as columns)")
        header_line = lines[1]
        data_lines = lines[2:]
    else:
        # Single header row
        header_line = lines[0]
        data_lines = lines[1:]

    # Parse header
    headers = [h.strip() for h in header_line.strip().split(",")]

    # Handle duplicate column names by appending suffix
    seen = {}
    unique_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)

    # Filter out repeated header rows and separator rows in data
    clean_lines = [",".join(unique_headers) + "\n"]
    header_check = set(headers[:5])

    for line in data_lines:
        vals = line.strip().split(",")[:5]
        stripped_vals = [v.strip() for v in vals]

        # Skip rows that repeat the header
        if set(stripped_vals) == header_check:
            continue
        # Skip separator rows (all dashes or empty)
        if all(v.startswith("-") or v == "" for v in stripped_vals):
            continue
        # Skip completely empty rows
        if not line.strip():
            continue

        clean_lines.append(line)

    df = pd.read_csv(io.StringIO("".join(clean_lines)))

    # Lowercase column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df


def merge_stat_tables(data_dir: str, season: str) -> pd.DataFrame:
    """
    Load and merge all available FBref stat CSVs for a season.
    """
    print(f"📡 Loading FBref CSVs for season {season}...")

    merged = None
    loaded_stats = []

    for stat in STAT_TYPES:
        filename = f"fbref_{stat}_{season}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"  ⚠  Not found: {filename} — skipping")
            continue

        print(f"  → Loading: {filename}...")
        try:
            df = load_fbref_csv(filepath)
            print(f"    ✓ {len(df)} rows, {len(df.columns)} columns")
            loaded_stats.append(stat)

            if merged is None:
                merged = df
            else:
                # Use rk + player + squad as keys for exact matching
                primary_keys = []
                for k in ["rk", "player", "squad"]:
                    if k in merged.columns and k in df.columns:
                        primary_keys.append(k)

                if not primary_keys:
                    print(f"    ⚠  No common merge keys — skipping {stat}")
                    continue

                # Only keep new columns + keys from the right table
                existing = set(merged.columns) - set(primary_keys)
                new_cols = [c for c in df.columns if c not in existing or c in primary_keys]

                if len(new_cols) <= len(primary_keys):
                    print(f"    ⚠  No new columns to add from {stat}")
                    continue

                merged = merged.merge(
                    df[new_cols],
                    on=primary_keys,
                    how="left",
                    suffixes=("", f"_{stat}"),
                )
                print(f"    → Merged: {len(merged)} rows")

        except Exception as e:
            print(f"    ✗ Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if merged is None:
        raise RuntimeError(
            f"No FBref CSVs found in {data_dir} for season {season}.\n"
            f"Expected files like: fbref_standard_{season}.csv"
        )

    print(f"\n  📋 Loaded {len(loaded_stats)} stat tables: {loaded_stats}")

    # ── Parse minutes ────────────────────────────────────────────────
    min_col = None
    for candidate in ["min", "minutes", "90s", "minutes_90s"]:
        if candidate in merged.columns:
            min_col = candidate
            break

    if min_col:
        merged["minutes"] = (
            merged[min_col].astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        if "90" in str(min_col):
            merged["minutes"] = merged["minutes"] * 90

    # ── Filter: ≥900 minutes ─────────────────────────────────────────
    if "minutes" in merged.columns:
        before = len(merged)
        merged = merged[merged["minutes"] >= 900]
        print(f"  🔽 Filtered ≥900 min: {before} → {len(merged)} players")

    # ── Drop goalkeepers ─────────────────────────────────────────────
    if "pos" in merged.columns:
        before = len(merged)
        merged = merged[~merged["pos"].astype(str).str.contains("GK", na=False)]
        print(f"  🔽 Dropped GKs: {before} → {len(merged)} players")

    # ── Clean up age column ──────────────────────────────────────────
    if "age" in merged.columns:
        merged["age"] = (
            merged["age"].astype(str)
            .str.split("-").str[0]
            .pipe(pd.to_numeric, errors="coerce")
        )

    print(f"\n✅ Final dataset: {len(merged)} players, {len(merged.columns)} features")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Load FBref player stats from CSVs")
    parser.add_argument(
        "--season", default="2526",
        help='Season code matching filenames, e.g. "2526" for 2025-26 (default: 2526)'
    )
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    raw_dir = os.path.join(base, "data", "raw")

    df = merge_stat_tables(raw_dir, args.season)

    # Save merged output
    out_path = os.path.join(raw_dir, f"fbref_stats_{args.season}.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
