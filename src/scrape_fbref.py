"""
scrape_fbref.py

Loads locally saved FBref CSV exports from data/raw/ and returns a single
clean DataFrame of per-90 stats for outfield players across the Big 5 leagues.

Expected files in data/raw/:
    fbref_standard_2324.csv   – playing time, goals, assists, cards
    fbref_defense_2324.csv    – tackles won, interceptions (only populated cols)

Notes on data availability
--------------------------
FBref CSV exports use a two-row header.  Row 0 carries section labels (blank
in the export); row 1 holds the actual column names.  Duplicate stat names
(e.g. "Gls" appears as both raw count and per-90) are disambiguated by pandas
as "Gls" and "Gls.1" etc.

The passing and possession CSV exports from this FBref page contained no
stat data beyond identity columns — only the standard and defense tables had
usable content.  Stats not available from these exports:

    xG, xAG          – need the "Expected Goals" FBref export (separate tab)
    Pressures         – need the "Defensive Actions" FBref export
    Progressive passes / carries counts – need dedicated passing/possession
                        exports; PrgP and PrgC columns were blank in the
                        exports provided.

Re-download those tabs from FBref and call fetch_fbref_stats() once those
files are in data/raw/ to include those features automatically.
"""

from pathlib import Path
from typing import Optional, List

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

MIN_MINUTES = 900

# ── Column rename maps ────────────────────────────────────────────────────────
# FBref exports per-90 duplicates with ".1" suffixes after pandas deduplication.

STANDARD_RENAME = {
    "Rk\n▲":  "rk",
    "Player":  "player",
    "Nation":  "nation",
    "Pos":     "position",
    "Squad":   "team",
    "Comp":    "league",
    "Age":     "age",
    "Born":    "birth_year",
    "MP":      "matches_played",
    "Starts":  "starts",
    "Min":     "minutes_played",
    "90s":     "minutes_90s",
    # counting stats
    "Gls":     "goals",
    "Ast":     "assists",
    "G+A":     "goals_assists",
    "G-PK":    "goals_non_pen",
    "PK":      "pen_scored",
    "PKatt":   "pen_attempted",
    "CrdY":    "yellow_cards",
    "CrdR":    "red_cards",
    # per-90 stats (pandas adds ".1" to deduplicate repeated column names)
    "Gls.1":   "goals_p90",
    "Ast.1":   "assists_p90",
    "G+A.1":   "goals_assists_p90",
    "G-PK.1":  "goals_non_pen_p90",
    "G+A-PK":  "goals_assists_non_pen_p90",
    "Matches": "_drop",
}

DEFENSE_RENAME = {
    "Rk":      "rk",
    "Player":  "player",
    "Nation":  "nation",
    "Pos":     "position",
    "Squad":   "team",
    "Comp":    "league",
    "Age":     "age",
    "Born":    "birth_year",
    "90s":     "minutes_90s",
    # only these two columns have data in the exported CSV
    "TklW":    "tackles_won",
    "Int":     "interceptions",
    # remaining cols are blank in the export — still rename to avoid collisions
    "Tkl":     "_tkl_raw",
    "Def 3rd": "_tkl_def3",
    "Mid 3rd": "_tkl_mid3",
    "Att 3rd": "_tkl_att3",
    "Tkl.1":   "_drib_tkl",
    "Att":     "_drib_att",
    "Tkl%":    "_drib_tkl_pct",
    "Lost":    "_drib_lost",
    "Blocks":  "_blocks",
    "Sh":      "_shots_blk",
    "Pass":    "_passes_blk",
    "Tkl+Int": "_tkl_int",
    "Clr":     "_clearances",
    "Err":     "_errors",
    "Matches": "_drop",
}

JOIN_KEYS = ["player", "team", "league"]


# ── Table loader ──────────────────────────────────────────────────────────────

def _load_table(path: Path, rename: dict) -> pd.DataFrame:
    """
    Read one FBref CSV export (two-row header), strip repeated header rows,
    apply column renames, and coerce numeric types.
    """
    df = pd.read_csv(path, header=1, dtype=str)

    # Drop rows where the rank column contains the literal text 'Rk'
    rk_col = df.columns[0]
    df = df[df[rk_col] != "Rk"].copy()

    # Drop rows without a player name
    if "Player" in df.columns:
        df = df[df["Player"].notna() & (df["Player"].str.strip() != "")].copy()

    df = df.rename(columns=rename)

    # Remove flagged columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_drop")
                           or (c.startswith("_") and c != "_drop")], errors="ignore")

    # Strip whitespace from string columns
    str_cols = {"player", "nation", "position", "team", "league"}
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Coerce non-string columns to numeric
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    return df.reset_index(drop=True)


# ── Main loader ───────────────────────────────────────────────────────────────

def fetch_fbref_stats(
    season_slug: str = "2324",
    leagues: Optional[List[str]] = None,
    min_minutes: int = MIN_MINUTES,
) -> pd.DataFrame:
    """
    Load and merge the FBref standard and defense CSV exports.

    Parameters
    ----------
    season_slug : str
        Filename suffix, e.g. "2324" for 2023-24.
    leagues : list of str, optional
        League name substrings to keep (e.g. ["Premier League"]).
        Pass None to return all Big 5 leagues.
    min_minutes : int
        Minimum minutes played filter.

    Returns
    -------
    pd.DataFrame
        One row per player–team combination with all available stats.
    """
    std_path = RAW_DIR / f"fbref_standard_{season_slug}.csv"
    def_path = RAW_DIR / f"fbref_defense_{season_slug}.csv"

    for p in (std_path, def_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    std = _load_table(std_path, STANDARD_RENAME)
    dfn = _load_table(def_path, DEFENSE_RENAME)

    # Carry only the two populated defense columns
    def_cols = JOIN_KEYS + [c for c in ("tackles_won", "interceptions")
                             if c in dfn.columns]
    merged = std.merge(dfn[def_cols], on=JOIN_KEYS, how="left")

    # ── Filters ───────────────────────────────────────────────────────────────
    if leagues:
        pattern = "|".join(leagues)
        merged = merged[
            merged["league"].str.contains(pattern, case=False, na=False)
        ].copy()

    if "minutes_played" in merged.columns:
        merged = merged[merged["minutes_played"] >= min_minutes].copy()

    # ── Per-90 rates for defense stats (standard per-90s already provided) ───
    if "minutes_90s" in merged.columns:
        for col in ("tackles_won", "interceptions"):
            if col in merged.columns:
                merged[f"{col}_p90"] = (merged[col] / merged["minutes_90s"]).round(3)

    return merged.sort_values(["league", "player"]).reset_index(drop=True)


# ── I/O ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, season_slug: str = "2324") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_out = RAW_DIR / f"fbref_merged_{season_slug}.csv"
    df.to_csv(raw_out, index=False)
    print(f"Saved merged FBref → {raw_out}  ({len(df)} rows, {len(df.columns)} cols)")

    p90_cols = sorted(c for c in df.columns if c.endswith("_p90"))
    id_cols  = ["league", "player", "team", "position", "age", "minutes_played"]
    keep     = [c for c in id_cols + p90_cols if c in df.columns]
    proc_out = PROCESSED_DIR / f"fbref_per90_{season_slug}.csv"
    df[keep].to_csv(proc_out, index=False)
    print(f"Saved per-90 subset → {proc_out}")


if __name__ == "__main__":
    df = fetch_fbref_stats()
    save(df)
    p90_cols = [c for c in df.columns if c.endswith("_p90")]
    display  = ["league", "player", "team", "position", "minutes_played"] + p90_cols
    print("\n", df[[c for c in display if c in df.columns]].head(20).to_string(index=False))
    print(f"\nTotal qualifying players : {len(df)}")
    print(f"Columns ({len(df.columns)})          : {list(df.columns)}")
