"""
preprocess.py

Merges FBref per-90 stats with Transfermarkt market valuations for the
2023-24 Big 5 European leagues season and produces a model-ready dataset.

Pipeline
--------
1. Load & clean FBref stats (via scrape_fbref.fetch_fbref_stats)
2. Engineer features (position groups, age buckets, log-value target)
3. Load & filter player_valuations.csv to Big 5 / 2023-24 season window
4. Join on player name + club  ← requires players.csv (see NOTE below)
5. Save processed outputs to data/processed/

NOTE — Missing file: players.csv
----------------------------------
player_valuations.csv is keyed on player_id only; it contains no player names.
To complete the FBref ↔ Transfermarkt join you need the Kaggle companion file
players.csv from the same dataset (transfermarkt-scraper or similar).
It should have at minimum: player_id, name, current_club_name.

Once you add data/raw/players.csv the full merge will run automatically.
Without it the script still runs and saves the FBref feature table, which
is sufficient to start exploratory analysis and baseline modelling.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scrape_fbref import fetch_fbref_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SEASON_SLUG  = "2324"
SEASON_START = "2023-07-01"
SEASON_END   = "2024-06-30"

BIG5_COMP_IDS = {"GB1", "ES1", "L1", "IT1", "FR1"}

# Map FBref's Comp strings to Transfermarkt competition_ids for league labelling
LEAGUE_TO_COMP_ID = {
    "Premier League": "GB1",
    "La Liga":        "ES1",
    "Bundesliga":     "L1",
    "Serie A":        "IT1",
    "Ligue 1":        "FR1",
}

# ── Position grouping ─────────────────────────────────────────────────────────

def _broad_position(pos_str: str) -> str:
    """
    Collapse FBref's multi-value position strings (e.g. 'MF,FW') to a single
    broad category: GK, DF, MF, FW.  The first-listed position is used.
    """
    if not isinstance(pos_str, str) or not pos_str.strip():
        return "Unknown"
    primary = pos_str.split(",")[0].strip().upper()
    mapping = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
    return mapping.get(primary, "Unknown")


# ── FBref feature engineering ─────────────────────────────────────────────────

def build_fbref_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the FBref stats DataFrame.

    New columns:
        position_group  – broad position (GK/DF/MF/FW)
        age_group       – U21 / 21-24 / 25-29 / 30+
        comp_id         – Transfermarkt competition ID (for joining)
        goal_inv_p90    – goals + assists per 90 (general involvement proxy)
    """
    out = df.copy()

    out["position_group"] = out["position"].apply(_broad_position)

    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 21, 24, 29, 99],
        labels=["U21", "21-24", "25-29", "30+"],
        right=True,
    )

    # Add competition ID for later valuation join
    def _comp_id(league_str: str) -> Optional[str]:
        if not isinstance(league_str, str):
            return None
        for key, cid in LEAGUE_TO_COMP_ID.items():
            if key.lower() in league_str.lower():
                return cid
        return None

    out["comp_id"] = out["league"].apply(_comp_id)

    # Composite offensive involvement rate
    if "goals_p90" in out.columns and "assists_p90" in out.columns:
        out["goal_inv_p90"] = (out["goals_p90"] + out["assists_p90"]).round(3)

    return out


# ── Transfermarkt valuation processing ───────────────────────────────────────

def load_tm_valuations(
    season_start: str = SEASON_START,
    season_end: str   = SEASON_END,
) -> pd.DataFrame:
    """
    Load player_valuations.csv, filter to Big 5 leagues and the 2023-24
    season window, and return one row per player_id (the latest valuation
    recorded within the window).

    Returns a DataFrame with columns:
        player_id, market_value_in_eur, valuation_date,
        current_club_name, comp_id
    """
    path = RAW_DIR / "player_valuations.csv"
    val  = pd.read_csv(path, parse_dates=["date"])

    # Filter to Big 5
    val = val[val["player_club_domestic_competition_id"].isin(BIG5_COMP_IDS)].copy()

    # Filter to season window
    val = val[(val["date"] >= season_start) & (val["date"] <= season_end)].copy()

    log.info("Valuation rows after Big5 + season filter: %d  (%d unique player_ids)",
             len(val), val["player_id"].nunique())

    # Keep the latest record per player_id within the window
    val = (
        val.sort_values("date")
           .groupby("player_id", as_index=False)
           .last()
           .rename(columns={
               "date":                                  "valuation_date",
               "player_club_domestic_competition_id":  "comp_id",
           })
    )

    # Log-transform market value (right-skewed; avoids outlier dominance)
    val["log_market_value"] = np.log1p(val["market_value_in_eur"])

    keep = ["player_id", "market_value_in_eur", "log_market_value",
            "valuation_date", "current_club_name", "comp_id"]
    return val[[c for c in keep if c in val.columns]].reset_index(drop=True)


# ── Name normalisation (used during fuzzy join) ───────────────────────────────

def _normalise_name(name: str) -> str:
    """Lowercase, strip diacritics, remove punctuation/extra spaces."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    replacements = {
        # Western European
        "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ā": "a", "ă": "a",
        "æ": "ae",
        "é": "e", "è": "e", "ê": "e", "ë": "e", "ě": "e", "ē": "e", "ę": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ő": "o", "ø": "o", "ō": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u", "ű": "u", "ū": "u", "ů": "u",
        "ý": "y", "ÿ": "y",
        "ñ": "n", "ń": "n", "ň": "n",
        "ç": "c", "ć": "c", "č": "c",
        "ß": "ss",
        "ș": "s", "ś": "s", "š": "s",
        "ț": "t", "ť": "t",
        "ž": "z", "ź": "z", "ż": "z",
        "ř": "r",
        "ľ": "l", "ĺ": "l", "ļ": "l",
        "đ": "d", "ď": "d",
        "ğ": "g",
        # Scandinavian / Baltic
        "ā": "a", "ē": "e", "ī": "i", "ū": "u",
        "ķ": "k", "ļ": "l", "ņ": "n", "ģ": "g",
    }
    for src, tgt in replacements.items():
        name = name.replace(src, tgt)
    # Remove anything that isn't a letter, digit, or space
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _normalise_club(name: str) -> str:
    """Strip common prefixes/suffixes so 'Brighton' matches 'Brighton & Hove Albion'."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # Remove legal suffixes and common prefixes
    for pat in (r"\bfc\b", r"\bsc\b", r"\bac\b", r"\bsv\b", r"\bvfb\b",
                r"\bvfl\b", r"\btsv\b", r"\bfsv\b", r"\bbsc\b", r"\bss\b",
                r"&.*$", r"\d{4}$", r"1\.", r"0\.",):
        name = re.sub(pat, "", name)
    # Diacritics
    for src, tgt in {"ü": "u", "ö": "o", "ä": "a", "ñ": "n"}.items():
        name = name.replace(src, tgt)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ── FBref ↔ Transfermarkt join ────────────────────────────────────────────────

def join_fbref_to_valuations(
    fbref: pd.DataFrame,
    valuations: pd.DataFrame,
    players_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge FBref features with Transfermarkt market values.

    Requires a players.csv mapping player_id → player name.
    If the file is not present, returns the FBref DataFrame unchanged with
    a warning, so downstream code can still run without valuation data.

    Match strategy (in order):
        1. Exact:  normalised name + exact comp_id
        2. Fuzzy:  normalised name token overlap + comp_id  (>= 0.8 similarity)

    Parameters
    ----------
    fbref       : DataFrame from build_fbref_features()
    valuations  : DataFrame from load_tm_valuations()
    players_path: path to players.csv; defaults to data/raw/players.csv
    """
    if players_path is None:
        players_path = RAW_DIR / "players.csv"

    if not players_path.exists():
        log.warning(
            "players.csv not found at %s — skipping valuation join.\n"
            "Download it from the same Kaggle dataset as player_valuations.csv\n"
            "(columns needed: player_id, name, current_club_name).",
            players_path,
        )
        return fbref

    players = pd.read_csv(players_path, usecols=lambda c: c in
                          {"player_id", "name", "current_club_name"})

    # Attach player names to valuations.
    # comp_id in the valuation row = the league the player was in at valuation
    # time, which is the correct field for league-scoped name matching.
    val_named = valuations.merge(
        players[["player_id", "name"]].drop_duplicates("player_id"),
        on="player_id",
        how="left",
    )
    val_named = val_named[val_named["name"].notna()].copy()
    val_named["name_norm"] = val_named["name"].apply(_normalise_name)
    val_named["club_norm"] = val_named["current_club_name"].apply(_normalise_club)

    fbref_copy = fbref.copy()
    fbref_copy["name_norm"] = fbref_copy["player"].apply(_normalise_name)
    fbref_copy["club_norm"] = fbref_copy["team"].apply(_normalise_club)

    # ── Exact match on (name_norm, comp_id) ──────────────────────────────────
    exact = fbref_copy.merge(
        val_named[["name_norm", "comp_id",
                   "market_value_in_eur", "log_market_value", "valuation_date"]],
        on=["name_norm", "comp_id"],
        how="left",
    )

    matched   = exact["market_value_in_eur"].notna().sum()
    total     = len(exact)
    log.info("Exact name+league match: %d / %d players (%.1f%%)",
             matched, total, 100 * matched / max(total, 1))

    # ── Token-overlap fuzzy pass for unmatched rows ───────────────────────────
    unmatched_mask = exact["market_value_in_eur"].isna()
    if unmatched_mask.any():
        log.info("Attempting fuzzy match for %d unmatched players …", unmatched_mask.sum())

        # Build a lookup dict: comp_id → list of (name_norm, value)
        val_lookup: dict = {}
        for _, row in val_named.iterrows():
            val_lookup.setdefault(row["comp_id"], []).append(
                (row["name_norm"], row["market_value_in_eur"],
                 row["log_market_value"], row["valuation_date"])
            )

        def _fuzzy_lookup(row: pd.Series) -> tuple:
            candidates = val_lookup.get(row["comp_id"], [])
            if not candidates:
                return (np.nan, np.nan, pd.NaT)
            q_tokens = set(row["name_norm"].split())
            best_score, best_val, best_log, best_date = 0.0, np.nan, np.nan, pd.NaT
            for (cname, cval, clog, cdate) in candidates:
                c_tokens = set(cname.split())
                union = q_tokens | c_tokens
                if not union:
                    continue
                score = len(q_tokens & c_tokens) / len(union)
                if score > best_score:
                    best_score, best_val, best_log, best_date = score, cval, clog, cdate
            if best_score >= 0.8:
                return (best_val, best_log, best_date)
            return (np.nan, np.nan, pd.NaT)

        fuzzy_cols = exact.loc[unmatched_mask].apply(_fuzzy_lookup, axis=1, result_type="expand")
        fuzzy_cols.columns = ["market_value_in_eur", "log_market_value", "valuation_date"]
        exact["market_value_in_eur"] = exact["market_value_in_eur"].astype("float64")
        exact["log_market_value"]    = exact["log_market_value"].astype("float64")
        exact.loc[unmatched_mask, "market_value_in_eur"] = fuzzy_cols["market_value_in_eur"].values
        exact.loc[unmatched_mask, "log_market_value"]    = fuzzy_cols["log_market_value"].values
        exact.loc[unmatched_mask, "valuation_date"]      = fuzzy_cols["valuation_date"].values

        total_matched = exact["market_value_in_eur"].notna().sum()
        log.info("After fuzzy pass: %d / %d players matched (%.1f%%)",
                 total_matched, total, 100 * total_matched / max(total, 1))

    # Drop temporary normalisation columns
    exact = exact.drop(columns=["name_norm", "club_norm"], errors="ignore")
    return exact


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(season_slug: str = SEASON_SLUG) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline and return the merged dataset.

    Saves three files to data/processed/:
        fbref_features_{slug}.csv   – FBref stats + engineered features
        tm_valuations_{slug}.csv    – Transfermarkt Big5 season valuations
        model_dataset_{slug}.csv    – Merged dataset (if players.csv present)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: FBref ─────────────────────────────────────────────────────────
    log.info("Loading FBref stats …")
    raw_fbref = fetch_fbref_stats(season_slug=season_slug)
    fbref     = build_fbref_features(raw_fbref)

    # Drop goalkeepers — different feature set, separate model needed
    fbref = fbref[fbref["position_group"] != "GK"].copy()
    log.info("FBref outfield players (≥ 900 min): %d", len(fbref))

    fbref_path = PROCESSED_DIR / f"fbref_features_{season_slug}.csv"
    fbref.to_csv(fbref_path, index=False)
    log.info("Saved FBref features → %s", fbref_path)

    # ── Step 2: Transfermarkt valuations ──────────────────────────────────────
    log.info("Processing Transfermarkt valuations …")
    valuations = load_tm_valuations()

    val_path = PROCESSED_DIR / f"tm_valuations_{season_slug}.csv"
    valuations.to_csv(val_path, index=False)
    log.info("Saved TM valuations → %s  (%d rows)", val_path, len(valuations))

    # ── Step 3: Join ──────────────────────────────────────────────────────────
    log.info("Joining FBref to Transfermarkt …")
    merged = join_fbref_to_valuations(fbref, valuations)

    model_path = PROCESSED_DIR / f"model_dataset_{season_slug}.csv"
    merged.to_csv(model_path, index=False)
    log.info("Saved model dataset → %s  (%d rows, %d cols)", model_path,
             len(merged), len(merged.columns))

    # Summary
    has_value = merged["market_value_in_eur"].notna().sum() if "market_value_in_eur" in merged.columns else 0
    log.info("Players with market value: %d / %d", has_value, len(merged))
    if has_value > 0:
        log.info("Value stats (EUR): median=%.0f  max=%.0f",
                 merged["market_value_in_eur"].median(),
                 merged["market_value_in_eur"].max())

    return merged


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_pipeline()

    p90_cols = [c for c in df.columns if c.endswith("_p90")]
    display  = ["league", "player", "team", "position_group", "age",
                "minutes_played"] + p90_cols

    if "market_value_in_eur" in df.columns:
        display.append("market_value_in_eur")

    print("\n", df[[c for c in display if c in df.columns]].head(20).to_string(index=False))
    print(f"\nTotal rows : {len(df)}")
    print(f"Columns    : {list(df.columns)}")
