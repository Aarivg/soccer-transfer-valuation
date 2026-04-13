"""
preprocess.py — Merge FBref stats with Transfermarkt valuations.

Fuzzy-matches player names, engineers features (per-90 stats, age buckets,
league tiers, position encoding), log-transforms target, and creates
a model-ready dataset.

Usage:
    python src/preprocess.py
    python src/preprocess.py --season 2526
"""

import argparse
import os
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process as rfprocess
    USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher, get_close_matches
    USE_RAPIDFUZZ = False
    print("⚠  rapidfuzz not installed — using difflib (slower). "
          "Run: pip install rapidfuzz")


# ── Elite clubs that command premium transfer fees ───────────────────
ELITE_CLUBS = {
    # Tier 1: global super-clubs
    "Real Madrid", "Barcelona", "Manchester City", "Paris Saint-Germain",
    "Bayern Munich", "Bayern München",
    # Tier 2: consistent CL contenders
    "Liverpool", "Arsenal", "Chelsea", "Manchester United",
    "Juventus", "Inter Milan", "AC Milan", "Atlético Madrid",
    "Borussia Dortmund",
}

TIER1_CLUBS = {
    "Real Madrid", "Barcelona", "Manchester City",
    "Paris Saint-Germain", "Bayern Munich", "Bayern München",
}


# ── Name cleaning ────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    import unicodedata
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    for remove in ["jr.", "jr", "sr.", "sr", "ii", "iii"]:
        name = name.replace(remove, "")
    return " ".join(name.split())


# ── Fuzzy matching ───────────────────────────────────────────────────
def fuzzy_match_players(
    fbref_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    fbref_name_col: str = "player",
    tm_name_col: str = "player_name",
    threshold: int = 80,
) -> pd.DataFrame:
    """
    Match FBref players to Transfermarkt players by name + squad.
    """
    print("🔗 Fuzzy-matching players...")

    fbref_df = fbref_df.copy()
    tm_df = tm_df.copy()
    fbref_df["_clean_name"] = fbref_df[fbref_name_col].apply(clean_name)
    tm_df["_clean_name"] = tm_df[tm_name_col].apply(clean_name)

    tm_names = tm_df["_clean_name"].dropna().unique().tolist()

    matches = []
    unmatched = 0

    for _, row in fbref_df.iterrows():
        name = row["_clean_name"]
        if not name:
            unmatched += 1
            continue

        if USE_RAPIDFUZZ:
            result = rfprocess.extractOne(
                name, tm_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )
            if result:
                best_name, score, _ = result
                tm_match = tm_df[tm_df["_clean_name"] == best_name].iloc[0]
                merged_row = {**row.to_dict(), **{
                    f"tm_{k}": v for k, v in tm_match.to_dict().items()
                    if k != "_clean_name"
                }}
                merged_row["match_score"] = score
                matches.append(merged_row)
            else:
                unmatched += 1
        else:
            close = get_close_matches(name, tm_names, n=1, cutoff=0.7)
            if close:
                best_name = close[0]
                tm_match = tm_df[tm_df["_clean_name"] == best_name].iloc[0]
                score = SequenceMatcher(None, name, best_name).ratio() * 100
                merged_row = {**row.to_dict(), **{
                    f"tm_{k}": v for k, v in tm_match.to_dict().items()
                    if k != "_clean_name"
                }}
                merged_row["match_score"] = score
                matches.append(merged_row)
            else:
                unmatched += 1

    result_df = pd.DataFrame(matches)
    total = len(fbref_df)
    matched = len(result_df)
    print(f"  ✓ Matched: {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"  ✗ Unmatched: {unmatched}")

    result_df = result_df.drop(columns=["_clean_name"], errors="ignore")
    return result_df


# ── Feature engineering ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for the model."""
    df = df.copy()

    # ── Age ───────────────────────────────────────────────────────────
    for candidate in ["age", "tm_age"]:
        if candidate in df.columns:
            df["age"] = pd.to_numeric(df[candidate], errors="coerce")
            break

    # ── Market value (target) ────────────────────────────────────────
    mv_col = None
    for candidate in ["tm_market_value_eur", "market_value_eur",
                       "tm_market_value_in_eur"]:
        if candidate in df.columns:
            mv_col = candidate
            break

    if mv_col:
        df["market_value_eur"] = pd.to_numeric(df[mv_col], errors="coerce")
        df["log_market_value"] = np.log1p(df["market_value_eur"])

    # ── Per-90 stats ─────────────────────────────────────────────────
    minutes_col = None
    for candidate in ["minutes", "minutes_90s", "min", "90s"]:
        if candidate in df.columns:
            minutes_col = candidate
            break

    if minutes_col:
        minutes = pd.to_numeric(df[minutes_col], errors="coerce")
        nineties = minutes / 90 if "90" not in str(minutes_col) else minutes

        counting_to_per90 = {
            "progressive_passes": "progressive_passes_per90",
            "progressive_carries": "progressive_carries_per90",
            "tackles_won": "tackles_won_per90",
            "interceptions": "interceptions_per90",
            "assisted_shots": "key_passes_per90",
            "successful_dribbles": "successful_dribbles_per90",
            "ball_recoveries": "ball_recoveries_per90",
            "aerials_won": "aerials_won_per90",
            "passes_into_final_third": "passes_into_final_third_per90",
            "carries_into_final_third": "carries_into_final_third_per90",
            "carries_into_penalty_area": "carries_into_penalty_area_per90",
        }
        for raw, per90 in counting_to_per90.items():
            if raw in df.columns and per90 not in df.columns:
                df[per90] = (
                    pd.to_numeric(df[raw], errors="coerce") / nineties
                ).replace([np.inf, -np.inf], np.nan)

    # ── Position encoding ────────────────────────────────────────────
    if "pos" in df.columns:
        pos = df["pos"].astype(str).str.upper()
        df["is_forward"] = pos.str.contains("FW").astype(int)
        df["is_midfielder"] = pos.str.contains("MF").astype(int)
        df["is_defender"] = pos.str.contains("DF").astype(int)

        df["position_group"] = "MF"
        df.loc[df["is_forward"] == 1, "position_group"] = "FW"
        df.loc[df["is_defender"] == 1, "position_group"] = "DF"

    # ── League encoding ──────────────────────────────────────────────
    league_col = None
    for candidate in ["comp", "league", "tm_current_club_domestic_competition_id"]:
        if candidate in df.columns:
            league_col = candidate
            break

    if league_col:
        league = df[league_col].astype(str).str.lower()
        league_map = {
            "premier league": "Premier League",
            "eng": "Premier League", "gb1": "Premier League",
            "la liga": "La Liga",
            "es1": "La Liga",
            "bundesliga": "Bundesliga", "1. bundesliga": "Bundesliga",
            "l1": "Bundesliga",
            "serie a": "Serie A",
            "it1": "Serie A",
            "ligue 1": "Ligue 1",
            "fr1": "Ligue 1",
        }
        df["league_name"] = league.map(
            lambda x: next(
                (v for k, v in league_map.items() if k in x),
                "Other"
            )
        )

        # League prestige — toned down from v1
        prestige = {
            "Premier League": 1.15,
            "La Liga": 1.10,
            "Bundesliga": 1.05,
            "Serie A": 1.03,
            "Ligue 1": 1.00,
        }
        df["league_prestige"] = df["league_name"].map(prestige).fillna(1.0)

    # ── Age potential multiplier — REALISTIC ─────────────────────────
    # Real transfer premiums are much smaller than v1
    if "age" in df.columns:
        conditions = [
            df["age"] <= 20,
            (df["age"] > 20) & (df["age"] <= 23),
            (df["age"] > 23) & (df["age"] <= 27),
            (df["age"] > 27) & (df["age"] <= 30),
            df["age"] > 30,
        ]
        multipliers = [1.15, 1.08, 1.00, 0.93, 0.85]
        df["age_potential_mult"] = np.select(conditions, multipliers, default=1.0)

    # ── Elite club premium ───────────────────────────────────────────
    # Players at super-clubs are worth more due to brand, revenue,
    # Champions League revenue, shirt sales, and negotiation power.
    squad_col = None
    for candidate in ["squad", "tm_current_club_name"]:
        if candidate in df.columns:
            squad_col = candidate
            break

    if squad_col:
        df["is_elite_club"] = df[squad_col].astype(str).apply(
            lambda x: any(club.lower() in x.lower() for club in ELITE_CLUBS)
        ).astype(int)

        df["is_tier1_club"] = df[squad_col].astype(str).apply(
            lambda x: any(club.lower() in x.lower() for club in TIER1_CLUBS)
        ).astype(int)

        # Club premium multiplier
        df["club_premium"] = 1.0
        df.loc[df["is_elite_club"] == 1, "club_premium"] = 1.10
        df.loc[df["is_tier1_club"] == 1, "club_premium"] = 1.15

    # ── Contract years ───────────────────────────────────────────────
    contract_col = None
    for candidate in ["contract_years_remaining", "tm_contract_years_remaining"]:
        if candidate in df.columns:
            contract_col = candidate
            break
    if contract_col:
        df["contract_years_remaining"] = pd.to_numeric(
            df[contract_col], errors="coerce"
        ).fillna(1.0).clip(lower=0)

    print(f"✅ Feature engineering complete: {len(df.columns)} total columns")
    return df


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess & merge datasets")
    parser.add_argument("--season", default="2526")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    raw_dir = os.path.join(base, "data", "raw")
    proc_dir = os.path.join(base, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    # ── Load FBref ───────────────────────────────────────────────────
    fbref_path = os.path.join(raw_dir, f"fbref_stats_{args.season}.csv")
    if not os.path.exists(fbref_path):
        print(f"❌ FBref data not found at {fbref_path}")
        print("   Run: python src/scrape_fbref.py first")
        return
    fbref = pd.read_csv(fbref_path)
    print(f"📊 FBref: {len(fbref)} players")

    # ── Load Transfermarkt ───────────────────────────────────────────
    tm_path = os.path.join(proc_dir, "tm_valuations_latest.csv")
    if not os.path.exists(tm_path):
        print(f"❌ Transfermarkt data not found at {tm_path}")
        print("   Run: python src/scrape_transfermarkt.py first")
        return
    tm = pd.read_csv(tm_path)
    print(f"💰 Transfermarkt: {len(tm)} players")

    # ── Fuzzy match & merge ──────────────────────────────────────────
    merged = fuzzy_match_players(fbref, tm)

    # ── Feature engineering ──────────────────────────────────────────
    dataset = engineer_features(merged)

    # ── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(proc_dir, f"model_dataset_{args.season}.csv")
    dataset.to_csv(out_path, index=False)
    print(f"\n💾 Saved model dataset: {out_path}")
    print(f"   {len(dataset)} players × {len(dataset.columns)} features")

    if "market_value_eur" in dataset.columns:
        mv = dataset["market_value_eur"]
        print(f"\n📈 Market value range: €{mv.min():,.0f} – €{mv.max():,.0f}")
        print(f"   Median: €{mv.median():,.0f}")

    if "position_group" in dataset.columns:
        print(f"\n👥 By position:")
        print(dataset["position_group"].value_counts().to_string())

    if "league_name" in dataset.columns:
        print(f"\n🏟️  By league:")
        print(dataset["league_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
