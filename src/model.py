"""
model.py — Train XGBoost valuation model with reality-grounded predictions.

Key improvements over v1:
  - Blended predictions: mix model output with actual market value
    (the market knows things stats don't — brand, scarcity, contract leverage)
  - Capped value gaps: no player can be >60% under/overvalued
  - Club premium: elite clubs command higher fees
  - Toned-down age/league multipliers
  - SHAP explainability + position-specific models
  - Confidence intervals via quantile regression

Usage:
    python src/model.py
    python src/model.py --season 2526
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed. Run: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠  shap not installed. Run: pip install shap")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_QUANTILE = True
except ImportError:
    HAS_QUANTILE = False


# ── Configuration ────────────────────────────────────────────────────
# How much to trust the model vs the market (0 = pure market, 1 = pure model)
MODEL_WEIGHT = 0.65

# Maximum percentage a player can be under/overvalued
MAX_GAP_PCT = 60  # ±60%


# ── Feature sets ─────────────────────────────────────────────────────
CORE_FEATURES = [
    "age",
    "contract_years_remaining",
    "league_prestige",
    "is_forward", "is_midfielder", "is_defender",
    "is_elite_club", "club_premium",
]

PERFORMANCE_FEATURES = [
    "goals_per90", "assists_per90",
    "goals_pens_per90",
    "xg_per90", "xg_assist_per90",
    "npxg_per90",
    "shots_per90", "shots_on_target_per90",
    "goals_per_shot",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "tackles_won_per90",
    "interceptions_per90",
    "key_passes_per90",
    "successful_dribbles_per90",
    "aerials_won_per90",
    "passes_into_final_third_per90",
    "carries_into_final_third_per90",
]

POSITION_FEATURES = {
    "FW": [
        "goals_per90", "goals_pens_per90", "xg_per90", "npxg_per90",
        "assists_per90", "xg_assist_per90",
        "shots_per90", "shots_on_target_per90", "goals_per_shot",
        "progressive_carries_per90", "successful_dribbles_per90",
        "age", "contract_years_remaining", "league_prestige",
        "is_elite_club", "club_premium",
    ],
    "MF": [
        "goals_per90", "assists_per90",
        "xg_per90", "xg_assist_per90",
        "progressive_passes_per90", "progressive_carries_per90",
        "key_passes_per90", "passes_into_final_third_per90",
        "tackles_won_per90", "interceptions_per90",
        "successful_dribbles_per90",
        "age", "contract_years_remaining", "league_prestige",
        "is_elite_club", "club_premium",
    ],
    "DF": [
        "tackles_won_per90", "interceptions_per90",
        "aerials_won_per90",
        "progressive_passes_per90", "progressive_carries_per90",
        "passes_into_final_third_per90",
        "goals_per90", "assists_per90",
        "age", "contract_years_remaining", "league_prestige",
        "is_elite_club", "club_premium",
    ],
}

ALL_FEATURES = list(set(CORE_FEATURES + PERFORMANCE_FEATURES))


def get_available_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    available = [f for f in wanted if f in df.columns]
    missing = set(wanted) - set(available)
    if missing:
        print(f"  ⚠  Missing features (skipped): {missing}")
    return available


def prepare_data(df: pd.DataFrame, features: list[str], target: str):
    available = get_available_features(df, features)
    subset = df[available + [target]].dropna()
    X = subset[available].values
    y = subset[target].values
    return X, y, available, subset.index


# ── Train global model ───────────────────────────────────────────────
def train_global_model(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("🌍 GLOBAL MODEL (all positions)")
    print("="*60)

    target = "log_market_value"
    X, y, features, idx = prepare_data(df, ALL_FEATURES, target)
    print(f"  Features: {len(features)}, Samples: {len(X)}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(X, y, idx, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_eur = np.sqrt(mean_squared_error(
        np.expm1(y_test), np.expm1(y_pred)
    ))

    print(f"\n  📊 Results (test set):")
    print(f"     R²:   {r2:.4f}")
    print(f"     RMSE: {rmse:.4f} (log) | €{rmse_eur:,.0f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"     CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # SHAP
    shap_values = None
    if HAS_SHAP:
        print("\n  🔍 Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": features,
            "shap_importance": shap_importance,
        }).sort_values("shap_importance", ascending=False)
        print("\n  📋 Feature importance (SHAP):")
        for _, row in importance_df.head(10).iterrows():
            bar = "█" * int(row["shap_importance"] * 50)
            print(f"     {row['feature']:35s} {bar} {row['shap_importance']:.3f}")

    return {
        "model": model,
        "features": features,
        "r2": r2,
        "rmse_eur": rmse_eur,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "shap_values": shap_values,
        "X_test": X_test,
        "y_test": y_test,
        "idx_test": idx_test,
    }


# ── Train position-specific models ───────────────────────────────────
def train_position_models(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("🎯 POSITION-SPECIFIC MODELS")
    print("="*60)

    results = {}

    for pos, pos_features in POSITION_FEATURES.items():
        pos_df = df[df["position_group"] == pos]
        if len(pos_df) < 50:
            print(f"\n  ⚠  {pos}: only {len(pos_df)} players, skipping")
            continue

        print(f"\n  ── {pos} ({len(pos_df)} players) ──")
        target = "log_market_value"
        X, y, features, idx = prepare_data(pos_df, pos_features, target)

        if len(X) < 30:
            print(f"     Too few samples after NaN drop ({len(X)}), skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse_eur = np.sqrt(mean_squared_error(
            np.expm1(y_test), np.expm1(y_pred)
        ))

        print(f"     R²:   {r2:.4f}")
        print(f"     RMSE: €{rmse_eur:,.0f}")

        results[pos] = {
            "model": model,
            "features": features,
            "r2": r2,
            "rmse_eur": rmse_eur,
            "n_players": len(pos_df),
        }

    return results


# ── Confidence intervals ─────────────────────────────────────────────
def compute_confidence_intervals(
    df: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
    print("\n  📐 Computing confidence intervals...")

    X, y, avail_features, idx = prepare_data(df, features, target)

    model_lo = GradientBoostingRegressor(
        loss="quantile", alpha=0.10,
        n_estimators=300, max_depth=5, random_state=42
    )
    model_lo.fit(X, y)

    model_hi = GradientBoostingRegressor(
        loss="quantile", alpha=0.90,
        n_estimators=300, max_depth=5, random_state=42
    )
    model_hi.fit(X, y)

    pred_lo = np.expm1(model_lo.predict(X))
    pred_hi = np.expm1(model_hi.predict(X))

    result = pd.DataFrame(index=idx)
    result["predicted_value_lower"] = pred_lo
    result["predicted_value_upper"] = pred_hi

    print(f"     ✓ 80% prediction intervals computed for {len(result)} players")
    return result


# ── Reality-grounded predictions ─────────────────────────────────────
def generate_predictions(df: pd.DataFrame, global_result: dict) -> pd.DataFrame:
    """
    Generate predictions with reality checks:
    1. Blend model prediction with actual market value
    2. Apply modest age/league/club adjustments
    3. Cap maximum value gap at ±MAX_GAP_PCT%
    """
    print("\n" + "="*60)
    print("📈 GENERATING PREDICTIONS (reality-grounded)")
    print("="*60)
    print(f"  Model weight: {MODEL_WEIGHT:.0%} model / "
          f"{1-MODEL_WEIGHT:.0%} market")
    print(f"  Max gap: ±{MAX_GAP_PCT}%")

    model = global_result["model"]
    features = global_result["features"]

    X_full, _, _, idx_full = prepare_data(
        df, features, "log_market_value"
    )

    # Step 1: Raw model prediction (log scale → EUR)
    log_predictions = model.predict(X_full)
    raw_predictions = np.expm1(log_predictions)

    subset = df.loc[idx_full].copy()
    subset["predicted_raw_eur"] = raw_predictions

    # Step 2: Apply MODEST adjustments
    adjusted = raw_predictions.copy()

    if "age_potential_mult" in subset.columns:
        adjusted = adjusted * subset["age_potential_mult"].values

    if "club_premium" in subset.columns:
        adjusted = adjusted * subset["club_premium"].values

    subset["predicted_adjusted_eur"] = adjusted

    # Step 3: BLEND with actual market value
    # This is the key insight — the market isn't stupid.
    # A pure model misses brand value, scarcity, hype, agent power.
    # Blending says: "I trust the model 65%, the market 35%"
    if "market_value_eur" in subset.columns:
        actual = subset["market_value_eur"].values
        subset["predicted_value_eur"] = (
            MODEL_WEIGHT * adjusted + (1 - MODEL_WEIGHT) * actual
        )
    else:
        subset["predicted_value_eur"] = adjusted

    # Step 4: Cap the value gap
    if "market_value_eur" in subset.columns:
        actual = subset["market_value_eur"]

        # Raw gap
        subset["value_gap_eur_raw"] = (
            subset["predicted_value_eur"] - actual
        )

        # Cap: predicted can't be more than MAX_GAP_PCT% above/below actual
        max_pred = actual * (1 + MAX_GAP_PCT / 100)
        min_pred = actual * (1 - MAX_GAP_PCT / 100)
        subset["predicted_value_eur"] = subset["predicted_value_eur"].clip(
            lower=min_pred, upper=max_pred
        )

        # Final gap after capping
        subset["value_gap_eur"] = (
            subset["predicted_value_eur"] - actual
        )
        subset["value_gap_pct"] = (
            subset["value_gap_eur"] / actual * 100
        )

    # Confidence intervals
    if HAS_QUANTILE:
        ci = compute_confidence_intervals(df, features, "log_market_value")
        subset = subset.join(ci, how="left")

    # Sort and display
    if "value_gap_eur" in subset.columns:
        subset = subset.sort_values("value_gap_eur", ascending=False)

        print(f"\n  🟢 Top 10 UNDERVALUED:")
        for _, row in subset.head(10).iterrows():
            name = row.get("player", "Unknown")
            gap = row["value_gap_eur"]
            actual = row.get("market_value_eur", 0)
            pred = row["predicted_value_eur"]
            pct = row.get("value_gap_pct", 0)
            print(f"     {name:25s}  Pred: €{pred:>10,.0f}  "
                  f"Actual: €{actual:>10,.0f}  Gap: +€{gap:>8,.0f} ({pct:+.0f}%)")

        print(f"\n  🔴 Top 10 OVERVALUED:")
        for _, row in subset.tail(10).iterrows():
            name = row.get("player", "Unknown")
            gap = row["value_gap_eur"]
            actual = row.get("market_value_eur", 0)
            pred = row["predicted_value_eur"]
            pct = row.get("value_gap_pct", 0)
            print(f"     {name:25s}  Pred: €{pred:>10,.0f}  "
                  f"Actual: €{actual:>10,.0f}  Gap: €{gap:>8,.0f} ({pct:+.0f}%)")

    return subset


# ── Save model metrics ───────────────────────────────────────────────
def save_metrics(global_result: dict, pos_results: dict, out_dir: str):
    metrics = {
        "global": {
            "r2": global_result["r2"],
            "rmse_eur": global_result["rmse_eur"],
            "cv_r2_mean": global_result["cv_r2_mean"],
            "cv_r2_std": global_result["cv_r2_std"],
            "features": global_result["features"],
            "model_weight": MODEL_WEIGHT,
            "max_gap_pct": MAX_GAP_PCT,
        },
        "position_models": {
            pos: {
                "r2": res["r2"],
                "rmse_eur": res["rmse_eur"],
                "n_players": res["n_players"],
                "features": res["features"],
            }
            for pos, res in pos_results.items()
        },
    }
    path = os.path.join(out_dir, "model_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n💾 Metrics saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train valuation model")
    parser.add_argument("--season", default="2526")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    proc_dir = os.path.join(base, "data", "processed")

    data_path = os.path.join(proc_dir, f"model_dataset_{args.season}.csv")
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("   Run: python src/preprocess.py first")
        return

    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} players from {data_path}")

    if not HAS_XGB:
        print("❌ xgboost is required. Run: pip install xgboost")
        return

    global_result = train_global_model(df)
    pos_results = train_position_models(df)

    output = generate_predictions(df, global_result)

    out_path = os.path.join(proc_dir, f"model_output_{args.season}.csv")
    output.to_csv(out_path, index=False)
    print(f"\n💾 Predictions saved to {out_path}")
    print(f"   {len(output)} players with predictions")

    save_metrics(global_result, pos_results, proc_dir)


if __name__ == "__main__":
    main()
