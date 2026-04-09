"""
model.py — Train XGBoost valuation model with SHAP + position models.

Trains:
  1. Global XGBoost model across all positions
  2. Position-specific models (FW, MF, DF)
  3. SHAP explainability for each model
  4. Confidence intervals via quantile regression

Saves predictions + SHAP values to data/processed/.

Usage:
    python src/model.py
    python src/model.py --season 2425
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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


# ── Feature sets ─────────────────────────────────────────────────────
# Core features used by all models
CORE_FEATURES = [
    "age",
    "contract_years_remaining",
    "league_prestige",
    "is_forward", "is_midfielder", "is_defender",
]

# Performance features (per 90)
PERFORMANCE_FEATURES = [
    "goals_per90", "assists_per90",
    "goals_pens_per90",       # non-penalty goals
    "xg_per90", "xg_assist_per90",  # xG, xAG
    "npxg_per90",             # non-penalty xG
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

# Position-specific features
POSITION_FEATURES = {
    "FW": [
        "goals_per90", "goals_pens_per90", "xg_per90", "npxg_per90",
        "assists_per90", "xg_assist_per90",
        "shots_per90", "shots_on_target_per90", "goals_per_shot",
        "progressive_carries_per90", "successful_dribbles_per90",
        "age", "contract_years_remaining", "league_prestige",
    ],
    "MF": [
        "goals_per90", "assists_per90",
        "xg_per90", "xg_assist_per90",
        "progressive_passes_per90", "progressive_carries_per90",
        "key_passes_per90", "passes_into_final_third_per90",
        "tackles_won_per90", "interceptions_per90",
        "successful_dribbles_per90",
        "age", "contract_years_remaining", "league_prestige",
    ],
    "DF": [
        "tackles_won_per90", "interceptions_per90",
        "aerials_won_per90",
        "progressive_passes_per90", "progressive_carries_per90",
        "passes_into_final_third_per90",
        "goals_per90", "assists_per90",
        "age", "contract_years_remaining", "league_prestige",
    ],
}

ALL_FEATURES = list(set(CORE_FEATURES + PERFORMANCE_FEATURES))


def get_available_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return only the features that exist in the DataFrame."""
    available = [f for f in wanted if f in df.columns]
    missing = set(wanted) - set(available)
    if missing:
        print(f"  ⚠  Missing features (skipped): {missing}")
    return available


def prepare_data(df: pd.DataFrame, features: list[str], target: str):
    """Prepare X, y arrays with NaN handling."""
    available = get_available_features(df, features)
    subset = df[available + [target]].dropna()

    X = subset[available].values
    y = subset[target].values

    return X, y, available, subset.index


# ── Train global model ───────────────────────────────────────────────
def train_global_model(df: pd.DataFrame) -> dict:
    """Train XGBoost on all players."""
    print("\n" + "="*60)
    print("🌍 GLOBAL MODEL (all positions)")
    print("="*60)

    target = "log_market_value"
    X, y, features, idx = prepare_data(df, ALL_FEATURES, target)
    print(f"  Features: {len(features)}, Samples: {len(X)}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(X, y, idx, test_size=0.2, random_state=42)

    # XGBoost
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

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Convert back from log scale for interpretability
    rmse_eur = np.sqrt(mean_squared_error(
        np.expm1(y_test), np.expm1(y_pred)
    ))

    print(f"\n  📊 Results (test set):")
    print(f"     R²:   {r2:.4f}")
    print(f"     RMSE: {rmse:.4f} (log) | €{rmse_eur:,.0f}")
    print(f"     MAE:  {mae:.4f} (log)")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"     CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # SHAP
    shap_values = None
    if HAS_SHAP:
        print("\n  🔍 Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Feature importance ranking
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
    """Train separate models for FW, MF, DF."""
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
    """
    Compute prediction intervals using quantile regression.
    Returns DataFrame with lower/upper bound columns.
    """
    print("\n  📐 Computing confidence intervals...")

    X, y, avail_features, idx = prepare_data(df, features, target)

    # Lower bound (10th percentile)
    model_lo = GradientBoostingRegressor(
        loss="quantile", alpha=0.10,
        n_estimators=300, max_depth=5, random_state=42
    )
    model_lo.fit(X, y)

    # Upper bound (90th percentile)
    model_hi = GradientBoostingRegressor(
        loss="quantile", alpha=0.90,
        n_estimators=300, max_depth=5, random_state=42
    )
    model_hi.fit(X, y)

    # Predict on full dataset
    pred_lo = np.expm1(model_lo.predict(X))
    pred_hi = np.expm1(model_hi.predict(X))

    result = pd.DataFrame(index=idx)
    result["predicted_value_lower"] = pred_lo
    result["predicted_value_upper"] = pred_hi

    print(f"     ✓ 80% prediction intervals computed for {len(result)} players")
    return result


# ── Generate predictions for all players ─────────────────────────────
def generate_predictions(df: pd.DataFrame, global_result: dict) -> pd.DataFrame:
    """Apply model to all players and compute value gaps."""
    print("\n" + "="*60)
    print("📈 GENERATING PREDICTIONS")
    print("="*60)

    model = global_result["model"]
    features = global_result["features"]

    # Prepare full dataset
    X_full, _, _, idx_full = prepare_data(
        df, features, "log_market_value"
    )

    # Predict (log scale)
    log_predictions = model.predict(X_full)

    # Back to EUR
    raw_predictions = np.expm1(log_predictions)

    # Apply adjustment multipliers
    subset = df.loc[idx_full].copy()
    subset["predicted_raw_eur"] = raw_predictions

    # Age-potential adjustment
    if "age_potential_mult" in subset.columns:
        subset["predicted_value_eur"] = (
            subset["predicted_raw_eur"] * subset["age_potential_mult"]
        )
    else:
        subset["predicted_value_eur"] = subset["predicted_raw_eur"]

    # League prestige adjustment
    if "league_prestige" in subset.columns:
        subset["predicted_value_eur"] = (
            subset["predicted_value_eur"] * subset["league_prestige"]
        )

    # Value gap
    if "market_value_eur" in subset.columns:
        subset["value_gap_eur"] = (
            subset["predicted_value_eur"] - subset["market_value_eur"]
        )
        subset["value_gap_pct"] = (
            subset["value_gap_eur"] / subset["market_value_eur"] * 100
        )

    # Confidence intervals
    if HAS_QUANTILE:
        ci = compute_confidence_intervals(df, features, "log_market_value")
        subset = subset.join(ci, how="left")

    # Sort by value gap
    if "value_gap_eur" in subset.columns:
        subset = subset.sort_values("value_gap_eur", ascending=False)

        print(f"\n  🟢 Top 10 UNDERVALUED:")
        for _, row in subset.head(10).iterrows():
            name = row.get("player", "Unknown")
            gap = row["value_gap_eur"]
            actual = row.get("market_value_eur", 0)
            pred = row["predicted_value_eur"]
            print(f"     {name:25s}  Pred: €{pred:>10,.0f}  "
                  f"Actual: €{actual:>10,.0f}  Gap: +€{gap:>10,.0f}")

        print(f"\n  🔴 Top 10 OVERVALUED:")
        for _, row in subset.tail(10).iterrows():
            name = row.get("player", "Unknown")
            gap = row["value_gap_eur"]
            actual = row.get("market_value_eur", 0)
            pred = row["predicted_value_eur"]
            print(f"     {name:25s}  Pred: €{pred:>10,.0f}  "
                  f"Actual: €{actual:>10,.0f}  Gap: €{gap:>10,.0f}")

    return subset


# ── Save model metrics ───────────────────────────────────────────────
def save_metrics(global_result: dict, pos_results: dict, out_dir: str):
    """Save model performance metrics as JSON."""
    metrics = {
        "global": {
            "r2": global_result["r2"],
            "rmse_eur": global_result["rmse_eur"],
            "cv_r2_mean": global_result["cv_r2_mean"],
            "cv_r2_std": global_result["cv_r2_std"],
            "features": global_result["features"],
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
    parser.add_argument("--season", default="2425")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    proc_dir = os.path.join(base, "data", "processed")

    # Load dataset
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

    # Train models
    global_result = train_global_model(df)
    pos_results = train_position_models(df)

    # Generate predictions
    output = generate_predictions(df, global_result)

    # Save
    out_path = os.path.join(proc_dir, f"model_output_{args.season}.csv")
    output.to_csv(out_path, index=False)
    print(f"\n💾 Predictions saved to {out_path}")
    print(f"   {len(output)} players with predictions")

    # Save metrics
    save_metrics(global_result, pos_results, proc_dir)


if __name__ == "__main__":
    main()
