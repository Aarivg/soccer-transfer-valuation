"""
model.py

Phase 2: Train and evaluate market-value prediction models.

Features used
-------------
All features below are fully populated for every matched player.

    goals_p90               – goals per 90 min
    assists_p90             – assists per 90 min
    goals_non_pen_p90       – non-penalty goals per 90 (removes PK noise)
    tackles_won_p90         – defensive output
    interceptions_p90       – defensive positioning
    age                     – strong negative predictor; market discounts older players
    minutes_played          – proxy for fitness / coach trust
    pos_DF / pos_FW / pos_MF – position one-hot (MF is reference, dropped)

Features requested but unavailable in the current CSV exports
-------------------------------------------------------------
    xG, xAG, progressive passes/carries, pressures
    → These require additional FBref tab exports (Expected Goals, Misc).
      When those files are added and preprocess.py is re-run, add the
      corresponding _p90 columns to FEATURE_COLS below.

Target
------
    log_market_value  (log1p of EUR value — right-skewed, stabilises variance)
    Predictions are exponentiated back to EUR for the output file.

Models
------
    1. Ridge regression   – linear baseline with L2 regularisation
    2. XGBoost            – gradient-boosted trees, tuned with 5-fold CV
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "model_dataset_2324.csv"
OUTPUT_FILE   = PROCESSED_DIR / "model_output_2324.csv"

# ── Feature / target config ───────────────────────────────────────────────────

FEATURE_COLS = [
    "goals_p90",
    "assists_p90",
    "goals_non_pen_p90",
    "tackles_won_p90",
    "interceptions_p90",
    "age",
    "minutes_played",
    # Added from fbref_shooting_2324.csv (r > 0.2 with target)
    "shots_p90",
    "sot_p90",
    "g_per_shot",
    # Added from players.csv via TM player_id chain (87.4% coverage, r=0.444)
    "contract_years_remaining",
]

# Baseline metrics from the 10-feature model (shooting features added)
BASELINE_R2   = 0.4856
BASELINE_RMSE = 20_132_528

POSITION_COL = "position_group"   # will be one-hot encoded; MF dropped as reference
TARGET_COL   = "log_market_value"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ── XGBoost hyperparameter search space ───────────────────────────────────────

XGB_PARAM_DIST = {
    "xgb__n_estimators":      [200, 400, 600, 800],
    "xgb__max_depth":         [3, 4, 5, 6],
    "xgb__learning_rate":     [0.01, 0.03, 0.05, 0.10],
    "xgb__subsample":         [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "xgb__reg_alpha":         [0, 0.1, 0.5, 1.0],
    "xgb__reg_lambda":        [1.0, 2.0, 5.0],
    "xgb__min_child_weight":  [1, 3, 5],
}

N_ITER_SEARCH = 60   # random search iterations


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load model_dataset_2324.csv, restrict to rows with a market value,
    one-hot encode position, and return (df_matched, X, y).

    Columns in FEATURE_COLS that are absent from the CSV are silently dropped
    with a warning, so the model degrades gracefully if a file is missing.
    """
    df = pd.read_csv(INPUT_FILE)

    total = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    print(f"Rows with market value: {len(df)} / {total}  "
          f"({total - len(df)} dropped — no Transfermarkt match)")

    # Filter to columns that are present and have at least some non-null values
    available = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [WARN] Feature '{col}' not found in dataset — skipped")
        elif df[col].notna().sum() == 0:
            print(f"  [WARN] Feature '{col}' is all-null — skipped")
        else:
            null_count = df[col].isna().sum()
            if null_count > 0:
                df[col] = df[col].fillna(df[col].median())
                print(f"  [INFO] Feature '{col}': {null_count} nulls filled with median")
            available.append(col)

    # One-hot encode position (drop MF as reference category)
    pos_dummies = pd.get_dummies(df[POSITION_COL], prefix="pos", drop_first=False)
    pos_dummies = pos_dummies.drop(columns=["pos_MF"], errors="ignore")

    X = pd.concat([df[available], pos_dummies], axis=1).astype(float)
    y = df[TARGET_COL].values

    print(f"Feature matrix: {X.shape}  (features: {list(X.columns)})")
    return df, X, y


# ── Metrics helper ────────────────────────────────────────────────────────────

def _rmse_eur(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSE in EUR space (after exponentiating log predictions)."""
    return float(np.sqrt(mean_squared_error(
        np.expm1(y_true_log), np.expm1(y_pred_log)
    )))


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse_log = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2       = float(r2_score(y_true, y_pred))
    rmse_eur = _rmse_eur(y_true, y_pred)
    print(f"\n{name}")
    print(f"  R²              : {r2:.4f}")
    print(f"  RMSE (log scale): {rmse_log:.4f}")
    print(f"  RMSE (EUR)      : €{rmse_eur:,.0f}")
    return {"model": name, "r2": r2, "rmse_log": rmse_log, "rmse_eur": rmse_eur}


# ── Model training ────────────────────────────────────────────────────────────

def train_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge()),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Tune alpha over a log-spaced grid with CV
    best_alpha, best_cv_rmse = 1.0, np.inf
    for alpha in np.logspace(-2, 4, 30):
        fold_rmses = []
        for train_idx, val_idx in cv.split(X_train):
            p = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            p.fit(X_train[train_idx], y_train[train_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(
                y_train[val_idx], p.predict(X_train[val_idx])
            )))
        cv_rmse = float(np.mean(fold_rmses))
        if cv_rmse < best_cv_rmse:
            best_cv_rmse, best_alpha = cv_rmse, alpha

    print(f"  Ridge best alpha (5-fold CV): {best_alpha:.4f}  "
          f"CV-RMSE(log): {best_cv_rmse:.4f}")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=best_alpha)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    pipe = Pipeline([("xgb", base_xgb)])
    cv   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=XGB_PARAM_DIST,
        n_iter=N_ITER_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_params_
    best_cv = -search.best_score_

    print(f"  XGBoost best CV-RMSE(log): {best_cv:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best.items()):
        print(f"    {k.replace('xgb__', ''):<22}: {v}")

    return search.best_estimator_


# ── Post-prediction adjustment multipliers ────────────────────────────────────

def _age_multiplier(age: int) -> float:
    """
    Younger players command higher values due to sell-on potential
    and longer peak years ahead.
    """
    if age <= 20:
        return 1.50
    if age <= 23:
        return 1.25
    if age <= 27:
        return 1.00
    if age <= 30:
        return 0.85
    return 0.70


LEAGUE_MULTIPLIERS = {
    "eng Premier League": 1.30,
    "es La Liga":         1.20,
    "de Bundesliga":      1.10,
    "it Serie A":         1.05,
    "fr Ligue 1":         1.00,
}


def apply_adjustments(out: pd.DataFrame) -> pd.DataFrame:
    """
    Apply age-potential and league-prestige multipliers sequentially to
    predicted_value_eur, then recalculate value_gap_eur.

    Adds:
        age_multiplier        – factor from age bracket
        league_multiplier     – factor from league prestige
        adjustment_factor     – combined multiplier (age × league)
        predicted_value_eur   – overwritten with adjusted value
        value_gap_eur         – adjusted predicted − actual (EUR)
    """
    out["age_multiplier"]    = out["age"].apply(_age_multiplier)
    out["league_multiplier"] = out["league"].map(LEAGUE_MULTIPLIERS).fillna(1.0)
    out["adjustment_factor"] = (out["age_multiplier"] * out["league_multiplier"]).round(4)

    # Apply sequentially: age first, then league (equivalent to one combined multiply)
    raw_pred = out["predicted_value_eur"].copy()
    out["predicted_value_eur"] = (raw_pred * out["adjustment_factor"]).round(0)
    out["value_gap_eur"]       = (out["predicted_value_eur"] - out["market_value_in_eur"]).round(0)
    return out


# ── Prediction & output ───────────────────────────────────────────────────────

def build_output(
    df: pd.DataFrame,
    X: np.ndarray,
    best_model,
    winner_name: str,
) -> pd.DataFrame:
    """
    Generate raw XGBoost predictions, apply age-potential and league-prestige
    multipliers, then attach all columns to the matched DataFrame.

    value_gap_eur = adjusted predicted − actual
        positive → model thinks player is undervalued
        negative → model thinks player is overvalued
    """
    log_preds     = best_model.predict(X)
    predicted_eur = np.expm1(log_preds)

    out = df.copy()
    out["predicted_value_eur"] = predicted_eur.round(0)
    out["model_used"]          = winner_name

    out = apply_adjustments(out)
    return out


def print_top_players(df: pd.DataFrame, n: int = 10) -> None:
    cols = ["player", "team", "league", "position_group", "age",
            "adjustment_factor", "market_value_in_eur",
            "predicted_value_eur", "value_gap_eur"]
    cols = [c for c in cols if c in df.columns]

    def _fmt(val):
        return f"€{val/1e6:.1f}M"

    print(f"\n{'─'*80}")
    print(f"TOP {n} MOST UNDERVALUED  (after age + league adjustments)")
    print(f"{'─'*80}")
    under = df.nlargest(n, "value_gap_eur")[cols].copy()
    for _, row in under.iterrows():
        adj = f"  adj={row['adjustment_factor']:.2f}x" if "adjustment_factor" in row else ""
        print(f"  {row['player']:<28} {row['team']:<22} age={int(row['age'])}{adj}  "
              f"actual={_fmt(row['market_value_in_eur'])}  "
              f"predicted={_fmt(row['predicted_value_eur'])}  "
              f"gap={_fmt(row['value_gap_eur'])}")

    print(f"\n{'─'*80}")
    print(f"TOP {n} MOST OVERVALUED   (after age + league adjustments)")
    print(f"{'─'*80}")
    over = df.nsmallest(n, "value_gap_eur")[cols].copy()
    for _, row in over.iterrows():
        adj = f"  adj={row['adjustment_factor']:.2f}x" if "adjustment_factor" in row else ""
        print(f"  {row['player']:<28} {row['team']:<22} age={int(row['age'])}{adj}  "
              f"actual={_fmt(row['market_value_in_eur'])}  "
              f"predicted={_fmt(row['predicted_value_eur'])}  "
              f"gap={_fmt(row['value_gap_eur'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SOCCER TRANSFER VALUATION MODEL — PHASE 2")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    df, X, y = load_data()
    X_arr = X.values

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_arr, y, df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("\n── Ridge Regression ─────────────────────────────────────────")
    ridge = train_ridge(X_train, y_train)
    ridge_metrics = evaluate("Ridge (test set)", y_test, ridge.predict(X_test))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n── XGBoost (RandomizedSearchCV, 60 iters, 5-fold) ──────────")
    xgb = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate("XGBoost (test set)", y_test, xgb.predict(X_test))

    # ── Compare (vs each other and vs baseline) ───────────────────────────────
    print("\n── Model comparison ─────────────────────────────────────────")
    header = f"{'Model':<34} {'R²':>8} {'ΔR² vs base':>13} {'RMSE(EUR)':>14} {'ΔRMSE vs base':>15}"
    print(header)
    print("-" * len(header))
    for m in (ridge_metrics, xgb_metrics):
        dr2   = m["r2"]       - BASELINE_R2
        drmse = m["rmse_eur"] - BASELINE_RMSE
        print(f"  {m['model']:<32} {m['r2']:>8.4f} "
              f"  {dr2:>+8.4f}     "
              f"€{m['rmse_eur']/1e6:>10.2f}M "
              f"  {drmse/1e6:>+8.2f}M")
    print(f"  {'Baseline (7-feat XGBoost)':<32} {BASELINE_R2:>8.4f} "
          f"  {'—':>8}     "
          f"€{BASELINE_RMSE/1e6:>10.2f}M "
          f"  {'—':>8}")

    if xgb_metrics["r2"] >= ridge_metrics["r2"]:
        winner, winner_name, loser_name = xgb, "XGBoost", "Ridge"
    else:
        winner, winner_name, loser_name = ridge, "Ridge", "XGBoost"

    r2_delta   = abs(xgb_metrics["r2"]       - ridge_metrics["r2"])
    rmse_delta = abs(xgb_metrics["rmse_eur"] - ridge_metrics["rmse_eur"])
    print(f"\nWinner: {winner_name}  "
          f"(ΔR²={r2_delta:.4f}, ΔRMSE=€{rmse_delta/1e6:.2f}M vs {loser_name})")

    xgb_r2_gain = xgb_metrics["r2"] - BASELINE_R2
    print(f"XGBoost vs baseline: ΔR²={xgb_r2_gain:+.4f}  "
          f"({'improvement' if xgb_r2_gain > 0 else 'no improvement'} from 3 new shooting features)")

    # ── Feature importance (XGBoost) ──────────────────────────────────────────
    print("\n── XGBoost feature importances (gain) ───────────────────────")
    xgb_step = xgb.named_steps["xgb"]
    importances = pd.Series(
        xgb_step.feature_importances_, index=list(X.columns)
    ).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    # ── Predictions on full matched dataset ───────────────────────────────────
    out_df = build_output(df, X_arr, winner, winner_name)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved predictions → {OUTPUT_FILE}  ({len(out_df)} rows, {len(out_df.columns)} cols)")

    # ── Top over/under-valued ─────────────────────────────────────────────────
    print_top_players(out_df, n=10)

    # ── Summary stats on gaps ─────────────────────────────────────────────────
    print(f"\n── Value gap distribution ───────────────────────────────────")
    gap = out_df["value_gap_eur"]
    print(f"  Mean gap  : €{gap.mean()/1e6:+.1f}M")
    print(f"  Median gap: €{gap.median()/1e6:+.1f}M")
    print(f"  Undervalued (gap > 0): {(gap > 0).sum()} players")
    print(f"  Overvalued  (gap < 0): {(gap < 0).sum()} players")


if __name__ == "__main__":
    main()
