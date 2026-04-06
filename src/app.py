"""
app.py — Soccer Transfer Market Valuation Dashboard

Run with:
    streamlit run src/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Soccer Transfer Valuation",
    page_icon="⚽",
)

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_output_2324.csv"

LEAGUE_COLORS = {
    "eng Premier League": "#3d195b",
    "es La Liga":         "#ee8707",
    "de Bundesliga":      "#d3010c",
    "it Serie A":         "#1a56db",
    "fr Ligue 1":         "#1e4d2b",
}

LEAGUE_LABELS = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
    "fr Ligue 1":         "Ligue 1",
}

# Feature importances from trained XGBoost model (hardcoded — model not re-run)
FEATURE_IMPORTANCES = {
    "Age":                    0.307696,
    "Goals (non-pen) p90":    0.151258,
    "Goals p90":              0.137338,
    "Minutes Played":         0.123635,
    "Assists p90":            0.117250,
    "Tackles Won p90":        0.048869,
    "Position: Defender":     0.044598,
    "Interceptions p90":      0.035768,
    "Position: Forward":      0.033587,
}

STAT_DISPLAY = {
    "goals_p90":           "Goals / 90",
    "assists_p90":         "Assists / 90",
    "goals_non_pen_p90":   "Non-Pen Goals / 90",
    "goal_inv_p90":        "Goal Involvement / 90",
    "tackles_won_p90":     "Tackles Won / 90",
    "interceptions_p90":   "Interceptions / 90",
    "minutes_played":      "Minutes Played",
    "age":                 "Age",
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # EUR → €M for display
    df["actual_m"]    = (df["market_value_in_eur"]  / 1e6).round(1)
    df["predicted_m"] = (df["predicted_value_eur"]   / 1e6).round(1)
    df["gap_m"]       = (df["value_gap_eur"]          / 1e6).round(1)

    # Friendly league label used in legends / tables
    df["league_label"] = df["league"].map(LEAGUE_LABELS)

    # Undervaluation rank (1 = most undervalued)
    df["underval_rank"] = df["value_gap_eur"].rank(ascending=False).astype(int)

    return df


df_all = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    selected_leagues = st.multiselect(
        "League",
        options=list(LEAGUE_LABELS.values()),
        default=list(LEAGUE_LABELS.values()),
    )

    selected_positions = st.multiselect(
        "Position",
        options=["FW", "MF", "DF"],
        default=["FW", "MF", "DF"],
    )

    min_minutes = st.slider(
        "Minimum Minutes Played",
        min_value=900,
        max_value=3000,
        value=900,
        step=50,
    )

    age_range = st.slider(
        "Age Range",
        min_value=16,
        max_value=40,
        value=(16, 40),
    )

    # Apply filters
    mask = (
        df_all["league_label"].isin(selected_leagues)
        & df_all["position_group"].isin(selected_positions)
        & (df_all["minutes_played"] >= min_minutes)
        & df_all["age"].between(age_range[0], age_range[1])
    )
    df = df_all[mask].copy()

    st.divider()
    st.metric("Players shown", len(df))
    st.caption(f"of {len(df_all)} total matched players")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚽ Soccer Transfer Market Valuation Model")
st.markdown(
    "Comparing **predicted vs actual** Transfermarkt market values "
    "across the Big 5 European leagues using **XGBoost**"
)
st.divider()

# Guard: no data after filtering
if df.empty:
    st.warning("No players match the current filters. Adjust the sidebar.")
    st.stop()

# ── Section 1: Scatter plot ───────────────────────────────────────────────────

st.subheader("Predicted vs Actual Market Value")

# Build colour sequence aligned to the filtered leagues (preserves legend order)
present_leagues = df["league_label"].unique().tolist()
color_sequence  = [LEAGUE_COLORS[k] for k, v in LEAGUE_LABELS.items() if v in present_leagues]

fig_scatter = px.scatter(
    df,
    x="actual_m",
    y="predicted_m",
    color="league_label",
    color_discrete_sequence=color_sequence,
    size="minutes_played",
    size_max=16,
    hover_name="player",
    hover_data={
        "team":          True,
        "position_group": True,
        "age":           True,
        "actual_m":      ":.1f",
        "predicted_m":   ":.1f",
        "gap_m":         ":.1f",
        "league_label":  False,
        "minutes_played": False,
    },
    labels={
        "actual_m":      "Actual Market Value (€M)",
        "predicted_m":   "Predicted Market Value (€M)",
        "league_label":  "League",
        "team":          "Club",
        "position_group":"Position",
        "age":           "Age",
        "gap_m":         "Gap (€M)",
    },
    title="Predicted vs Actual Market Value",
)

# Perfect prediction line
max_val = max(df["actual_m"].max(), df["predicted_m"].max()) * 1.05
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val],
    y=[0, max_val],
    mode="lines",
    line=dict(color="#555555", width=1.5, dash="dash"),
    name="Perfect prediction",
    hoverinfo="skip",
))

fig_scatter.update_layout(
    height=560,
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter, Arial, sans-serif", color="#ffffff"),
    legend=dict(title="League", orientation="v", x=1.01, y=1, font=dict(color="#ffffff")),
    xaxis=dict(
        title="Actual Market Value (€M)",
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
        gridcolor="#333333",
        zerolinecolor="#333333",
    ),
    yaxis=dict(
        title="Predicted Market Value (€M)",
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
        gridcolor="#333333",
        zerolinecolor="#333333",
    ),
)
fig_scatter.update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.4, color="#0e1117")),
    selector=dict(mode="markers"),
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Section 2: Leaderboards ───────────────────────────────────────────────────

st.subheader("Player Leaderboards")

TABLE_COLS_RAW = ["player", "league_label", "team", "age",
                  "actual_m", "predicted_m", "gap_m"]
TABLE_HEADERS  = {
    "player":       "Player",
    "league_label": "League",
    "team":         "Club",
    "age":          "Age",
    "actual_m":     "Actual (€M)",
    "predicted_m":  "Predicted (€M)",
    "gap_m":        "Gap (€M)",
}


def _fmt_gap(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}"


def _style_gap(col: pd.Series, positive_is_good: bool) -> list[str]:
    styles = []
    for v in col:
        if positive_is_good:
            color = "color: #16a34a; font-weight:600" if v > 0 else "color: #dc2626; font-weight:600"
        else:
            color = "color: #dc2626; font-weight:600" if v < 0 else "color: #16a34a; font-weight:600"
        styles.append(color)
    return styles


col_under, col_over = st.columns(2)

with col_under:
    st.markdown("### 🟢 Most Undervalued Players")
    st.caption("Model predicts a higher value than Transfermarkt")
    top_under = (
        df.nlargest(15, "gap_m")[TABLE_COLS_RAW]
        .rename(columns=TABLE_HEADERS)
        .reset_index(drop=True)
    )
    top_under.index = top_under.index + 1  # 1-indexed rank
    top_under["Gap (€M)"] = top_under["Gap (€M)"].apply(_fmt_gap)
    st.dataframe(
        top_under.style.apply(
            lambda col: _style_gap(
                top_under["Gap (€M)"].str.replace("+", "", regex=False).astype(float),
                positive_is_good=True,
            ),
            subset=["Gap (€M)"],
        ),
        use_container_width=True,
        height=490,
    )

with col_over:
    st.markdown("### 🔴 Most Overvalued Players")
    st.caption("Model predicts a lower value than Transfermarkt")
    top_over = (
        df.nsmallest(15, "gap_m")[TABLE_COLS_RAW]
        .rename(columns=TABLE_HEADERS)
        .reset_index(drop=True)
    )
    top_over.index = top_over.index + 1
    top_over["Gap (€M)"] = top_over["Gap (€M)"].apply(_fmt_gap)
    st.dataframe(
        top_over.style.apply(
            lambda col: _style_gap(
                top_over["Gap (€M)"].str.replace("+", "", regex=False).astype(float),
                positive_is_good=False,
            ),
            subset=["Gap (€M)"],
        ),
        use_container_width=True,
        height=490,
    )

st.divider()

# ── Section 3: Player search ──────────────────────────────────────────────────

st.subheader("Player Search")

player_options = sorted(df["player"].unique().tolist())
selected_player = st.selectbox(
    "Search for a player",
    options=["— select a player —"] + player_options,
    index=0,
)

if selected_player != "— select a player —":
    row = df[df["player"] == selected_player].iloc[0]

    gap_eur  = row["value_gap_eur"]
    gap_sign = "+" if gap_eur >= 0 else ""

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Actual Value",
        f"€{row['actual_m']:.1f}M",
    )
    m2.metric(
        "Predicted Value",
        f"€{row['predicted_m']:.1f}M",
    )
    m3.metric(
        "Value Gap",
        f"{gap_sign}€{abs(row['gap_m']):.1f}M",
        delta=f"{gap_sign}{row['gap_m']:.1f}M vs actual",
        delta_color="normal" if gap_eur >= 0 else "inverse",
    )
    m4.metric(
        "Undervaluation Rank",
        f"#{int(row['underval_rank'])}",
        help="Rank 1 = most undervalued among all matched players",
    )

    # Player context
    st.caption(
        f"**{row['position_group']}**  ·  {row['team']}  ·  "
        f"{LEAGUE_LABELS.get(row['league'], row['league'])}  ·  "
        f"Age {int(row['age'])}  ·  {int(row['minutes_played'])} min played"
    )

    # Stat table
    stat_rows = []
    for col, label in STAT_DISPLAY.items():
        val = row[col]
        if col in ("minutes_played", "age"):
            stat_rows.append({"Stat": label, "Value": f"{int(val)}"})
        else:
            stat_rows.append({"Stat": label, "Value": f"{val:.3f}"})

    st.dataframe(
        pd.DataFrame(stat_rows).set_index("Stat"),
        use_container_width=False,
    )

st.divider()

# ── Section 4: Feature importance ────────────────────────────────────────────

st.subheader("What Drives Market Value?")
st.caption("XGBoost feature importance (gain) from the trained model")

fi_df = (
    pd.DataFrame.from_dict(FEATURE_IMPORTANCES, orient="index", columns=["Importance"])
    .sort_values("Importance")
    .reset_index()
    .rename(columns={"index": "Feature"})
)

fig_fi = px.bar(
    fi_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text=fi_df["Importance"].apply(lambda v: f"{v:.1%}"),
    title="What Drives Market Value? (XGBoost Feature Importance)",
    color="Importance",
    color_continuous_scale=[[0, "#e8f4fd"], [1, "#1a56db"]],
)
fig_fi.update_traces(
    textposition="outside",
    marker_line_width=0,
)
fig_fi.update_layout(
    height=400,
    margin=dict(l=20, r=80, t=50, b=20),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter, Arial, sans-serif", color="#ffffff"),
    coloraxis_showscale=False,
    xaxis=dict(
        title="Feature Importance (Gain)",
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
        tickformat=".0%",
        gridcolor="#333333",
        zerolinecolor="#333333",
    ),
    yaxis=dict(
        title="",
        tickfont=dict(color="#ffffff"),
        gridcolor="#333333",
        zerolinecolor="#333333",
    ),
)

st.plotly_chart(fig_fi, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: FBref (2023–24 season) · Transfermarkt (via Kaggle)  |  "
    "Model: XGBoost · R²=0.475 · RMSE=€20.4M on test set  |  "
    "Portfolio project — not for commercial use"
)
