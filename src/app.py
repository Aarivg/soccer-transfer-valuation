"""
app.py — Soccer Transfer Market Valuation Dashboard (V4 — Final)

Features:
  - Interactive scatter plot with hover details
  - Under/Overvalued leaderboards
  - Player search with photo, confidence interval, stat card
  - Transfer Recommendation Engine (position + budget + age)
  - Player comparison with radar charts + photos
  - League-level analysis (avg under/overvaluation by league)
  - SHAP explainability + value gap distribution
  - CSV export for filtered results
  - Historical value trends per player
  - AI Scouting Assistant (powered by Claude API)
  - Player Similarity Engine (cosine similarity)
  - Last-updated timestamp

Run with:
    streamlit run src/app.py
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import requests as http_requests
import re

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Soccer Transfer Valuation",
    page_icon="⚽",
)

# ── Constants ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

# Auto-detect season
if (PROCESSED_DIR / "model_output_2526.csv").exists():
    DATA_PATH = PROCESSED_DIR / "model_output_2526.csv"
    SEASON = "2025-26"
elif (PROCESSED_DIR / "model_output_2324.csv").exists():
    DATA_PATH = PROCESSED_DIR / "model_output_2324.csv"
    SEASON = "2023-24"
else:
    DATA_PATH = None
    SEASON = "Unknown"

METRICS_PATH = PROCESSED_DIR / "model_metrics.json"
VALUATIONS_PATH = RAW_DIR / "player_valuations.csv"

LEAGUE_COLORS = {
    "Premier League": "#3d195b",
    "La Liga":        "#ee8707",
    "Bundesliga":     "#d3010c",
    "Serie A":        "#1a56db",
    "Ligue 1":        "#1e4d2b",
}

POSITION_LABELS = {"FW": "Forward", "MF": "Midfielder", "DF": "Defender"}

RADAR_STATS = {
    "FW": {
        "goals_per90": "Goals",
        "xg_per90": "xG",
        "assists_per90": "Assists",
        "shots_on_target_per90": "Shots OT",
        "progressive_carries_per90": "Prog Carries",
        "successful_dribbles_per90": "Dribbles",
    },
    "MF": {
        "goals_per90": "Goals",
        "assists_per90": "Assists",
        "progressive_passes_per90": "Prog Passes",
        "progressive_carries_per90": "Prog Carries",
        "key_passes_per90": "Key Passes",
        "tackles_won_per90": "Tackles",
    },
    "DF": {
        "tackles_won_per90": "Tackles",
        "interceptions_per90": "Interceptions",
        "aerials_won_per90": "Aerials",
        "progressive_passes_per90": "Prog Passes",
        "progressive_carries_per90": "Prog Carries",
        "goals_per90": "Goals",
    },
}

BG = "#0e1117"
CARD = "#1a1d24"


# ── Data loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH is None:
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    for src, dst in [("market_value_eur", "actual_m"),
                     ("predicted_value_eur", "predicted_m"),
                     ("value_gap_eur", "gap_m")]:
        if src in df.columns:
            df[dst] = (df[src] / 1e6).round(1)
    for c in ["predicted_value_lower", "predicted_value_upper"]:
        if c in df.columns:
            df[c + "_m"] = (df[c] / 1e6).round(1)
    if "tm_highest_market_value_in_eur" in df.columns:
        df["peak_m"] = (pd.to_numeric(df["tm_highest_market_value_in_eur"],
                                       errors="coerce") / 1e6).round(1)
    if "league_name" in df.columns:
        df["league_label"] = df["league_name"]
    elif "comp" in df.columns:
        df["league_label"] = df["comp"]
    if "position_group" not in df.columns and "pos" in df.columns:
        pos = df["pos"].astype(str).str.upper()
        df["position_group"] = "MF"
        df.loc[pos.str.contains("FW"), "position_group"] = "FW"
        df.loc[pos.str.contains("DF"), "position_group"] = "DF"
    return df


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_valuations_history() -> pd.DataFrame:
    if VALUATIONS_PATH.exists():
        v = pd.read_csv(VALUATIONS_PATH)
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        v["value_m"] = (pd.to_numeric(v["market_value_in_eur"],
                                       errors="coerce") / 1e6).round(1)
        return v
    return pd.DataFrame()


df = load_data()
metrics = load_metrics()
val_history = load_valuations_history()

if df.empty:
    st.error("No data found. Run the pipeline first.")
    st.stop()

model_weight = metrics.get("global", {}).get("model_weight", 0.65)


# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Page layout ── */
    .main .block-container { padding-top: 0.5rem; max-width: 1500px; }

    /* ── Tab styling — bigger, bolder ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: #111318;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #2a2d34;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 14px 24px;
        font-size: 1.05rem;
        font-weight: 500;
        border-radius: 8px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background: #1e40af !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: #1a1d24;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d24 0%, #1e2028 100%);
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #2a2d34;
        transition: border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0c0e12;
        border-right: 1px solid #1a1d24;
    }
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 500;
        color: #d1d5db;
    }

    /* ── Dataframes ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Headers ── */
    h1 { font-size: 2.2rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }

    /* ── Buttons ── */
    .stDownloadButton button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stDownloadButton button:hover {
        background: #2563eb;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #d1d5db;
    }

    /* ── Dividers ── */
    hr { border-color: #1a1d24 !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <h1 style="margin:0; font-size:2.4rem; font-weight:800;">
        ⚽ Soccer Transfer Valuation Model
    </h1>
</div>
""", unsafe_allow_html=True)
st.caption(
    f"Comparing predicted market values (from on-pitch stats) vs actual "
    f"Transfermarkt valuations · **{SEASON}** · Big 5 European Leagues · "
    f"**{len(df)}** players"
)

# Last updated
if DATA_PATH and DATA_PATH.exists():
    mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
    st.caption(f"📅 Last updated: {mod_time.strftime('%B %d, %Y')}")


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    all_leagues = sorted(df["league_label"].dropna().unique().tolist())
    selected_leagues = st.multiselect("League", all_leagues, default=all_leagues)

    all_positions = sorted(df["position_group"].dropna().unique().tolist())
    selected_positions = st.multiselect(
        "Position", all_positions, default=all_positions,
        format_func=lambda x: POSITION_LABELS.get(x, x),
    )

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    min_minutes = st.slider("Min. Minutes Played", 900,
                            int(df.get("minutes", pd.Series([3500])).max()),
                            900, step=100) if "minutes" in df.columns else 900

    st.divider()
    st.caption(
        f"Model: XGBoost\n\n"
        f"Blended: {int(model_weight*100)}% model / "
        f"{int((1-model_weight)*100)}% market\n\n"
        f"Gap cap: ±60%"
    )

# Apply filters
mask = (df["league_label"].isin(selected_leagues) &
        df["position_group"].isin(selected_positions))
if "age" in df.columns:
    mask &= df["age"].between(age_range[0], age_range[1])
if "minutes" in df.columns:
    mask &= df["minutes"] >= min_minutes

filtered = df[mask].copy()
if filtered.empty:
    st.warning("No players match filters.")
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🔍 Transfer Finder", "⚔️ Compare",
    "🏟️ League Analysis", "🧠 Explainability",
    "🤖 AI Scout", "🔗 Similar Players"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predicted vs Actual Market Value")
    color_seq = [LEAGUE_COLORS.get(l, "#666") for l in filtered["league_label"].unique()]

    fig = px.scatter(
        filtered, x="actual_m", y="predicted_m",
        color="league_label", color_discrete_sequence=color_seq,
        size="actual_m", size_max=16, hover_name="player",
        hover_data={"squad": True, "position_group": True, "age": True,
                    "actual_m": ":.1f", "predicted_m": ":.1f",
                    "gap_m": ":.1f", "league_label": False},
        labels={"actual_m": "Actual Value (€M)", "predicted_m": "Predicted Value (€M)",
                "league_label": "League", "squad": "Club",
                "position_group": "Position", "gap_m": "Gap (€M)"},
    )
    mx = max(filtered["actual_m"].max(), filtered["predicted_m"].max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(color="#555", width=1.5, dash="dash"),
                             name="Perfect prediction"))
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#fff"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=0.5),
                      xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"))
    st.plotly_chart(fig, use_container_width=True)

    # Leaderboards
    col_u, col_o = st.columns(2)
    for col, title, emoji, ascending in [
        (col_u, "Most Undervalued", "🟢", False),
        (col_o, "Most Overvalued", "🔴", True),
    ]:
        with col:
            st.markdown(f"### {emoji} {title}")
            board = (filtered.nlargest(15, "gap_m") if not ascending
                     else filtered.nsmallest(15, "gap_m"))
            show = board[["player", "squad", "position_group", "age",
                          "actual_m", "predicted_m", "gap_m"]].copy()
            show.columns = ["Player", "Club", "Pos", "Age",
                            "Actual (€M)", "Pred (€M)", "Gap (€M)"]
            show.index = range(1, len(show) + 1)
            st.dataframe(show, use_container_width=True, height=480)

    st.divider()

    # ── Player Search with Photo ─────────────────────────────────
    st.subheader("🔎 Player Search")
    player_opts = sorted(filtered["player"].dropna().unique().tolist())
    sel = st.selectbox("Search for a player",
                       ["— select —"] + player_opts, index=0)

    if sel != "— select —":
        row = filtered[filtered["player"] == sel].iloc[0]

        col_photo, col_info = st.columns([1, 3])

        with col_photo:
            img = row.get("tm_image_url", "")
            if pd.notna(img) and str(img).startswith("http"):
                st.image(str(img), width=160)

            # Transfermarkt link
            tm_url = row.get("tm_url", "")
            if pd.notna(tm_url) and str(tm_url).startswith("http"):
                st.markdown(f"[View on Transfermarkt ↗]({tm_url})")

        with col_info:
            pos_label = POSITION_LABELS.get(row.get("position_group", ""), "")
            squad = row.get("squad", "")
            league = row.get("league_label", "")
            nationality = row.get("nation", row.get("tm_country_of_citizenship", ""))
            if pd.isna(nationality):
                nationality = ""

            st.markdown(f"### {sel}")
            st.caption(f"**{pos_label}** · {squad} · {league} · {nationality}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Actual Value", f"€{row['actual_m']:.1f}M")
            m2.metric("Predicted", f"€{row['predicted_m']:.1f}M")
            gap = row.get("gap_m", 0)
            pct = row.get("value_gap_pct", 0)
            m3.metric("Gap", f"€{abs(gap):.1f}M",
                      delta=f"{'Under' if gap >= 0 else 'Over'}valued ({pct:+.0f}%)"
                      if pd.notna(pct) else None,
                      delta_color="normal" if gap >= 0 else "inverse")
            age_v = row.get("age", None)
            m4.metric("Age", f"{int(age_v)}" if pd.notna(age_v) else "N/A")

            # Extra info row
            e1, e2, e3, e4 = st.columns(4)
            contract = row.get("contract_years_remaining", None)
            e1.metric("Contract", f"{contract:.1f} yrs" if pd.notna(contract) else "N/A")

            peak = row.get("peak_m", None)
            e2.metric("Peak Value", f"€{peak:.1f}M" if pd.notna(peak) else "N/A")

            caps = row.get("tm_international_caps", None)
            goals_int = row.get("tm_international_goals", None)
            if pd.notna(caps):
                e3.metric("Int'l Caps", f"{int(caps)}")
            if pd.notna(goals_int):
                e4.metric("Int'l Goals", f"{int(goals_int)}")

        # Confidence interval
        lo = row.get("predicted_value_lower_m", None)
        hi = row.get("predicted_value_upper_m", None)
        if pd.notna(lo) and pd.notna(hi):
            st.info(f"📐 **80% Confidence Interval:** €{lo:.1f}M – €{hi:.1f}M")

        # Historical value trend
        pid = row.get("tm_player_id", None)
        if pd.notna(pid) and not val_history.empty:
            player_hist = val_history[
                val_history["player_id"] == int(pid)
            ].sort_values("date")
            if len(player_hist) > 2:
                st.markdown("#### 📈 Market Value History")
                fig_hist = px.line(
                    player_hist, x="date", y="value_m",
                    labels={"date": "", "value_m": "Market Value (€M)"},
                )
                fig_hist.update_traces(line_color="#3b82f6", line_width=2)
                fig_hist.update_layout(
                    height=250, margin=dict(l=20, r=20, t=10, b=20),
                    paper_bgcolor=BG, plot_bgcolor=BG,
                    font=dict(color="#fff"),
                    xaxis=dict(gridcolor="#1a1d24"),
                    yaxis=dict(gridcolor="#1a1d24"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: TRANSFER FINDER
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Transfer Recommendation Engine")
    st.caption("Find undervalued players by position, budget, and age")

    f1, f2, f3 = st.columns(3)
    with f1:
        finder_pos = st.selectbox("Position", ["All"] + list(POSITION_LABELS.keys()),
                                  format_func=lambda x: "All" if x == "All"
                                  else POSITION_LABELS.get(x, x))
    with f2:
        max_budget = st.slider("Max Budget (€M)", 1, 200, 50)
    with f3:
        max_age_finder = st.slider("Max Age", 18, 38, 28)

    fdf = filtered.copy()
    if finder_pos != "All":
        fdf = fdf[fdf["position_group"] == finder_pos]
    fdf = fdf[(fdf["actual_m"] <= max_budget) &
              (fdf["age"] <= max_age_finder) &
              (fdf["gap_m"] > 0)].sort_values("gap_m", ascending=False)

    if fdf.empty:
        st.info("No undervalued players match. Try wider filters.")
    else:
        st.success(f"Found **{len(fdf)}** undervalued players under €{max_budget}M")

        show = fdf[["player", "squad", "position_group", "age",
                     "actual_m", "predicted_m", "gap_m"]].head(25).copy()
        show.columns = ["Player", "Club", "Pos", "Age",
                        "Price (€M)", "Model Value (€M)", "Bargain (€M)"]
        show.index = range(1, len(show) + 1)
        st.dataframe(show, use_container_width=True, height=600)

        # Export button
        csv = fdf[["player", "squad", "position_group", "age",
                    "actual_m", "predicted_m", "gap_m"]].to_csv(index=False)
        st.download_button("📥 Export results as CSV", csv,
                           "transfer_recommendations.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: COMPARE PLAYERS
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚔️ Player Comparison")
    opts = sorted(filtered["player"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    with c1:
        p1 = st.selectbox("Player 1", ["—"] + opts, key="c1")
    with c2:
        p2 = st.selectbox("Player 2", ["—"] + opts, key="c2")
    with c3:
        p3 = st.selectbox("Player 3 (optional)", ["—"] + opts, key="c3")

    sels = [p for p in [p1, p2, p3] if p != "—"]

    if len(sels) >= 2:
        cdf = filtered[filtered["player"].isin(sels)]

        # Photo + value cards
        pcols = st.columns(len(sels))
        for i, (_, row) in enumerate(cdf.iterrows()):
            with pcols[i]:
                img = row.get("tm_image_url", "")
                if pd.notna(img) and str(img).startswith("http"):
                    st.image(str(img), width=100)
                st.markdown(f"**{row['player']}**")
                st.caption(f"{row.get('squad', '')} · {int(row.get('age', 0))}")
                st.metric("Actual", f"€{row['actual_m']:.1f}M")
                st.metric("Predicted", f"€{row['predicted_m']:.1f}M")
                g = row.get("gap_m", 0)
                st.metric("Gap", f"{'+'if g>=0 else ''}€{g:.1f}M",
                          delta_color="normal" if g >= 0 else "inverse")

        # Radar chart
        st.markdown("#### 📊 Stats Radar")
        positions = cdf["position_group"].mode()
        rpos = positions.iloc[0] if not positions.empty else "MF"
        rcfg = RADAR_STATS.get(rpos, RADAR_STATS["MF"])
        avail = {k: v for k, v in rcfg.items() if k in cdf.columns}

        if len(avail) >= 3:
            fig_r = go.Figure()
            colors = ["#3b82f6", "#ef4444", "#22c55e"]
            for i, (_, row) in enumerate(cdf.iterrows()):
                vals = []
                for sc in avail:
                    v = pd.to_numeric(row.get(sc, 0), errors="coerce")
                    vals.append(v if pd.notna(v) else 0)
                # Normalize to percentile
                for j, sc in enumerate(avail):
                    cd = pd.to_numeric(filtered[sc], errors="coerce")
                    mx = cd.quantile(0.95)
                    vals[j] = min(vals[j] / mx * 100, 100) if mx > 0 else 0
                fig_r.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=list(avail.values()) + [list(avail.values())[0]],
                    fill="toself", name=row["player"],
                    line=dict(color=colors[i % 3]), opacity=0.6))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                           showticklabels=False),
                           bgcolor=BG),
                height=420, paper_bgcolor=BG, font=dict(color="#fff"),
                margin=dict(l=60, r=60, t=30, b=30))
            st.plotly_chart(fig_r, use_container_width=True)
    elif len(sels) == 1:
        st.info("Select at least 2 players to compare.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: LEAGUE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🏟️ League-Level Analysis")
    st.caption("Which leagues have the most undervalued or overvalued players?")

    if "value_gap_pct" in filtered.columns and "league_label" in filtered.columns:
        league_stats = filtered.groupby("league_label").agg(
            avg_gap_pct=("value_gap_pct", "mean"),
            median_gap_pct=("value_gap_pct", "median"),
            avg_actual=("actual_m", "mean"),
            avg_predicted=("predicted_m", "mean"),
            n_players=("player", "count"),
            n_undervalued=("gap_m", lambda x: (x > 0).sum()),
        ).reset_index()
        league_stats["pct_undervalued"] = (
            league_stats["n_undervalued"] / league_stats["n_players"] * 100
        ).round(1)
        league_stats = league_stats.sort_values("avg_gap_pct", ascending=False)

        # Bar chart: avg gap by league
        fig_league = px.bar(
            league_stats, x="league_label", y="avg_gap_pct",
            color="league_label",
            color_discrete_map=LEAGUE_COLORS,
            labels={"league_label": "", "avg_gap_pct": "Avg Gap (%)"},
            text=league_stats["avg_gap_pct"].apply(lambda x: f"{x:+.1f}%"),
        )
        fig_league.update_layout(
            height=400, showlegend=False,
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#fff"),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_league.update_traces(textposition="outside")
        st.plotly_chart(fig_league, use_container_width=True)

        # Stats table
        st.markdown("#### Summary")
        show_ls = league_stats[["league_label", "n_players", "avg_actual",
                                "avg_predicted", "avg_gap_pct",
                                "pct_undervalued"]].copy()
        show_ls.columns = ["League", "Players", "Avg Actual (€M)",
                           "Avg Predicted (€M)", "Avg Gap (%)",
                           "% Undervalued"]
        show_ls["Avg Actual (€M)"] = show_ls["Avg Actual (€M)"].round(1)
        show_ls["Avg Predicted (€M)"] = show_ls["Avg Predicted (€M)"].round(1)
        show_ls["Avg Gap (%)"] = show_ls["Avg Gap (%)"].apply(
            lambda x: f"{x:+.1f}%")
        show_ls.index = range(1, len(show_ls) + 1)
        st.dataframe(show_ls, use_container_width=True)

        st.caption(
            "**Positive gap** = league's players are undervalued on average "
            "(stats predict higher). **Negative** = overvalued."
        )

    # Position breakdown
    st.divider()
    st.markdown("#### By Position")
    if "position_group" in filtered.columns and "value_gap_pct" in filtered.columns:
        pos_stats = filtered.groupby("position_group").agg(
            avg_gap=("value_gap_pct", "mean"),
            avg_value=("actual_m", "mean"),
            count=("player", "count"),
        ).reset_index()
        pos_stats["position_group"] = pos_stats["position_group"].map(POSITION_LABELS)

        p1, p2, p3 = st.columns(3)
        for i, (_, row) in enumerate(pos_stats.iterrows()):
            with [p1, p2, p3][i]:
                st.metric(row["position_group"],
                          f"€{row['avg_value']:.1f}M avg",
                          delta=f"{row['avg_gap']:+.1f}% avg gap")
                st.caption(f"{int(row['count'])} players")


# ═══════════════════════════════════════════════════════════════════
# TAB 5: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧠 What Drives Market Value?")

    # Position model performance
    pos_models = metrics.get("position_models", {})
    if pos_models:
        st.markdown("#### Position-Specific Model Performance")
        pcols = st.columns(len(pos_models))
        for i, (pos, data) in enumerate(pos_models.items()):
            with pcols[i]:
                st.markdown(f"**{POSITION_LABELS.get(pos, pos)}**")
                st.metric("R²", f"{data['r2']:.3f}")
                st.metric("RMSE", f"€{data['rmse_eur']/1e6:.1f}M")
                st.caption(f"{data['n_players']} players")

    st.divider()

    # Value gap distribution
    st.markdown("#### Value Gap Distribution")
    if "value_gap_pct" in filtered.columns:
        fig_d = px.histogram(
            filtered, x="value_gap_pct", nbins=40,
            color_discrete_sequence=["#3b82f6"],
            labels={"value_gap_pct": "Value Gap (%)"},
        )
        fig_d.add_vline(x=0, line_dash="dash", line_color="#555")
        fig_d.update_layout(height=320, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color="#fff"), showlegend=False,
                            xaxis=dict(gridcolor="#222"),
                            yaxis=dict(title="Players", gridcolor="#222"),
                            margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_d, use_container_width=True)

    st.divider()

    # Global model stats
    g = metrics.get("global", {})
    st.markdown("#### Model Details")
    d1, d2, d3, d4 = st.columns(4)
    if "r2" in g:
        d1.metric("R²", f"{g['r2']:.3f}")
    if "rmse_eur" in g:
        d2.metric("RMSE", f"€{g['rmse_eur']/1e6:.1f}M")
    if "cv_r2_mean" in g:
        d3.metric("CV R² (5-fold)", f"{g['cv_r2_mean']:.3f} ± {g.get('cv_r2_std', 0):.3f}")
    d4.metric("Players", f"{len(df)}")

    st.divider()

    # Methodology
    st.markdown("#### Methodology")
    st.markdown(f"""
**Data:** FBref (per-90 performance stats) + Transfermarkt via Kaggle (market values).
{len(df)} outfield players with ≥900 minutes across the Big 5 European Leagues.

**Model:** XGBoost gradient-boosted trees. Features include goals, assists, xG, xAG,
progressive passes & carries, tackles, interceptions, age, contract length, league prestige, and club tier.

**Reality Grounding:** Predictions are blended {int(model_weight*100)}% model /
{int((1-model_weight)*100)}% market value. The market captures factors pure stats miss —
brand value, shirt sales, social media following, agent leverage, and scarcity premiums.

**Gap Cap:** ±60% maximum. No player can be more than 60% under/overvalued.

**Position Models:** Separate XGBoost models for forwards, midfielders, and defenders,
since value drivers differ by position (goals for strikers vs tackles for center-backs).
    """)

    # Export full dataset
    st.divider()
    st.markdown("#### 📥 Export Data")
    full_csv = filtered[["player", "squad", "position_group", "league_label",
                          "age", "actual_m", "predicted_m", "gap_m",
                          "value_gap_pct"]].to_csv(index=False)
    st.download_button("Download full dataset (CSV)", full_csv,
                       "transfer_valuation_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# TAB 6: AI SCOUT
# ═══════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 AI Scouting Assistant")
    st.caption(
        "Ask a natural-language question about your transfer targets. "
        "Powered by Claude — searches your player database in real time."
    )

    with st.expander("💡 Example queries"):
        st.markdown("""
- *Find me a young left winger under €20M who plays like a young Neymar*
- *Who are the most undervalued defensive midfielders in the Bundesliga?*
- *I need a striker under 23 with high xG and good pressing stats, budget €30M*
- *Compare the top 3 undervalued center-backs in Serie A*
- *Which Premier League players have the highest goals per 90 but are still undervalued?*
        """)

    scout_query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. Find undervalued U23 forwards in La Liga under €25M...",
    )

    if scout_query:
        with st.spinner("🔍 Scouting..."):
            scout_cols = ["player", "squad", "pos", "position_group",
                          "league_label", "age", "actual_m", "predicted_m",
                          "gap_m", "value_gap_pct"]
            for c in ["gls", "ast", "xg", "xag", "sh/90", "sot/90",
                       "npxg", "ppa", "prgdist", "prgc", "prgp",
                       "tkl", "tkld", "succ", "touches",
                       "contract_years_remaining", "tm_international_caps"]:
                if c in filtered.columns:
                    scout_cols.append(c)
            scout_cols = [c for c in scout_cols if c in filtered.columns]
            data_summary = filtered[scout_cols].head(200).to_csv(index=False)

            system_prompt = f"""You are an elite football scout assistant analyzing player data
from the Big 5 European Leagues ({SEASON} season).

You have a dataset of {len(filtered)} players with actual market values,
model-predicted values, and the gap (positive = undervalued).

Key columns: actual_m (market value €M), predicted_m (model value €M),
gap_m (value gap €M), gls/ast (goals/assists), xg/xag (expected goals/assists),
sh/90 sot/90 (shots per 90), prgp/prgc (progressive passes/carries), tkl (tackles).

Give specific player recommendations with numbers. Be concise and actionable.
Format player names in bold. Include values and predicted values."""

            user_msg = f"""Query: {scout_query}

Player data (top 200 by value gap):
{data_summary}

Answer with specific player recommendations."""

            try:
                response = http_requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1500,
                        "messages": [{"role": "user", "content": user_msg}],
                        "system": system_prompt,
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()
                    answer = "".join(
                        b["text"] for b in result.get("content", [])
                        if b.get("type") == "text"
                    )
                    st.markdown(answer)
                else:
                    st.warning("AI Scout unavailable. Using keyword search instead.")
                    _fallback_search(scout_query, filtered)
            except Exception:
                st.warning("AI Scout unavailable. Using keyword search instead.")
                _fallback_search(scout_query, filtered)


def _fallback_search(query: str, data: pd.DataFrame):
    """Keyword-based fallback when AI is unavailable."""
    q = query.lower()
    result = data.copy()
    if any(w in q for w in ["forward", "striker", "winger", "fw"]):
        result = result[result["position_group"] == "FW"]
    elif any(w in q for w in ["midfielder", "mid", "mf"]):
        result = result[result["position_group"] == "MF"]
    elif any(w in q for w in ["defender", "back", "cb", "df"]):
        result = result[result["position_group"] == "DF"]
    for league in ["premier league", "la liga", "bundesliga", "serie a", "ligue 1"]:
        if league in q:
            result = result[result["league_label"].str.lower() == league]
    age_match = re.search(r'under (\d+)', q)
    if age_match:
        result = result[result["age"] <= int(age_match.group(1))]
    budget_match = re.search(r'€(\d+)m|(\d+)m budget|under (\d+)\s*m', q)
    if budget_match:
        budget = int(next(g for g in budget_match.groups() if g))
        result = result[result["actual_m"] <= budget]
    if "overvalued" not in q:
        result = result[result["gap_m"] > 0]
    result = result.sort_values("gap_m", ascending=False).head(10)
    if result.empty:
        st.warning("No players match your criteria.")
    else:
        show = result[["player", "squad", "position_group", "age",
                        "actual_m", "predicted_m", "gap_m"]].copy()
        show.columns = ["Player", "Club", "Pos", "Age",
                        "Price (€M)", "Model Value (€M)", "Gap (€M)"]
        show.index = range(1, len(show) + 1)
        st.dataframe(show, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 7: SIMILAR PLAYERS
# ═══════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🔗 Find Similar Players")
    st.caption(
        "Select a player to find statistically similar players across the Big 5 leagues. "
        "Uses cosine similarity across performance metrics."
    )

    SIM_FEATURES = [c for c in ["gls", "ast", "xg", "xag", "npxg", "sh/90",
                                 "sot/90", "g/sh", "ppa", "prgp", "prgc",
                                 "prgdist", "tkl", "tkld", "tkld%", "succ",
                                 "succ%", "touches", "rec", "totdist",
                                 "age", "minutes"]
                    if c in filtered.columns]

    sim_options = sorted(filtered["player"].dropna().unique().tolist())
    sim_player = st.selectbox("Select a player", ["— select —"] + sim_options,
                              key="sim_sel")

    cs1, cs2 = st.columns(2)
    with cs1:
        n_similar = st.slider("Number of similar players", 5, 20, 10)
    with cs2:
        sim_pos_filter = st.checkbox("Same position only", value=True)
        sim_cheaper = st.checkbox("Cheaper alternatives only", value=False)

    if sim_player != "— select —" and len(SIM_FEATURES) >= 3:
        target_row = filtered[filtered["player"] == sim_player]
        if target_row.empty:
            st.warning("Player not found.")
        else:
            target = target_row.iloc[0]

            col_tp, col_ti = st.columns([1, 3])
            with col_tp:
                img = target.get("tm_image_url", "")
                if pd.notna(img) and str(img).startswith("http"):
                    st.image(str(img), width=130)
            with col_ti:
                st.markdown(f"### {sim_player}")
                st.caption(
                    f"**{POSITION_LABELS.get(target.get('position_group',''),'')}** · "
                    f"{target.get('squad','')} · {target.get('league_label','')} · "
                    f"Age {int(target.get('age',0))}"
                )
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Value", f"€{target['actual_m']:.1f}M")
                sm2.metric("Predicted", f"€{target['predicted_m']:.1f}M")
                g = target.get("gap_m", 0)
                sm3.metric("Gap", f"{'+'if g>=0 else ''}€{g:.1f}M")

            st.divider()

            sim_data = filtered.copy()
            if sim_pos_filter:
                sim_data = sim_data[sim_data["position_group"] == target["position_group"]]
            if sim_cheaper:
                sim_data = sim_data[sim_data["actual_m"] < target["actual_m"]]
            sim_data = sim_data[sim_data["player"] != sim_player]

            if len(sim_data) < 3:
                st.warning("Not enough players. Try unchecking filters.")
            else:
                all_p = pd.concat([target_row, sim_data])
                X = all_p[SIM_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                sims = cosine_similarity(X_s[0:1], X_s[1:])[0]
                sim_data = sim_data.copy()
                sim_data["similarity"] = sims
                top_sim = sim_data.sort_values("similarity", ascending=False).head(n_similar)

                st.markdown(f"#### Top {n_similar} Most Similar Players")
                for rank, (_, row) in enumerate(top_sim.iterrows(), 1):
                    c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2.5, 1, 0.8, 0.8, 0.8])
                    with c1:
                        img = row.get("tm_image_url", "")
                        if pd.notna(img) and str(img).startswith("http"):
                            st.image(str(img), width=40)
                    with c2:
                        st.markdown(f"**{rank}. {row['player']}**")
                        st.caption(f"{row.get('squad','')} · {row.get('league_label','')}")
                    with c3:
                        st.metric("Match", f"{row['similarity']*100:.0f}%")
                    with c4:
                        st.metric("Value", f"€{row['actual_m']:.0f}M")
                    with c5:
                        st.metric("Age", f"{int(row.get('age',0))}")
                    with c6:
                        g = row.get("gap_m", 0)
                        st.metric("Gap", f"{'+'if g>=0 else ''}€{g:.1f}M")

                # Radar: target vs top 3
                st.divider()
                st.markdown("#### Radar: Target vs Top 3 Matches")
                radar_p = pd.concat([target_row, top_sim.head(3)])
                rpos = target.get("position_group", "MF")
                rcfg = RADAR_STATS.get(rpos, RADAR_STATS["MF"])
                avail_r = {k: v for k, v in rcfg.items() if k in radar_p.columns}

                if len(avail_r) >= 3:
                    fig_sim = go.Figure()
                    clrs = ["#f59e0b", "#3b82f6", "#ef4444", "#22c55e"]
                    for i, (_, row) in enumerate(radar_p.iterrows()):
                        vals = []
                        for sc in avail_r:
                            v = pd.to_numeric(row.get(sc, 0), errors="coerce")
                            vals.append(v if pd.notna(v) else 0)
                        for j, sc in enumerate(avail_r):
                            mx = pd.to_numeric(filtered[sc], errors="coerce").quantile(0.95)
                            vals[j] = min(vals[j]/mx*100, 100) if mx > 0 else 0
                        name = f"⭐ {row['player']}" if i == 0 else row["player"]
                        fig_sim.add_trace(go.Scatterpolar(
                            r=vals+[vals[0]],
                            theta=list(avail_r.values())+[list(avail_r.values())[0]],
                            fill="toself", name=name,
                            line=dict(color=clrs[i%4], width=3 if i==0 else 1.5),
                            opacity=0.7 if i==0 else 0.4))
                    fig_sim.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                                    showticklabels=False), bgcolor=BG),
                        height=450, paper_bgcolor=BG, font=dict(color="#fff"),
                        margin=dict(l=60, r=60, t=30, b=30))
                    st.plotly_chart(fig_sim, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data: FBref ({SEASON}) · Transfermarkt (via Kaggle)  |  "
    f"Model: XGBoost · Blended predictions  |  "
    f"Portfolio project — not for commercial use"
)
