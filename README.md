# ⚽ Soccer Transfer Market Valuation Model

A full-stack data science project that predicts player market values from on-pitch performance stats and compares them to actual Transfermarkt valuations — surfacing the most **undervalued** and **overvalued** players across the Big 5 European Leagues.

### 🔗 [Live App](https://soccer-transfer-valuation.streamlit.app/) · [Dataset: FBref](https://fbref.com/) · [Dataset: Transfermarkt](https://www.transfermarkt.com/)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat&logo=plotly&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude_API-AI_Scout-D97757?style=flat)

---

## The Idea

Football clubs pay transfer fees based on a mix of performance, age, contract leverage, marketability, and brand power. This project isolates the **performance component**: given a player's on-pitch stats, what market value would a purely data-driven model assign?

The gap between predicted and actual value reveals players who are potentially **undervalued** (stats say they're worth more) or **overvalued** (brand/reputation inflates their price beyond statistical output).

### What makes this different from a typical ML project:

- **Reality-grounded predictions** — the model blends 65% statistical prediction with 35% actual market value, because the market captures things stats can't (brand, shirt sales, scarcity)
- **Capped value gaps** — no player can be more than ±60% under/overvalued, reflecting real-world transfer dynamics
- **Elite club premium** — players at Real Madrid, Barcelona, Man City, PSG, and Bayern are worth more due to revenue, CL draw, and negotiating power
- **Position-specific models** — separate XGBoost models for forwards, midfielders, and defenders, since value drivers differ by position

---

## Features (7 interactive tabs)

### 📊 Overview
Interactive scatter plot (predicted vs actual market value), color-coded by league. Hover any dot to see the player's stats. Includes top 15 undervalued and overvalued leaderboards.

### 🔍 Transfer Recommendation Engine
Pick a position, set a max budget and age, and get a ranked list of the most undervalued players that fit. Includes CSV export for offline analysis.

### ⚔️ Player Comparison
Select 2-3 players for side-by-side comparison with photos, valuations, and a radar chart of per-90 stats normalized to league percentiles.

### 🏟️ League Analysis
Bar chart showing average under/overvaluation by league. Which leagues have the most bargains? Includes position-level breakdown.

### 🧠 Explainability
Position-specific model performance (R², RMSE per position), value gap distribution histogram, full methodology writeup, and dataset export.

### 🤖 AI Scouting Assistant
Natural language search powered by Claude's API. Ask questions like *"Find me an undervalued U23 striker in La Liga under €25M"* and get specific player recommendations with stats. Falls back to keyword-based search when offline.

### 🔗 Similar Players
Select any player to find statistically similar players using cosine similarity across 20+ performance metrics. Toggle "same position only" and "cheaper alternatives only." Includes radar chart comparing the target vs their top 3 statistical matches.

---

## How It Works

### 1. Data Collection

| Source | What | Method |
|--------|------|--------|
| **FBref** | 2025-26 stats: goals, assists, penalties, starts, minutes, shots, shots on target | Manual CSV export (standard + shooting tables; the rest are blank since FBref lost its advanced data) |
| **Transfermarkt** (live) | Current club, market value, contract expiry, position, photo for every Big 5 squad player | Daily scrape of all 96 Big 5 squad pages (`scrape_tm_squads.py`) |
| **Transfermarkt** (Kaggle archive) | Valuation history to June 2026, caps, birthplace, peak value | `player_valuations.csv` + `players.csv` ([dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores), no longer updated) |

**Coverage:** outfield players with ≥900 minutes in the Big 5 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) in 2025-26 who are still at a Big 5 club, valued at today's Transfermarkt price and shown at their current club.

**Why 2025-26 stats:** FBref lost its advanced stats (xG, progressive passes, shot-creating actions) in January 2026, so there is no 2026-27 equivalent. The model predicts each player's current value from his last full season. Players who joined the Big 5 from other leagues this summer have no 2025-26 Big 5 stats and aren't included.

### 2. Preprocessing
- Fuzzy name matching across FBref and Transfermarkt via `rapidfuzz`, restricted to the same birth year (±1) so a player who left the Big 5 can't be matched to a namesake
- Per-90 stat computation for counting stats
- Feature engineering: age potential multiplier, league prestige, elite club premium, position encoding
- Log-transform of market values (right-skewed distribution)

### 3. Modeling
- **XGBoost** gradient-boosted trees (primary model) on per-90 goals, assists, non-penalty goals, shots, shots on target, shot accuracy, conversion, starts and minutes, plus age, contract, league and club
- **Monotonic constraints** so more output never lowers a predicted value
- **Out-of-fold predictions** (5-fold): each player is priced by a model that never saw his price
- **Position-specific models** for FW, MF, DF
- **SHAP** for feature importance and explainability
- **Quantile regression** for 80% confidence intervals
- **Blended predictions:** 65% model / 35% market value
- **Gap cap:** ±60% maximum under/overvaluation

FBref lost its advanced data (xG, passing, possession, defensive actions) in January 2026, so the model sees attacking output only. Defenders and deep midfielders are judged on less, and tend to look overvalued.

### 4. Post-Prediction Adjustments

| Adjustment | Purpose |
|-----------|---------|
| Age potential (1.15x for U21 → 0.85x for 31+) | Young players command premium for resale |
| Elite output (top 2/5/10% of FW & MF by non-pen G+A per 90: 1.15x/1.10x/1.05x) | Tree models can't extrapolate above the priciest players they've seen; elite producers 30+ are priced at prime age and their age discount is capped at 0.95x |
| League prestige (1.15x PL → 1.0x Ligue 1) | Premier League revenue inflates fees |
| Club brand (Bayern 1.25x; Real Madrid, Barcelona 1.20x; PSG, Man City 1.15x; other elite clubs 1.10x) | Brand power, CL revenue, negotiation leverage — deliberately opinionated |
| Star power (50+ caps 1.04x, 100+ caps 1.08x) | Global profile and marketability |
| Market blend (35% actual value) | Acknowledges the market knows things stats can't capture |
| Gap cap (±60%) | Prevents unrealistic extreme predictions |

-----------|---------|
| Age potential (1.15x for U20 → 0.85x for 31+) | Young players command premium for resale |
| League prestige (1.15x PL → 1.0x Ligue 1) | Premier League revenue inflates fees |
| Elite club premium (1.15x for top 6 clubs) | Brand power, CL revenue, negotiation leverage |
| Market blend (35% actual value) | Acknowledges the market knows things stats can't capture |
| Gap cap (±60%) | Prevents unrealistic extreme predictions |

---

## Tech Stack

- **Language:** Python 3.12
- **ML:** XGBoost, scikit-learn, SHAP
- **Dashboard:** Streamlit, Plotly
- **Data:** pandas, rapidfuzz (name matching)
- **AI:** Claude API (Anthropic) for natural language scouting
- **Similarity:** scikit-learn (cosine similarity, StandardScaler)
- **Deployment:** Streamlit Community Cloud (auto-deploys from GitHub)
- **CI/CD:** GitHub Actions (imports Transfermarkt exports, retrains, commits new data)

---

## Project Structure

```
soccer-transfer-valuation/
├── .github/workflows/
│   └── daily_refresh.yml       # Data refresh pipeline (runs on a new export)
├── data/
│   ├── raw/                    # FBref CSVs, live Transfermarkt squads, Kaggle archive
│   └── processed/              # Merged datasets + model output
├── src/
│   ├── scrape_fbref.py         # Loads & merges 6 FBref stat tables
│   ├── scrape_tm_squads.py     # Scrapes current Big 5 squads + values from Transfermarkt
│   ├── scrape_transfermarkt.py # Builds one row per player (live squads + Kaggle metadata)
│   ├── preprocess.py           # Fuzzy matching + feature engineering
│   ├── model.py                # XGBoost training + SHAP + predictions
│   └── app.py                  # Streamlit dashboard (7 tabs, 950+ lines)
├── tools/
│   └── export_squads.js        # Browser export of current Big 5 squads
├── requirements.txt
└── README.md
```

---

## Refreshing the Data

Transfermarkt blocks GitHub's servers (it serves a bot check instead of data), so new squads and values come from a normal browser:

1. Open [transfermarkt.com](https://www.transfermarkt.com) in Chrome, open DevTools (Cmd+Option+J), paste [`tools/export_squads.js`](tools/export_squads.js) into the Console and press Enter. After ~4 minutes it copies all 96 Big 5 squads to your clipboard.
2. On GitHub, add a file at `data/raw/tm_browser_export.tsv` on `main`, paste, and commit.
3. The **Data Refresh** workflow imports it, retrains the model, commits the new data and deletes the export. Streamlit redeploys automatically, and the app shows the date the values are from.

---

## Run Locally

```bash
# Clone
git clone https://github.com/Aarivg/soccer-transfer-valuation.git
cd soccer-transfer-valuation

# Set up environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
python src/scrape_fbref.py --season 2526    # loads local FBref CSVs
python src/scrape_tm_squads.py               # current squads + values (~100 requests, a few minutes)
python src/scrape_transfermarkt.py           # one row per player
python src/preprocess.py --season 2526       # merge + feature engineering
python src/model.py --season 2526            # train + predict

# Launch dashboard
streamlit run src/app.py
```

---

## Key Findings (2025-26 Season)

**Most undervalued:** Players whose stats predict significantly higher value than their current Transfermarkt price — typically young breakout players at mid-table clubs.

**Most overvalued:** Superstars where brand value, marketing power, and scarcity premium inflate their fee beyond what on-pitch output alone justifies.

**Age is the strongest predictor** — the market heavily prices in future potential and resale value.

**The model explains ~72% of the variance in (log) market value out of sample** (5-fold CV R² ≈ 0.72). The rest is brand, hype, agent power, injury history, defensive and creative quality FBref no longer publishes, and other factors stats don't measure — which is exactly why we blend predictions with market values.

---

## Limitations

- No xG from open play vs set pieces (FBref granularity)
- No injury history or availability data
- No brand/social media metrics
- Performance stats are from 2025-26 (FBref no longer publishes advanced stats)
- Transfermarkt valuations update every 3-6 months per league
- Values refresh when a new Transfermarkt export is committed (GitHub's servers are blocked from scraping it directly)
- ~3% of players unmatched due to name differences between FBref and Transfermarkt

---

## Future Improvements

- [ ] Goalkeeper-specific model (separate feature set)
- [ ] Automated FBref data pipeline (pending Cloudflare solution)
- [ ] Historical season comparison (track prediction accuracy over time)
- [ ] Neural network baseline (target R² 0.60+)
- [ ] Squad builder optimization (best XI under budget constraint)
- [ ] Injury-adjusted valuations

---

## Disclaimer

This project is for educational and portfolio purposes only. Data is used for non-commercial analysis in accordance with each source's reasonable-use policies.

---

Built by [Aariv Gupta](https://github.com/Aarivg)
