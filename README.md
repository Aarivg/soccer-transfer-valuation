# ⚽ Soccer Transfer Market Valuation Model

A full-stack data science project that predicts player market values from on-pitch performance stats and compares them to actual Transfermarkt valuations — surfacing the most **undervalued** and **overvalued** players across the Big 5 European Leagues.

### 🔗 [Live App](https://soccer-transfer-valuation.streamlit.app/) · [Dataset: FBref](https://fbref.com/) · [Dataset: Transfermarkt via Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)

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
| **FBref** | Per-90 stats: goals, assists, xG, xAG, progressive passes/carries, tackles, interceptions, shots, dribbles | Manual CSV export (6 stat tables) |
| **Transfermarkt** (Kaggle) | Market values, contract expiry, player metadata, historical valuations, photos | `player_valuations.csv` + `players.csv` |

**Coverage:** 1,284 outfield players with ≥900 minutes across all Big 5 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) — 2025-26 season.

### 2. Preprocessing
- Fuzzy name matching across FBref and Transfermarkt (96.3% match rate via `rapidfuzz`)
- Per-90 stat computation for counting stats
- Feature engineering: age potential multiplier, league prestige, elite club premium, position encoding
- Log-transform of market values (right-skewed distribution)

### 3. Modeling
- **XGBoost** gradient-boosted trees (primary model)
- **Position-specific models** for FW, MF, DF
- **SHAP** for feature importance and explainability
- **Quantile regression** for 80% confidence intervals
- **Blended predictions:** 65% model / 35% market value
- **Gap cap:** ±60% maximum under/overvaluation

### 4. Post-Prediction Adjustments

| Adjustment | Purpose |
|-----------|---------|
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
- **CI/CD:** GitHub Actions (daily Transfermarkt data refresh)

---

## Project Structure

```
soccer-transfer-valuation/
├── .github/workflows/
│   └── daily_refresh.yml       # Daily auto-refresh pipeline
├── data/
│   ├── raw/                    # FBref CSVs + Kaggle Transfermarkt files
│   └── processed/              # Merged datasets + model output
├── src/
│   ├── scrape_fbref.py         # Loads & merges 6 FBref stat tables
│   ├── scrape_transfermarkt.py # Processes Kaggle Transfermarkt CSVs
│   ├── preprocess.py           # Fuzzy matching + feature engineering
│   ├── model.py                # XGBoost training + SHAP + predictions
│   └── app.py                  # Streamlit dashboard (7 tabs, 950+ lines)
├── requirements.txt
└── README.md
```

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
python src/scrape_transfermarkt.py           # processes Kaggle data
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

**The model captures ~48% of variance** (R² = 0.48). The other 52% is brand, hype, agent power, injury history, and factors stats don't measure — which is exactly why we blend predictions with market values.

---

## Limitations

- No xG from open play vs set pieces (FBref granularity)
- No injury history or availability data
- No brand/social media metrics
- Static FBref data (manual CSV refresh needed)
- Transfermarkt valuations update every 3-6 months per league
- 6.6% of players unmatched due to name encoding differences

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
