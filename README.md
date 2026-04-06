# Soccer Transfer Market Valuation Model

A data science portfolio project that compares players' actual Transfermarkt
market values against values predicted from on-pitch performance metrics sourced
from FBref — across all **Big 5 European leagues** (Premier League, La Liga,
Bundesliga, Serie A, Ligue 1).

https://soccer-transfer-valuation.streamlit.app/

---

## Project Overview

Football clubs pay transfer fees based on a mix of performance, age, contract
length, marketability, and negotiation leverage. This project attempts to
isolate the **performance component**: given a player's on-pitch output, what
market value would a purely data-driven model assign?

The delta between predicted and actual value surfaces players who are
potentially **undervalued** (predicted > actual) or **overvalued**
(actual > predicted) relative to their statistical output.

---

## Methodology

### 1. Data Collection

| Source | Tool | Data |
|--------|------|------|
| FBref | `soccerdata` Python library | Per-90 performance metrics |
| Transfermarkt | Custom scraper (`requests` + `BeautifulSoup`) | Market values in EUR |

**FBref stats collected (per 90 minutes):**
- Goals, Assists
- Expected Goals (xG), Expected Assisted Goals (xAG)
- Progressive Passes, Progressive Carries
- Pressures (defensive pressure attempts)
- Minutes Played (filter: ≥ 900 minutes)

### 2. Preprocessing ([preprocess.py](src/preprocess.py))
- Fuzzy-match player names across FBref and Transfermarkt
- Log-transform market values (right-skewed distribution)
- Encode position, league, and age as features
- Train/test split stratified by league

### 3. Modelling ([model.py](src/model.py))
- Baseline: median market value per position
- Linear Regression with L2 regularisation
- **XGBoost** gradient-boosted trees (primary model)
- Evaluation: RMSE and R² on held-out test set

### 4. Application ([app.py](src/app.py))
- Interactive **Streamlit** dashboard
- Scatter plot: predicted vs. actual market value (Plotly)
- Player search with over/undervaluation badge
- League and position filters

---

## Folder Structure

```
soccer-transfer-valuation/
├── data/
│   ├── raw/          # Unmodified scraped files
│   └── processed/    # Cleaned, merged, feature-engineered CSVs
├── notebooks/        # Exploratory analysis
├── src/
│   ├── scrape_fbref.py          # FBref stats via soccerdata
│   ├── scrape_transfermarkt.py  # Transfermarkt market values
│   ├── preprocess.py            # Cleaning, merging, feature engineering
│   ├── model.py                 # Training and evaluation
│   └── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Scrape data

```bash
# FBref stats (requires network access; soccerdata caches locally)
python src/scrape_fbref.py

# Transfermarkt market values
python src/scrape_transfermarkt.py
```

> **Note:** FBref data is fetched via the `soccerdata` library which caches
> responses in `~/.soccerdata/`. The first run may take a few minutes per
> league. Transfermarkt scraping uses polite rate-limiting; see the scraper
> for configurable delay settings.

### Run the dashboard

```bash
streamlit run src/app.py
```

---

## Scraping Limitations & Workarounds

| Library / Site | Limitation | Workaround |
|----------------|-----------|------------|
| `soccerdata` (FBref) | Scrapes FBref HTML; FBref occasionally rate-limits or changes table structure | Library caches pages locally; pin `soccerdata==1.6.0` for table-structure stability |
| `soccerdata` (FBref) | Season strings must match FBref's naming exactly (`"2324"`, `"2425"`) | Check `sd.FBref.available_seasons()` if a season is missing |
| Transfermarkt | No public API; site has anti-bot headers | Use `requests` with browser `User-Agent` + `Referer` headers; add `time.sleep` between requests |
| Transfermarkt | Player name encoding varies (accents, dots) | Fuzzy name matching (`difflib` or `rapidfuzz`) during merge |

---

## Roadmap

- [ ] Transfermarkt scraper and player-name matching
- [ ] Preprocessing pipeline and EDA notebook
- [ ] XGBoost model with SHAP feature importance
- [ ] Streamlit dashboard with Plotly scatter plot
- [ ] Age-curve adjustment for under/over-24 players
- [ ] Goalkeeper model (separate feature set)

---

## Disclaimer

This project is for educational and portfolio purposes only. Data is scraped
for non-commercial use in accordance with each site's reasonable-use policies.
