"""
scrape_tm_squads.py — Pull current Big 5 squads, market values and contracts
straight from Transfermarkt.

Why this exists: the Kaggle "player-scores" dataset this project started on
stopped updating in July 2026 (valuations end 2026-06-12, no 2026-27 squads),
so it no longer reflects transfers or new valuations. This script reads each
Big 5 club's "detailed squad" page for the current season instead:

    league page   → the clubs in that league this season
    squad page    → every player: club, position, date of birth, height,
                    foot, joined date, contract expiry, market value, photo

Outputs (both in data/raw/):
    tm_squads_current.csv       one row per player currently in a Big 5 squad
    tm_valuations_scraped.csv   valuation history, appended only when a
                                player's value or club changes

It is deliberately strict: if Transfermarkt blocks the request (it serves an
anti-bot page with a 200 status and no squad table) or any league/club comes
back incomplete, the script exits non-zero and writes nothing, so a bad
scrape can never silently replace good data.

Usage:
    python src/scrape_tm_squads.py
    python src/scrape_tm_squads.py --season-id 2026 --delay 3
"""

import argparse
import os
import random
import re
import sys
import time
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com"

# Transfermarkt competition code → league slug (the slug is cosmetic; the
# code is what Transfermarkt routes on)
LEAGUES = {
    "GB1": "premier-league",
    "ES1": "laliga",
    "L1": "bundesliga",
    "IT1": "serie-a",
    "FR1": "ligue-1",
}
# Each Big 5 league has 18 or 20 clubs
MIN_CLUBS = {"GB1": 20, "ES1": 20, "L1": 18, "IT1": 20, "FR1": 18}
MIN_PLAYERS_PER_CLUB = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Transfermarkt's detailed position → the broad position used in players.csv
POSITION_GROUP = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender", "Left-Back": "Defender",
    "Right-Back": "Defender", "Sweeper": "Defender", "Defender": "Defender",
    "Defensive Midfield": "Midfield", "Central Midfield": "Midfield",
    "Attacking Midfield": "Midfield", "Left Midfield": "Midfield",
    "Right Midfield": "Midfield", "Midfield": "Midfield",
    "Left Winger": "Attack", "Right Winger": "Attack",
    "Centre-Forward": "Attack", "Second Striker": "Attack",
    "Attack": "Attack",
}


class BlockedError(RuntimeError):
    """Transfermarkt answered, but without the data (anti-bot page)."""


def current_season_id(today: date | None = None) -> int:
    """Transfermarkt labels a season by its starting year (2026 = 2026-27)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


# ── Value / field parsing ────────────────────────────────────────────
def parse_market_value(text: str) -> float | None:
    """'€28M' / '€4.0M' / '€500K' / '€1.20bn' / '-' → euros."""
    if not text:
        return None
    t = text.strip().lower().replace(",", ".").replace("€", "").replace(" ", "")
    m = re.match(r"^([\d.]+)(bn|m|k|th\.?|mio\.?)?$", t)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").rstrip(".")
    mult = {"bn": 1e9, "m": 1e6, "mio": 1e6, "k": 1e3, "th": 1e3, "": 1}[unit]
    return round(num * mult)


DATE_FORMATS = ["%d/%m/%Y", "%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y"]


def parse_date(text: str) -> str | None:
    """Return an ISO date from any of Transfermarkt's locale formats."""
    if not text:
        return None
    t = re.sub(r"\(\d+\)", "", text).strip()  # drop the "(26)" age suffix
    if not t or t in {"-", "?"}:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_height(text: str) -> float | None:
    """'1,89m' / '1.89 m' → 189."""
    m = re.search(r"(\d)[.,](\d{2})", text or "")
    return float(m.group(1) + m.group(2)) if m else None


def img_src(img) -> str:
    if img is None:
        return ""
    src = img.get("data-src") or img.get("src") or ""
    if src.startswith("data:"):
        src = img.get("data-src") or ""
    # Use the larger "header" portrait, matching the old Kaggle image_url
    return src.split("?")[0].replace("/portrait/medium/", "/portrait/header/") \
                            .replace("/portrait/small/", "/portrait/header/")


# ── Page parsing ─────────────────────────────────────────────────────
def parse_league_page(html: str) -> list[dict]:
    """Clubs listed on a competition's overview page."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        raise BlockedError("league page has no clubs table")
    clubs = {}
    for a in table.select('td.hauptlink a[href*="/startseite/verein/"]'):
        m = re.search(r"/verein/(\d+)", a.get("href", ""))
        if not m:
            continue
        cid = int(m.group(1))
        name = (a.get("title") or a.get_text(strip=True)).strip()
        if cid not in clubs and name:
            clubs[cid] = {"club_id": cid, "club_name": name}
    return list(clubs.values())


def parse_squad_page(html: str) -> list[dict]:
    """Players on a club's detailed squad page (/kader/.../plus/1)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        raise BlockedError("squad page has no players table")
    body = table.find("tbody") or table
    players = []
    for tr in body.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 10:
            continue
        cell = tds[1]
        link = cell.select_one("td.hauptlink a[href*='/profil/spieler/']") \
            or cell.select_one("a[href*='/profil/spieler/']")
        if link is None:
            continue
        pid = int(re.search(r"/spieler/(\d+)", link["href"]).group(1))
        inner_rows = cell.find_all("tr")
        sub_position = inner_rows[1].get_text(" ", strip=True) \
            if len(inner_rows) > 1 else ""
        nats = [i.get("title") or i.get("alt") or ""
                for i in tds[3].find_all("img")]
        signed_from = tds[7].find("img")
        players.append({
            "player_id": pid,
            "player_name": link.get_text(" ", strip=True),
            "sub_position": sub_position,
            "position": POSITION_GROUP.get(sub_position, "Missing"),
            "date_of_birth": parse_date(tds[2].get_text(" ", strip=True)),
            "country_of_citizenship": nats[0] if nats else "",
            "second_citizenship": nats[1] if len(nats) > 1 else "",
            "height_in_cm": parse_height(tds[4].get_text(" ", strip=True)),
            "foot": tds[5].get_text(" ", strip=True) or None,
            "joined_date": parse_date(tds[6].get_text(" ", strip=True)),
            "signed_from": (signed_from.get("title") or "") if signed_from else "",
            "contract_expiration_date": parse_date(tds[8].get_text(" ", strip=True)),
            "market_value_in_eur": parse_market_value(tds[9].get_text(" ", strip=True)),
            "image_url": img_src(cell.find("img")),
            "url": f"{BASE_URL}{link['href'].split('?')[0]}",
        })
    return players


# ── Fetching ─────────────────────────────────────────────────────────
class Fetcher:
    def __init__(self, delay: float = 3.0, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.retries = retries
        self.requests_made = 0

    def get(self, path: str) -> str:
        url = BASE_URL + path
        last_err = None
        for attempt in range(1, self.retries + 1):
            if self.requests_made:
                time.sleep(self.delay + random.uniform(0, self.delay))
            self.requests_made += 1
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as e:
                last_err = e
            else:
                if r.status_code == 200 and "table" in r.text and "items" in r.text:
                    return r.text
                last_err = RuntimeError(f"HTTP {r.status_code}, {len(r.text)} bytes")
                if r.status_code in (403, 429):
                    time.sleep(20 * attempt)
            print(f"    retry {attempt}/{self.retries} for {path}: {last_err}")
        raise BlockedError(f"{path}: {last_err}")


def scrape(season_id: int, delay: float) -> pd.DataFrame:
    f = Fetcher(delay=delay)
    rows = []
    for code, slug in LEAGUES.items():
        clubs = parse_league_page(f.get(f"/{slug}/startseite/wettbewerb/{code}/saison_id/{season_id}"))
        print(f"🏟️  {code}: {len(clubs)} clubs")
        if len(clubs) < MIN_CLUBS[code]:
            raise BlockedError(f"{code}: only {len(clubs)} clubs (expected {MIN_CLUBS[code]})")
        for club in clubs:
            html = f.get(f"/x/kader/verein/{club['club_id']}/saison_id/{season_id}/plus/1")
            players = parse_squad_page(html)
            valued = sum(p["market_value_in_eur"] is not None for p in players)
            print(f"   {club['club_name']:<32} {len(players):>3} players, {valued:>3} valued")
            if len(players) < MIN_PLAYERS_PER_CLUB or valued < MIN_PLAYERS_PER_CLUB // 2:
                raise BlockedError(
                    f"{club['club_name']}: {len(players)} players / {valued} valued — "
                    "page looks incomplete (likely an anti-bot response)")
            for p in players:
                p.update(current_club_id=club["club_id"],
                         current_club_name=club["club_name"],
                         current_club_domestic_competition_id=code)
            rows.extend(players)
    df = pd.DataFrame(rows)
    # A player can't be in two squads; keep the first (shouldn't happen)
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    df["scraped_at"] = date.today().isoformat()
    df["season_id"] = season_id
    print(f"\n✅ {len(df)} players across {df.current_club_id.nunique()} clubs "
          f"({f.requests_made} requests)")
    return df


# ── Valuation history ────────────────────────────────────────────────
HISTORY_COLS = ["player_id", "date", "market_value_in_eur", "current_club_name",
                "current_club_id", "player_club_domestic_competition_id"]


def update_history(squads: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """Append a valuation row for each player whose value or club changed
    since their last recorded valuation (Kaggle history + scraped history)."""
    hist_path = os.path.join(raw_dir, "tm_valuations_scraped.csv")
    scraped_hist = pd.read_csv(hist_path) if os.path.exists(hist_path) \
        else pd.DataFrame(columns=HISTORY_COLS)

    prior = [scraped_hist]
    kaggle_path = os.path.join(raw_dir, "player_valuations.csv")
    if os.path.exists(kaggle_path):
        prior.append(pd.read_csv(kaggle_path, usecols=HISTORY_COLS))
    prior = pd.concat(prior, ignore_index=True)
    prior["date"] = pd.to_datetime(prior["date"], errors="coerce")
    last = (prior.sort_values("date").drop_duplicates("player_id", keep="last")
            .set_index("player_id"))

    new = squads.dropna(subset=["market_value_in_eur"]).copy()
    new = new.assign(date=new["scraped_at"],
                     player_club_domestic_competition_id=new["current_club_domestic_competition_id"])
    prev_val = new["player_id"].map(last["market_value_in_eur"])
    prev_club = new["player_id"].map(last["current_club_id"])
    changed = (prev_val != new["market_value_in_eur"]) | (prev_club != new["current_club_id"])
    added = new.loc[changed, HISTORY_COLS]

    out = pd.concat([scraped_hist, added], ignore_index=True)
    out = out.drop_duplicates(subset=["player_id", "date"], keep="last")
    out.to_csv(hist_path, index=False)
    print(f"📈 Valuation history: +{len(added)} changes "
          f"({int(prev_val.isna().sum())} players with no earlier valuation)")
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--season-id", type=int, default=None,
                    help="Transfermarkt season id (2026 = 2026-27). Default: current season")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="Base seconds between requests (a random 0–delay is added)")
    ap.add_argument("--raw-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    args = ap.parse_args()

    season_id = args.season_id or current_season_id()
    print(f"🔎 Scraping Transfermarkt Big 5 squads, season {season_id}/{str(season_id + 1)[-2:]}")
    try:
        squads = scrape(season_id, args.delay)
    except BlockedError as e:
        print(f"\n❌ Scrape aborted, nothing written: {e}")
        print("   Transfermarkt is most likely blocking this client. Existing data is untouched.")
        sys.exit(1)

    os.makedirs(args.raw_dir, exist_ok=True)
    out_path = os.path.join(args.raw_dir, "tm_squads_current.csv")
    squads.to_csv(out_path, index=False)
    print(f"💾 Saved {out_path}")
    update_history(squads, args.raw_dir)


if __name__ == "__main__":
    main()
