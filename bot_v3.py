#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_v3.py — WeatherBet v3: Ensemble + CLOB Maker + Thesis Monitor
===================================================================
Core trading engine upgraded with:

  1. Ensemble CDFs (probabilistic, not deterministic)
  2. CLOB maker execution (capture spread, don't cross it)
  3. Thesis-based exits (no price stops)
  4. Spatial correlation penalty (no clustered weather bets)
  5. Sensor failure blacklisting (resolution risk)
  6. MOS bias correction (when trained data available)

Usage:
    python bot_v3.py            # main loop
    python bot_v3.py status     # positions & balance
    python bot_v3.py train_mos  # train MOS models from accumulated data
"""

import re, sys, json, math, time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ensemble import EnsembleForecast, ensemble_ev, ensemble_kelly
from clob_maker import MakerEngine
from thesis_monitor import ThesisMonitor, ThesisState, create_thesis_state
from spatial import SpatialRiskManager, OpenPosition
from sensor import SensorMonitor
from mos import MOSCorrecter


# ═══════════════════════════════════════════════════════════════
# CONFIG LOAD
# ═══════════════════════════════════════════════════════════════

CONFIG_PATH = Path("config_v3.json")
if not CONFIG_PATH.exists():
    print("config_v3.json not found — copy config_v3.json from the repo")
    sys.exit(1)

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

# Shorthand access
BANKROLL     = CFG["bankroll"]
FILTERS      = CFG["filters"]
KELLY_CFG    = CFG["kelly"]
SCAN_CFG     = CFG["scan"]
THESIS_CFG   = CFG["thesis"]
ENSEMBLE_CFG = CFG["ensemble"]
CLOB_CFG     = CFG["clob"]
SPATIAL_CFG  = CFG["spatial"]
SENSOR_CFG   = CFG["sensor"]
MOS_CFG      = CFG["mos"]
PROXY_CFG    = CFG["proxy"]

EXECUTION_MODE = CFG.get("execution_mode", "paper")

# ═══════════════════════════════════════════════════════════════
# MODULE INIT
# ═══════════════════════════════════════════════════════════════

ensemble     = EnsembleForecast(cache_dir=Path("data/ensemble"))
thesis_mon   = ThesisMonitor()
spatial_mgr  = SpatialRiskManager(max_total_correlation=SPATIAL_CFG["max_total_correlation"])
sensor_mon   = SensorMonitor()
mos_correct  = MOSCorrecter(base_dir=Path(MOS_CFG.get("model_dir", "data/mos/by_station")).parent.parent 
                            if MOS_CFG.get("model_dir") else Path("data/mos"))

maker = None
if EXECUTION_MODE == "live":
    maker = MakerEngine(
        private_key=CLOB_CFG["private_key"],
        funder=CLOB_CFG["funder_address"],
        proxy_url=PROXY_CFG.get("url"),
        host=CLOB_CFG["host"],
        chain_id=CLOB_CFG["chain_id"],
        max_spread=CLOB_CFG["execution"]["max_spread"],
        unfilled_minutes_max=CLOB_CFG["execution"]["unfilled_minutes_max"],
    )

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & LOCATIONS (same as v2)
# ═══════════════════════════════════════════════════════════════

LOCATIONS = {
    "nyc":          {"lat": 40.7772,  "lon":  -73.8726, "name": "New York City", "station": "KLGA", "unit": "F", "region": "us"},
    "chicago":      {"lat": 41.9742,  "lon":  -87.9073, "name": "Chicago",       "station": "KORD", "unit": "F", "region": "us"},
    "miami":        {"lat": 25.7959,  "lon":  -80.2870, "name": "Miami",         "station": "KMIA", "unit": "F", "region": "us"},
    "dallas":       {"lat": 32.8471,  "lon":  -96.8518, "name": "Dallas",        "station": "KDAL", "unit": "F", "region": "us"},
    "seattle":      {"lat": 47.4502,  "lon": -122.3088, "name": "Seattle",       "station": "KSEA", "unit": "F", "region": "us"},
    "atlanta":      {"lat": 33.6407,  "lon":  -84.4277, "name": "Atlanta",       "station": "KATL", "unit": "F", "region": "us"},
    "london":       {"lat": 51.5048,  "lon":    0.0495, "name": "London",        "station": "EGLC", "unit": "C", "region": "eu"},
    "paris":        {"lat": 48.9962,  "lon":    2.5979, "name": "Paris",         "station": "LFPG", "unit": "C", "region": "eu"},
    "munich":       {"lat": 48.3537,  "lon":   11.7750, "name": "Munich",        "station": "EDDM", "unit": "C", "region": "eu"},
    "ankara":       {"lat": 40.1281,  "lon":   32.9951, "name": "Ankara",        "station": "LTAC", "unit": "C", "region": "eu"},
    "seoul":        {"lat": 37.4691,  "lon":  126.4505, "name": "Seoul",         "station": "RKSI", "unit": "C", "region": "asia"},
    "tokyo":        {"lat": 35.7647,  "lon":  140.3864, "name": "Tokyo",         "station": "RJTT", "unit": "C", "region": "asia"},
    "shanghai":     {"lat": 31.1443,  "lon":  121.8083, "name": "Shanghai",      "station": "ZSPD", "unit": "C", "region": "asia"},
    "singapore":    {"lat":  1.3502,  "lon":  103.9940, "name": "Singapore",     "station": "WSSS", "unit": "C", "region": "asia"},
    "lucknow":      {"lat": 26.7606,  "lon":   80.8893, "name": "Lucknow",       "station": "VILK", "unit": "C", "region": "asia"},
    "tel-aviv":     {"lat": 32.0114,  "lon":   34.8867, "name": "Tel Aviv",      "station": "LLBG", "unit": "C", "region": "asia"},
    "toronto":      {"lat": 43.6772,  "lon":  -79.6306, "name": "Toronto",       "station": "CYYZ", "unit": "C", "region": "ca"},
    "sao-paulo":    {"lat": -23.4356, "lon":  -46.4731, "name": "Sao Paulo",     "station": "SBGR", "unit": "C", "region": "sa"},
    "buenos-aires": {"lat": -34.8222, "lon":  -58.5358, "name": "Buenos Aires",  "station": "SAEZ", "unit": "C", "region": "sa"},
    "wellington":   {"lat": -41.3272, "lon":  174.8052, "name": "Wellington",    "station": "NZWN", "unit": "C", "region": "oc"},
}

MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

DATA_DIR    = Path("data")
MARKETS_DIR = DATA_DIR / "markets"
STATE_FILE  = DATA_DIR / "state.json"
DATA_DIR.mkdir(exist_ok=True)
MARKETS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# MARKET DATA (compatible with v2 format)
# ═══════════════════════════════════════════════════════════════

def market_path(city_slug, date_str):
    return MARKETS_DIR / f"{city_slug}_{date_str}.json"

def load_market(city_slug, date_str):
    p = market_path(city_slug, date_str)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def save_market(m):
    p = market_path(m["city"], m["date"])
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"balance": BANKROLL["balance"], "starting_balance": BANKROLL["balance"],
            "total_trades": 0, "wins": 0, "losses": 0, "peak_balance": BANKROLL["balance"]}

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")

def load_all_markets():
    mkts = []
    for f in MARKETS_DIR.glob("*.json"):
        try:
            mkts.append(json.loads(f.read_text(encoding="utf-8")))
        except: pass
    return mkts

# ═══════════════════════════════════════════════════════════════
# POLYMARKET HELPERS
# ═══════════════════════════════════════════════════════════════

def get_polymarket_event(city_slug, month, day, year):
    slug = f"highest-temperature-in-{city_slug}-on-{month}-{day}-{year}"
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=(5, 8))
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    except: pass
    return None

def get_market_price(market_id):
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=(3, 5))
        prices = json.loads(r.json().get("outcomePrices", "[0.5,0.5]"))
        return float(prices[0])
    except: return None

def parse_temp_range(question):
    if not question: return None
    num = r'(-?\d+(?:\.\d+)?)'
    if re.search(r'or below', question, re.IGNORECASE):
        m = re.search(num + r'[°]?[FC] or below', question, re.IGNORECASE)
        if m: return (-999.0, float(m.group(1)))
    if re.search(r'or higher', question, re.IGNORECASE):
        m = re.search(num + r'[°]?[FC] or higher', question, re.IGNORECASE)
        if m: return (float(m.group(1)), 999.0)
    m = re.search(r'between ' + num + r'-' + num + r'[°]?[FC]', question, re.IGNORECASE)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.search(r'be ' + num + r'[°]?[FC] on', question, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return (v, v)
    return None

def hours_to_resolution(end_date_str):
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return max(0.0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)
    except: return 999.0

def check_market_resolved(market_id):
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=(5, 8))
        data = r.json()
        if not data.get("closed", False):
            return None
        prices = json.loads(data.get("outcomePrices", "[0.5,0.5]"))
        yes_price = float(prices[0])
        if yes_price >= 0.95: return True
        if yes_price <= 0.05: return False
        return None
    except: return None

# ═══════════════════════════════════════════════════════════════
# V3 CORE: ENSEMBLE-DRIVEN SCAN
# ═══════════════════════════════════════════════════════════════

def scan_and_update():
    """One full scan cycle — ensemble-driven, thesis-governed."""
    now = datetime.now(timezone.utc)
    state = load_state()
    balance = state["balance"]
    new_pos = 0
    closed = 0

    for city_slug, loc in LOCATIONS.items():
        station = loc["station"]
        unit = loc["unit"]
        unit_sym = "F" if unit == "F" else "C"
        lat, lon = loc["lat"], loc["lon"]
        
        print(f"  -> {loc['name']}...", end=" ", flush=True)

        # ── SENSOR CHECK ───────────────────────────────────────
        if SENSOR_CFG.get("enabled", False):
            healthy, warns = sensor_mon.is_station_healthy(station)
            if not healthy:
                print(f"[BLACKLIST] sensor: {'; '.join(warns)}")
                continue

        dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]

        for i, date in enumerate(dates):
            dt = datetime.strptime(date, "%Y-%m-%d")
            event = get_polymarket_event(city_slug, MONTHS[dt.month - 1], dt.day, dt.year)
            if not event:
                continue

            end_date = event.get("endDate", "")
            hours = hours_to_resolution(end_date) if end_date else 0
            horizon = f"D+{i}"

            if hours < FILTERS["min_hours"] or hours > FILTERS["max_hours"]:
                continue

            # ── ENSEMBLE FORECAST ──────────────────────────────
            result = ensemble.get_cdf(lat, lon, date)
            if not result:
                # Fall back to deterministic if ensemble unavailable
                continue

            ensemble_mean = result["mean_temp"]

            # MOS correction (if enabled and trained)
            if MOS_CFG.get("enabled", False):
                month_num = dt.month
                corrected = mos_correct.correct(station, ensemble_mean, month_num)
                if corrected != ensemble_mean:
                    ensemble_mean = corrected
                    # Note: MOS corrects the mean; full ensemble re-weighting via
                    # shifting all members by the bias delta is the next upgrade tier

            # ── MARKET OUTCOMES ────────────────────────────────
            outcomes = []
            for market in event.get("markets", []):
                question = market.get("question", "")
                mid = str(market.get("id", ""))
                volume = float(market.get("volume", 0))
                rng = parse_temp_range(question)
                if not rng:
                    continue
                try:
                    prices = json.loads(market.get("outcomePrices", "[0.5,0.5]"))
                    # outcomePrices = [YES_price, NO_price]
                    # We buy YES — use the YES ask, not NO price
                    yes_price = float(prices[0])
                    # Prefer CLOB bestAsk if available, fall back to YES trade price
                    best_ask = float(market.get("bestAsk", yes_price))
                    best_bid = float(market.get("bestBid", yes_price))
                    ask = best_ask if best_ask > 0 else yes_price
                    bid = best_bid if best_bid > 0 else yes_price
                except:
                    continue
                outcomes.append({
                    "question": question, "market_id": mid,
                    "token_id": json.loads(market["clobTokenIds"])[0] 
                                if market.get("clobTokenIds") else "",
                    "range": rng, "bid": round(bid, 4), "ask": round(ask, 4),
                    "price": round(bid, 4), "spread": round(ask - bid, 4),
                    "volume": round(volume, 0),
                })

            outcomes.sort(key=lambda x: x["range"][0])

            # ── FIND MATCHING BUCKET ───────────────────────────
            matched = None
            for o in outcomes:
                t_low, t_high = o["range"]
                
                # Ensemble probability: what % of members hit this bucket?
                prob = ensemble.get_bucket_prob(lat, lon, date, t_low, t_high)
                if prob is None or prob <= 0:
                    continue
                
                # Score: probability vs market price
                ev = ensemble_ev(prob, o["ask"])
                if ev >= FILTERS["min_ev"] and o["volume"] >= FILTERS["min_volume"]:
                    matched = {
                        "outcome": o,
                        "prob": prob,
                        "ev": ev,
                        "kelly": ensemble_kelly(prob, o["ask"], KELLY_CFG["base_fraction"]),
                        "ensemble_mean": ensemble_mean,
                        "ensemble_std": result["std_temp"],
                        "n_members": result["n_members"],
                    }
                    break

            if not matched:
                continue

            # ── SPATIAL CORRELATION PENALTY ────────────────────
            if SPATIAL_CFG.get("enabled", False):
                all_mkts = load_all_markets()
                open_positions = [
                    OpenPosition(
                        city=m["city"],
                        lat=LOCATIONS[m["city"]]["lat"],
                        lon=LOCATIONS[m["city"]]["lon"],
                        bucket_low=m["position"]["bucket_low"],
                        bucket_high=m["position"]["bucket_high"],
                        shares=m["position"]["shares"],
                        entry_price=m["position"]["entry_price"],
                    )
                    for m in all_mkts
                    if m.get("position") and m["position"].get("status") == "open"
                ]
                
                penalty, corr_details = spatial_mgr.compute_penalty(
                    city_slug, lat, lon, open_positions
                )
                
                if penalty == 0.0:
                    print(f"[SKIP] {loc['name']} {date} — spatial correlation cap exceeded")
                    continue
                
                matched["kelly"] = spatial_mgr.adjusted_kelly(matched["kelly"], penalty)
                matched["spatial_penalty"] = penalty

            # ── SIZE ───────────────────────────────────────────
            raw_size = matched["kelly"] * balance
            size_usd = min(raw_size, BANKROLL["max_bet"])
            if size_usd < BANKROLL["min_position_size"]:
                continue

            shares = round(size_usd / matched["outcome"]["ask"], 2)

            # ── EXECUTION ──────────────────────────────────────
            o = matched["outcome"]
            bucket_label = f"{o['range'][0]}-{o['range'][1]}{unit_sym}"
            order_id = None
            entry_price = o["ask"]  # default: Gamma API bestAsk

            if maker and CLOB_CFG["execution"].get("maker_active", False):
                # Try maker first — limit order inside spread
                order_id = maker.execute_buy_thesis(
                    token_id=o["token_id"],
                    thesis_prob=matched["prob"],
                    thesis_bucket_low=o["range"][0],
                    thesis_bucket_high=o["range"][1],
                    thesis_kelly=matched["kelly"],
                    balance=balance,
                    max_bet=BANKROLL["max_bet"],
                    ensemble_mean=matched["ensemble_mean"],
                    ensemble_std=matched["ensemble_std"],
                    min_ev=FILTERS["min_ev"],
                    aggression=CLOB_CFG["execution"]["aggression"],
                )
                
                if order_id:
                    entry_price = o.get("bid", o["ask"])  # maker got a better price
                else:
                    # Fall back to taker — buy at Gamma bestAsk
                    try:
                        from py_clob_client.clob_types import OrderArgs
                        order_id = maker.client.create_and_post_order(
                            OrderArgs(
                                token_id=o["token_id"],
                                price=round(entry_price, 4),
                                size=shares,
                                side="BUY",
                            )
                        )
                        order_id = order_id.get("orderID") or order_id.get("id") if isinstance(order_id, dict) else str(order_id)
                    except Exception as e:
                            print(f"  [TAKER-FALLBACK] failed: {e}")
                
                if not order_id:
                    print(f"[SKIP] {loc['name']} {date} — execution failed")
                    continue
                
                # Save position
                position = {
                    "market_id": o["market_id"],
                    "token_id": o["token_id"],
                    "question": o["question"],
                    "bucket_low": o["range"][0],
                    "bucket_high": o["range"][1],
                    "entry_price": entry_price,
                    "bid_at_entry": o["bid"],
                    "spread": o["spread"],
                    "shares": shares,
                    "cost": size_usd,
                    "p": matched["prob"],
                    "ev": matched["ev"],
                    "kelly": matched["kelly"],
                    "forecast_temp": matched["ensemble_mean"],
                    "forecast_src": ENSEMBLE_CFG["model"],
                    "sigma": matched["ensemble_std"],
                    "opened_at": now.isoformat(),
                    "status": "open",
                    "clob_order_id": order_id,
                    "n_members": result["n_members"],
                }
                
                # Register thesis for monitoring
                thesis_mon.register_position(o["market_id"], create_thesis_state(
                    o["market_id"], matched["outcome"]["ask"], shares,
                    o["range"][0], o["range"][1], result, station,
                ))

            elif EXECUTION_MODE == "paper":
                # Paper mode
                position = {
                    "market_id": o["market_id"],
                    "token_id": o["token_id"],
                    "question": o["question"],
                    "bucket_low": o["range"][0],
                    "bucket_high": o["range"][1],
                    "entry_price": o["ask"],
                    "bid_at_entry": o["bid"],
                    "spread": o["spread"],
                    "shares": shares,
                    "cost": size_usd,
                    "p": matched["prob"],
                    "ev": matched["ev"],
                    "kelly": matched["kelly"],
                    "forecast_temp": matched["ensemble_mean"],
                    "forecast_src": ENSEMBLE_CFG["model"],
                    "sigma": matched["ensemble_std"],
                    "opened_at": now.isoformat(),
                    "status": "open",
                    "n_members": result["n_members"],
                }
            else:
                continue

            # Update state
            balance -= size_usd
            mkt = load_market(city_slug, date) or {
                "city": city_slug, "city_name": loc["name"], "date": date,
                "unit": unit, "station": station, "status": "open",
                "position": None, "actual_temp": None, "resolved_outcome": None,
                "pnl": None, "forecast_snapshots": [], "market_snapshots": [],
                "all_outcomes": outcomes, "created_at": now.isoformat(),
            }
            mkt["position"] = position
            mkt["all_outcomes"] = outcomes
            save_market(mkt)
            state["total_trades"] += 1
            new_pos += 1

            print(f"[BUY]  {loc['name']} {horizon} {date} | {bucket_label} | "
                  f"${o['ask']:.3f} | P={matched['prob']:.3f} | EV={matched['ev']:+.3f} | "
                  f"${size_usd:.2f} | n={result['n_members']}")

            time.sleep(0.1)

        print("ok")

    # ── THESIS-BASED POSITION MANAGEMENT ───────────────────────
    all_mkts = load_all_markets()
    for mkt in all_mkts:
        pos = mkt.get("position")
        if not pos or pos.get("status") != "open":
            continue

        mid = pos["market_id"]
        city = mkt["city"]
        loc = LOCATIONS.get(city, {})
        station = loc.get("station", "")

        # Get fresh ensemble probability
        result = ensemble.get_cdf(loc.get("lat", 0), loc.get("lon", 0), mkt["date"])
        if not result:
            continue

        prob = ensemble.get_bucket_prob(
            loc.get("lat", 0), loc.get("lon", 0), mkt["date"],
            pos["bucket_low"], pos["bucket_high"],
        )
        if prob is None:
            continue

        # Check thesis
        should_exit, reason, verdict = thesis_mon.should_exit(
            mid, prob, result["mean_temp"], result["n_members"], station,
        )

        if should_exit and THESIS_CFG.get("enabled", True):
            # Exit position
            current_price = None
            for o in mkt.get("all_outcomes", []):
                if o["market_id"] == mid:
                    current_price = o.get("bid", o["price"])
                    break

            if current_price is not None:
                if maker:
                    maker.sell_position(pos.get("token_id", ""), pos["shares"], current_price)

                pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)
                balance += pos["cost"] + pnl
                pos["closed_at"] = now.isoformat()
                pos["close_reason"] = f"thesis_{reason}"
                pos["exit_price"] = current_price
                pos["pnl"] = pnl
                pos["status"] = "closed"
                closed += 1
                
                thesis_mon.unregister_position(mid)
                print(f"[THESIS] {loc.get('name', city)} {mkt['date']} — {reason} | PnL: {'+'if pnl>=0 else ''}{pnl:.2f}")
                save_market(mkt)

        # ── AUTO-RESOLUTION (same as v2) ───────────────────────
        if mkt["status"] == "resolved":
            continue

        won = check_market_resolved(mid)
        if won is None:
            continue

        pnl = round(pos["shares"] * (1 - pos["entry_price"]), 2) if won else round(-pos["cost"], 2)
        balance += pos["cost"] + pnl
        pos["exit_price"] = 1.0 if won else 0.0
        pos["pnl"] = pnl
        pos["close_reason"] = "resolved"
        pos["closed_at"] = now.isoformat()
        pos["status"] = "closed"
        mkt["status"] = "resolved"
        mkt["resolved_outcome"] = "win" if won else "loss"

        if won: state["wins"] += 1
        else: state["losses"] += 1

        thesis_mon.unregister_position(mid)
        print(f"[{'WIN' if won else 'LOSS'}] {loc.get('name', city)} {mkt['date']} | PnL: {'+'if pnl>=0 else ''}{pnl:.2f}")
        save_market(mkt)

        # MOS: record training pair
        actual_temp = pos.get("forecast_temp")  # TODO: fetch real actual from VisualCrossing
        if MOS_CFG.get("enabled") and actual_temp is not None:
            try:
                mos_correct.add_training_pair(
                    station, pos.get("forecast_temp", 0), actual_temp,
                    dt_month=int(mkt["date"][5:7]),
                )
            except: pass

    state["balance"] = round(balance, 2)
    state["peak_balance"] = max(state.get("peak_balance", balance), balance)
    save_state(state)

    # ── MANAGE OPEN MAKER ORDERS ───────────────────────────────
    if maker:
        probs = {}
        for mkt in all_mkts:
            pos = mkt.get("position")
            if pos and pos.get("status") == "open":
                mid = pos["market_id"]
                result = ensemble.get_cdf(
                    LOCATIONS.get(mkt["city"], {}).get("lat", 0),
                    LOCATIONS.get(mkt["city"], {}).get("lon", 0),
                    mkt["date"],
                )
                if result:
                    probs[pos.get("token_id", "")] = ensemble.get_bucket_prob(
                        LOCATIONS.get(mkt["city"], {}).get("lat", 0),
                        LOCATIONS.get(mkt["city"], {}).get("lon", 0),
                        mkt["date"],
                        pos["bucket_low"], pos["bucket_high"],
                    ) or 0.0

        actions = maker.manage_open_orders(current_probs=probs)
        if actions:
            for oid, act in actions.items():
                if act != "filled":
                    print(f"  [MAKER] {oid}: {act}")

    return new_pos, closed


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def run_loop():
    print("=" * 55)
    print("  WEATHERBET v3 — ENSEMBLE + MAKER + THESIS")
    print("=" * 55)
    print(f"  Mode:      {EXECUTION_MODE.upper()}")
    print(f"  Ensemble:  {ENSEMBLE_CFG['model']} ({ENSEMBLE_CFG.get('min_members', 10)}+ members)")
    print(f"  Execution: {'MAKER (CLOB limit orders)' if CLOB_CFG['execution'].get('maker_active') else 'TAKER (market orders)'}")
    print(f"  Stops:     THESIS-BASED (no price stops)")
    print(f"  Spatial:   {'ON' if SPATIAL_CFG.get('enabled') else 'OFF'}")
    print(f"  Cities:    {len(LOCATIONS)}")
    print(f"  Balance:   ${BANKROLL['balance']:,.2f} | Max bet: ${BANKROLL['max_bet']:.2f}")
    print(f"  Scan:      {SCAN_CFG['interval_seconds']}s | Monitor: {SCAN_CFG['monitor_interval_seconds']}s")
    print(f"  Ctrl+C to stop")
    print("=" * 55)

    state = load_state()
    balance = state["balance"]

    last_full_scan = 0
    last_monitor = 0
    last_ensemble_refresh = 0

    try:
        while True:
            now_ts = time.time()

            # Full scan
            if now_ts - last_full_scan >= SCAN_CFG["interval_seconds"]:
                print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] full scan...")
                new, closed = scan_and_update()
                state = load_state()
                print(f"  balance: ${state['balance']:,.2f} | new: {new} | closed: {closed}")
                last_full_scan = now_ts
                last_monitor = now_ts
                spatial_mgr.invalidate_cache()

            # Quick thesis monitor (every 10 min)
            elif now_ts - last_monitor >= SCAN_CFG["monitor_interval_seconds"]:
                all_mkts = load_all_markets()
                open_count = len([m for m in all_mkts 
                                 if m.get("position") and m["position"].get("status") == "open"])
                
                if open_count > 0:
                    # Check thesis on all open positions
                    for mkt in all_mkts:
                        pos = mkt.get("position")
                        if not pos or pos.get("status") != "open":
                            continue
                        
                        loc = LOCATIONS.get(mkt["city"], {})
                        result = ensemble.get_cdf(
                            loc.get("lat", 0), loc.get("lon", 0), mkt["date"]
                        )
                        if not result:
                            continue
                        
                        prob = ensemble.get_bucket_prob(
                            loc.get("lat", 0), loc.get("lon", 0), mkt["date"],
                            pos["bucket_low"], pos["bucket_high"],
                        )
                        if prob is None:
                            continue
                        
                        station = loc.get("station", "")
                        should_exit, reason, _ = thesis_mon.should_exit(
                            pos["market_id"], prob, result["mean_temp"],
                            result["n_members"], station,
                        )
                        
                        if should_exit:
                            print(f"  [WATCH] {loc.get('name', mkt['city'])} {mkt['date']} — "
                                  f"thesis warning: {reason} | prob: {prob:.3f}")

                last_monitor = now_ts

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n  Shutting down...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            state = load_state()
            bal = state["balance"]
            start = state["starting_balance"]
            ret = (bal - start) / start * 100
            print(f"Balance: ${bal:,.2f} ({'+'if ret>=0 else ''}{ret:.1f}%)")
            print(f"Trades: {state['total_trades']} | W:{state['wins']} L:{state['losses']}")
            
            all_mkts = load_all_markets()
            open_pos = [m for m in all_mkts if m.get("position") and m["position"].get("status") == "open"]
            print(f"Open: {len(open_pos)}")
            for m in open_pos:
                p = m["position"]
                print(f"  {m['city_name']} {m['date']} | ${p['entry_price']:.3f} × {p['shares']} | "
                      f"P={p.get('p', '?')}")
        elif sys.argv[1] == "train_mos":
            print("Training MOS models...")
            results = mos_correct.train_all()
            for station, model in results.items():
                if model:
                    print(f"  {station}: MAE={model.mae:.2f}° ({model.n_samples} samples)")
                else:
                    print(f"  {station}: insufficient data")
    else:
        run_loop()
