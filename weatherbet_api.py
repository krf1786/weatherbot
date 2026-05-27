#!/usr/bin/env python3
"""
weatherbet_api.py — Tiny API server for WeatherBet dashboard.
Reads bot data files and serves JSON for the dashboard.
Run: python weatherbet_api.py --port 8777
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "state.json"
MARKETS_DIR = DATA_DIR / "markets"

PORT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8777


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"balance": 0, "starting_balance": 0, "total_trades": 0, "wins": 0, "losses": 0}


def load_all_markets():
    markets = []
    if MARKETS_DIR.exists():
        for f in sorted(MARKETS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                markets.append(m)
            except Exception:
                pass
    return markets


def get_summary():
    state = load_state()
    markets = load_all_markets()

    open_pos = [m for m in markets if m.get("position") and m["position"].get("status") == "open"]
    resolved = [m for m in markets if m["status"] == "resolved"]
    closed_positions = [m for m in markets if m.get("position") and m["position"].get("status") == "closed"]

    # Unrealized PnL
    unrealized_total = 0.0
    positions = []
    for m in open_pos:
        pos = m["position"]
        entry = pos["entry_price"]
        # Get current best bid
        current = entry  # fallback
        # Try to get from latest snapshot
        snaps = m.get("market_snapshots", [])
        outcomes = m.get("all_outcomes", [])
        for o in outcomes:
            if o["market_id"] == pos.get("market_id"):
                current = o.get("bid", o["price"])
                break
        if current is None:
            current = entry

        unrealized = round((current - entry) * pos["shares"], 2)
        unrealized_total += unrealized

        positions.append({
            "city": m["city_name"],
            "date": m["date"],
            "station": m["station"],
            "unit": m["unit"],
            "bucket": f"{pos['bucket_low']}-{pos['bucket_high']}{'F' if m['unit']=='F' else 'C'}",
            "entry_price": round(entry, 3),
            "current_price": round(current, 3),
            "shares": pos["shares"],
            "cost": pos["cost"],
            "unrealized_pnl": unrealized,
            "unrealized_pct": round((current - entry) / entry * 100, 1) if entry else 0,
            "forecast_src": pos.get("forecast_src", "unknown").upper(),
            "forecast_temp": pos.get("forecast_temp"),
            "opened_at": pos.get("opened_at"),
            "clob_order_id": pos.get("clob_order_id"),
        })

    # Trade history
    history = []
    for m in resolved + closed_positions:
        pos = m.get("position", {})
        if not pos:
            continue
        history.append({
            "city": m["city_name"],
            "date": m["date"],
            "unit": m["unit"],
            "bucket": f"{pos.get('bucket_low')}-{pos.get('bucket_high')}{'F' if m['unit']=='F' else 'C'}" if pos.get('bucket_low') else "",
            "entry_price": round(pos.get("entry_price", 0), 3),
            "exit_price": round(pos.get("exit_price", 0), 3) if pos.get("exit_price") else None,
            "pnl": pos.get("pnl"),
            "close_reason": pos.get("close_reason", "unknown"),
            "outcome": m.get("resolved_outcome") or pos.get("close_reason", "closed"),
            "closed_at": pos.get("closed_at"),
        })

    # Performance by city
    by_city = {}
    for m in resolved + closed_positions:
        pos = m.get("position", {})
        if not pos:
            continue
        city = m["city_name"]
        if city not in by_city:
            by_city[city] = {"wins": 0, "losses": 0, "pnl": 0.0}
        pnl = pos.get("pnl") or 0
        if pnl > 0:
            by_city[city]["wins"] += 1
        else:
            by_city[city]["losses"] += 1
        by_city[city]["pnl"] = round(by_city[city]["pnl"] + pnl, 2)

    city_stats = []
    for name, stats in sorted(by_city.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total = stats["wins"] + stats["losses"]
        city_stats.append({
            "name": name,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total": total,
            "win_rate": round(stats["wins"] / total * 100, 1) if total else 0,
            "pnl": stats["pnl"],
        })

    # Balance
    bal = state["balance"]
    start = state["starting_balance"]
    ret_pct = round((bal - start) / start * 100, 2) if start else 0

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": os.environ.get("WEATHERBET_MODE", "paper"),
        "balance": {
            "current": bal,
            "starting": start,
            "return_pct": ret_pct,
            "peak": state.get("peak_balance", bal),
        },
        "trades": {
            "total": state.get("total_trades", 0),
            "wins": state.get("wins", 0),
            "losses": state.get("losses", 0),
            "win_rate": round(state["wins"] / max(state["wins"] + state["losses"], 1) * 100, 1),
        },
        "positions": {
            "open": len(positions),
            "unrealized_pnl": round(unrealized_total, 2),
            "items": positions,
        },
        "history": history,
        "by_city": city_stats,
    }


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/summary":
            self._json(get_summary())
        elif self.path == "/api/positions":
            data = get_summary()
            self._json(data["positions"])
        elif self.path == "/api/history":
            data = get_summary()
            self._json(data["history"])
        elif self.path == "/api/status":
            state = load_state()
            self._json(state)
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404")

    def _json(self, data):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silent


if __name__ == "__main__":
    print(f"WeatherBet API on http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), APIHandler).serve_forever()
