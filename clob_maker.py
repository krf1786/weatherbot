#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clob_maker.py — CLOB Limit Order Execution Engine
====================================================
Replaces Gamma API taker execution with CLOB maker strategy:
places limit orders inside the spread, captures the bid-ask,
auto-cancels and re-prices on thesis drift.

Usage:
    from clob_maker import MakerEngine
    engine = MakerEngine(private_key, funder, proxy_url)
    order_id = engine.place_bid(token_id, price, size)
    engine.cancel_and_replace(order_id, token_id, new_price, size)
"""

import time
import json
import math
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import requests

# Lazy import — only when going live
# from py_clob_client.client import ClobClient
# from py_clob_client.clob_types import OrderArgs, OrderType


# ────────────────────────────────────────────────────────────────
# Data types
# ────────────────────────────────────────────────────────────────

@dataclass
class OrderBook:
    """Snapshot of the CLOB order book for a token."""
    token_id: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    spread: float
    mid: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0.0
    
    @property
    def is_liquid(self) -> bool:
        """Liquidity threshold — spread < 8 cents is tradable."""
        return self.spread < 0.08 and self.bid_size > 0 and self.ask_size > 0


@dataclass  
class MakerOrder:
    """A resting limit order on the CLOB."""
    order_id: str
    token_id: str
    side: str       # "BUY" or "SELL"
    price: float
    size: float     # shares
    placed_at: str
    thesis_bucket_low: float
    thesis_bucket_high: float
    thesis_prob: float
    ensemble_mean: float
    ensemble_std: float
    last_checked: str
    replace_count: int = 0
    unfilled_minutes: float = 0.0


# ────────────────────────────────────────────────────────────────
# Maker Engine
# ────────────────────────────────────────────────────────────────

class MakerEngine:
    """
    Places limit orders at strategic price points inside the spread.
    Auto-cancels and re-prices on thesis drift or timeout.
    
    Strategy:
    - BUY:  Place limit at best_bid + spread/4  (slightly above bid, below ask)
    - SELL: Place limit at best_ask - spread/4  (slightly below ask, above bid)
    - Re-price every unfilled_minutes_max (default 15 min)
    - Cancel if thesis probability drops below retention threshold
    - Cancel if spread widens beyond max_spread
    """
    
    def __init__(
        self,
        private_key: str,
        funder: str,
        proxy_url: Optional[str] = None,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        unfilled_minutes_max: int = 15,
        max_spread: float = 0.08,
        gamma_host: str = "https://gamma-api.polymarket.com",
    ):
        self.private_key = private_key
        self.funder = funder
        self.proxy_url = proxy_url
        
        # Patch the py_clob_client httpx client to use proxy
        if proxy_url:
            import httpx as _httpx
            from py_clob_client.http_helpers import helpers as _helpers
            _helpers._http_client = _httpx.Client(http2=True, proxy=proxy_url)
        self.host = host
        self.chain_id = chain_id
        self.unfilled_minutes_max = unfilled_minutes_max
        self.max_spread = max_spread
        self.gamma_host = gamma_host
        
        self._client = None
        self._session = None
        self._orders: Dict[str, MakerOrder] = {}
        self._lock = threading.Lock()
    
    # ── Lazy initialization ────────────────────────────────────
    
    @property
    def client(self):
        """Lazy-init CLOB client (imports SDK only when needed)."""
        if self._client is None:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            
            # Check for cached creds
            creds_file = Path("data/executor/api_creds.json")
            creds = None
            if creds_file.exists():
                creds_dict = json.loads(creds_file.read_text())
                creds = ApiCreds(**creds_dict)
            
            self._client = ClobClient(
                host=self.host,
                chain_id=self.chain_id,
                key=self.private_key,
                creds=creds,
                signature_type=2,
                funder=self.funder,
            )
            
            # Create creds if needed
            if not creds:
                api_creds = self._client.create_or_derive_api_creds()
                creds_file.parent.mkdir(parents=True, exist_ok=True)
                creds_file.write_text(json.dumps({
                    "api_key": api_creds.api_key,
                    "api_secret": api_creds.api_secret,
                    "api_passphrase": api_creds.api_passphrase,
                }, indent=2))
                self._client.set_api_creds(api_creds)
        
        return self._client
    
    @property
    def session(self):
        """Lazy-init requests session with proxy."""
        if self._session is None:
            self._session = requests.Session()
            if self.proxy_url:
                self._session.proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url,
                }
        return self._session
    
    # ── Order Book ─────────────────────────────────────────────
    
    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """
        Fetch live order book from CLOB for a specific token.
        Uses the Gamma API's book endpoint (works without auth).
        """
        try:
            url = f"{self.gamma_host}/book?token_id={token_id}"
            resp = self.session.get(url, timeout=(3, 8))
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            
            if best_bid <= 0 or best_ask <= 0:
                return None
            
            spread = round(best_ask - best_bid, 4)
            mid = round((best_bid + best_ask) / 2, 4)
            
            return OrderBook(
                token_id=token_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=float(data.get("bidSize", 0)),
                ask_size=float(data.get("askSize", 0)),
                spread=spread,
                mid=mid,
            )
            
        except Exception as e:
            print(f"  [CLOB-BOOK] failed: {e}")
            return None
    
    # ── Strategic Pricing ──────────────────────────────────────
    
    def compute_limit_price(
        self,
        book: OrderBook,
        side: str,
        aggression: float = 0.25,  # 0 = passive (at best), 1 = aggressive (at mid)
    ) -> float:
        """
        Compute optimal limit price inside the spread.
        
        BUY side:  best_bid + aggression * spread
        SELL side: best_ask - aggression * spread
        
        Aggression 0.25 means: 25% into the spread from the resting side.
        Lower = more passive = better price, but may never fill.
        """
        if side.upper() == "BUY":
            price = book.best_bid + aggression * book.spread
            # Never cross the ask
            price = min(price, book.best_ask - 0.001)
        else:
            price = book.best_ask - aggression * book.spread
            # Never cross the bid
            price = max(price, book.best_bid + 0.001)
        
        return round(max(0.001, min(0.999, price)), 4)
    
    # ── Order Placement ────────────────────────────────────────
    
    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        thesis_prob: float = 0.0,
        thesis_bucket_low: float = 0.0,
        thesis_bucket_high: float = 0.0,
        ensemble_mean: float = 0.0,
        ensemble_std: float = 0.0,
    ) -> Optional[str]:
        """
        Place a limit order on the CLOB.
        
        Returns order_id on success, None on failure.
        Tracks the order internally for re-pricing.
        
        Args:
            token_id: CLOB token ID
            side: "BUY" or "SELL"
            price: limit price per share (0.0-1.0)
            size: number of shares
            thesis_*: metadata for thesis-based monitoring
        """
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            
            # Round to exchange requirements
            price = round(price, 4)
            size = round(size, 2)
            
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.upper(),
            )
            
            result = self.client.create_and_post_order(order_args)
            order_id = result.get("orderID") or result.get("id") or str(result)
            
            if not order_id:
                return None
            
            now = datetime.now(timezone.utc).isoformat()
            
            with self._lock:
                self._orders[order_id] = MakerOrder(
                    order_id=order_id,
                    token_id=token_id,
                    side=side.upper(),
                    price=price,
                    size=size,
                    placed_at=now,
                    thesis_bucket_low=thesis_bucket_low,
                    thesis_bucket_high=thesis_bucket_high,
                    thesis_prob=thesis_prob,
                    ensemble_mean=ensemble_mean,
                    ensemble_std=ensemble_std,
                    last_checked=now,
                )
            
            return order_id
            
        except Exception as e:
            print(f"  [CLOB-ORDER] failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting limit order."""
        try:
            self.client.cancel_order(order_id)
            with self._lock:
                self._orders.pop(order_id, None)
            return True
        except Exception as e:
            print(f"  [CLOB-CANCEL] {order_id}: {e}")
            return False
    
    def cancel_and_replace(
        self,
        order_id: str,
        token_id: str,
        new_price: float,
        new_size: float,
        thesis_prob: float = 0.0,
        thesis_bucket_low: float = 0.0,
        thesis_bucket_high: float = 0.0,
        ensemble_mean: float = 0.0,
        ensemble_std: float = 0.0,
    ) -> Optional[str]:
        """
        Cancel an existing order and place a new one at updated price.
        Atomic: if cancel fails, don't place new order.
        """
        if not self.cancel_order(order_id):
            return None
        
        return self.place_limit_order(
            token_id=token_id,
            side="BUY",  # weather bets are always BUY (we're betting YES)
            price=new_price,
            size=new_size,
            thesis_prob=thesis_prob,
            thesis_bucket_low=thesis_bucket_low,
            thesis_bucket_high=thesis_bucket_high,
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
        )
    
    # ── Order Monitoring ───────────────────────────────────────
    
    def check_order_status(self, order_id: str) -> Optional[str]:
        """
        Check if an order is still open.
        Returns: "open", "filled", "cancelled", or None on error.
        """
        try:
            result = self.client.get_order(order_id)
            status = result.get("status", "").lower()
            if status in ("open", "live", "active"):
                return "open"
            if status in ("filled", "matched"):
                return "filled"
            return "cancelled"
        except Exception:
            return None
    
    def should_reprice(self, order: MakerOrder) -> bool:
        """Check if an unfilled order needs re-pricing."""
        placed = datetime.fromisoformat(order.placed_at)
        now = datetime.now(timezone.utc)
        minutes_open = (now - placed).total_seconds() / 60
        order.unfilled_minutes = minutes_open
        return minutes_open >= self.unfilled_minutes_max
    
    def is_thesis_intact(
        self,
        order: MakerOrder,
        current_prob: float,
        prob_threshold_drop: float = 0.10,
    ) -> bool:
        """
        Check if the thesis still supports this position.
        Cancel if probability dropped more than threshold from entry.
        """
        drop = order.thesis_prob - current_prob
        return drop < prob_threshold_drop
    
    # ── High-Level Entry Point ─────────────────────────────────
    
    def execute_buy_thesis(
        self,
        token_id: str,
        thesis_prob: float,
        thesis_bucket_low: float,
        thesis_bucket_high: float,
        thesis_kelly: float,
        balance: float,
        max_bet: float,
        ensemble_mean: float,
        ensemble_std: float,
        min_ev: float = 0.05,
        aggression: float = 0.25,
    ) -> Optional[str]:
        """
        Full execution pipeline for a thesis-driven buy:
        1. Fetch order book
        2. Check liquidity
        3. Compute EV at limit price (not ask)
        4. Place limit order inside spread
        
        Returns order_id on success, None if conditions not met.
        """
        # 1. Order book
        book = self.get_order_book(token_id)
        if not book or not book.is_liquid:
            return None
        
        # 2. Compute limit price
        limit_price = self.compute_limit_price(book, "BUY", aggression)
        
        # 3. EV at limit price (not ask — this is the maker advantage)
        if limit_price <= 0 or limit_price >= 1:
            return None
        
        b = 1.0 / limit_price - 1.0
        ev = thesis_prob * b - (1.0 - thesis_prob)
        
        if ev < min_ev:
            return None
        
        # 4. Size — Kelly at limit price
        f = (thesis_prob * b - (1.0 - thesis_prob)) / b
        kelly_f = min(max(0.0, f * 0.25), 1.0)
        size_usd = min(kelly_f * balance, max_bet)
        shares = round(size_usd / limit_price, 2)
        
        if shares < 1.0 or size_usd < 0.50:
            return None
        
        # 5. Place order
        return self.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=limit_price,
            size=shares,
            thesis_prob=thesis_prob,
            thesis_bucket_low=thesis_bucket_low,
            thesis_bucket_high=thesis_bucket_high,
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
        )
    
    def manage_open_orders(
        self,
        current_probs: Dict[str, float],  # token_id → current ensemble prob
        prob_threshold_drop: float = 0.10,
    ) -> Dict[str, str]:
        """
        Monitor all open orders:
        - Check if filled (remove from tracking)
        - Re-price if unfilled too long
        - Cancel if thesis degraded
        
        Returns dict of {order_id: action_taken}
        """
        actions = {}
        stale = []
        
        with self._lock:
            for order_id, order in list(self._orders.items()):
                status = self.check_order_status(order_id)
                
                if status is None:
                    # Can't check — leave it
                    continue
                
                if status == "filled":
                    self._orders.pop(order_id, None)
                    actions[order_id] = "filled"
                    continue
                
                if status == "cancelled":
                    self._orders.pop(order_id, None)
                    actions[order_id] = "cancelled_external"
                    continue
                
                # Order is still open — check thesis
                current_prob = current_probs.get(order.token_id)
                
                if current_prob is not None:
                    # Check thesis degradation
                    if not self.is_thesis_intact(order, current_prob, prob_threshold_drop):
                        if self.cancel_order(order_id):
                            actions[order_id] = "cancelled_thesis"
                        continue
                
                # Check if needs re-pricing
                if self.should_reprice(order):
                    stale.append(order)
                    actions[order_id] = "needs_reprice"
                
                order.last_checked = datetime.now(timezone.utc).isoformat()
        
        return actions
    
    def get_open_order_count(self) -> int:
        with self._lock:
            return len(self._orders)
    
    def get_balance(self) -> Dict:
        """Get USDC balance from the CLOB."""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            bal = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            return {
                "usdc_balance": float(bal.get("balance", 0)),
                "available": float(bal.get("balance", 0)),
            }
        except Exception:
            return {"usdc_balance": 0, "available": 0}
    
    def sell_position(self, token_id: str, shares: float, price: float) -> Optional[str]:
        """
        Place a limit sell order to exit a position.
        Price at best_bid + small premium for quick fill.
        """
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            
            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=round(shares, 2),
                side="SELL",
            )
            
            result = self.client.create_and_post_order(order_args)
            order_id = result.get("orderID") or result.get("id") or str(result)
            return order_id
            
        except Exception as e:
            print(f"  [CLOB-SELL] failed: {e}")
            return None
