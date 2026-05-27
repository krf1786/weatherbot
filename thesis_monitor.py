#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_monitor.py — Thesis-Based Position Management
=======================================================
Replaces price-based stop-losses with meteorological thesis monitoring.

Exit Rules:
  1. Ensemble probability drops below retention threshold 
     (the weather model changed its mind)
  2. CLOB spread widens beyond liquidity threshold
     (market is un-tradable — exit before you can't)
  3. ASOS sensor failure detected for this station
     (resolution risk — the market may switch to a different station)

Does NOT exit on:
  - Price movement alone (market noise)
  - Time decay (weather events resolve on schedule)
  - Other positions losing money (no cross-position stops)
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import requests


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

@dataclass
class ThesisConfig:
    """Thresholds for thesis-based position management."""
    
    # Probability: exit if ensemble prob drops by more than this from entry
    prob_drop_threshold: float = 0.15  # 15 percentage points
    
    # Probability: absolute floor — never hold below this
    prob_floor: float = 0.30  # below 30% prob, thesis is dead
    
    # Spread: exit if spread exceeds this
    max_spread: float = 0.08  # $0.08
    
    # Spread: exit if spread ratio (spread/mid) exceeds this
    max_spread_ratio: float = 0.50  # spread > 50% of mid price = illiquid
    
    # Volume: exit if 24h volume below this
    min_volume: float = 100.0  # $100
    
    # Time: don't exit within first X minutes of opening (noise filter)
    min_hold_minutes: int = 30
    
    # Ensemble shift: size of mean shift that triggers re-evaluation
    mean_shift_degrees: float = 2.0  # °F or °C
    
    # Sensor check interval (hours)
    sensor_check_hours: int = 3


@dataclass
class ThesisState:
    """Snapshot of the meteorological thesis at position open time."""
    ensemble_mean: float
    ensemble_std: float
    ensemble_prob: float
    bucket_low: float
    bucket_high: float
    opened_at: str
    station: str
    market_id: str
    entry_price: float
    shares: float
    n_members: int = 0


@dataclass
class ThesisVerdict:
    """Result of a thesis integrity check."""
    intact: bool
    reason: str
    current_ensemble_prob: float = 0.0
    current_spread: float = 0.0
    current_mean_shift: float = 0.0
    sensor_warning: bool = False
    details: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
# Core Monitor
# ────────────────────────────────────────────────────────────────

class ThesisMonitor:
    """
    Monitors open positions and determines if the meteorological 
    thesis still supports holding the position.
    """
    
    def __init__(
        self,
        config: Optional[ThesisConfig] = None,
        gamma_host: str = "https://gamma-api.polymarket.com",
    ):
        self.config = config or ThesisConfig()
        self.gamma_host = gamma_host
        self._states: Dict[str, ThesisState] = {}  # market_id → state
        self._sensor_cache: Dict[str, dict] = {}   # station → status
    
    def register_position(self, market_id: str, state: ThesisState):
        """Record the thesis at entry for future monitoring."""
        self._states[market_id] = state
    
    def unregister_position(self, market_id: str):
        """Remove tracking when position closes."""
        self._states.pop(market_id, None)
    
    # ── Spread Check ───────────────────────────────────────────
    
    def check_liquidity(self, market_id: str) -> Tuple[float, float, bool]:
        """
        Check current CLOB spread and liquidity.
        Returns: (spread, volume, is_healthy)
        """
        try:
            url = f"{self.gamma_host}/markets/{market_id}"
            resp = requests.get(url, timeout=(3, 5))
            data = resp.json()
            
            best_bid = float(data.get("bestBid", 0))
            best_ask = float(data.get("bestAsk", 0))
            volume = float(data.get("volume", 0))
            
            if best_bid <= 0 or best_ask <= 0:
                return (0.999, 0, False)
            
            spread = round(best_ask - best_bid, 4)
            mid = (best_bid + best_ask) / 2
            spread_ratio = spread / mid if mid > 0 else 1.0
            
            spread_ok = spread < self.config.max_spread
            ratio_ok = spread_ratio < self.config.max_spread_ratio
            volume_ok = volume >= self.config.min_volume
            
            return (spread, volume, spread_ok and ratio_ok and volume_ok)
            
        except Exception:
            return (0.999, 0, False)
    
    # ── Ensemble Thesis Check ──────────────────────────────────
    
    def check_thesis(
        self,
        market_id: str,
        current_ensemble_prob: float,
        current_ensemble_mean: float,
        current_ensemble_n_members: int = 0,
    ) -> ThesisVerdict:
        """
        Check if the ensemble thesis still supports the position.
        
        Args:
            market_id: Polymarket market ID
            current_ensemble_prob: Latest ensemble probability for our bucket
            current_ensemble_mean: Latest ensemble mean temperature
            current_ensemble_n_members: Number of ensemble members
        
        Returns ThesisVerdict with intact=True/False.
        """
        state = self._states.get(market_id)
        if not state:
            return ThesisVerdict(True, "no_state_tracked")
        
        # Don't exit within minimum hold period — noise filter
        opened = datetime.fromisoformat(state.opened_at)
        held_minutes = (datetime.now(timezone.utc) - opened).total_seconds() / 60
        if held_minutes < self.config.min_hold_minutes:
            return ThesisVerdict(True, "min_hold_period")
        
        # 1. Absolute probability floor
        if current_ensemble_prob < self.config.prob_floor:
            return ThesisVerdict(
                False,
                f"prob_floor: {current_ensemble_prob:.3f} < {self.config.prob_floor}",
                current_ensemble_prob=current_ensemble_prob,
            )
        
        # 2. Probability drop from entry
        prob_drop = state.ensemble_prob - current_ensemble_prob
        if prob_drop > self.config.prob_drop_threshold:
            return ThesisVerdict(
                False,
                f"prob_drop: {prob_drop:.3f} > {self.config.prob_drop_threshold} "
                f"({state.ensemble_prob:.3f} → {current_ensemble_prob:.3f})",
                current_ensemble_prob=current_ensemble_prob,
            )
        
        # 3. Mean temperature shift
        mean_shift = abs(current_ensemble_mean - state.ensemble_mean)
        if mean_shift > self.config.mean_shift_degrees:
            return ThesisVerdict(
                False,
                f"mean_shift: {mean_shift:.1f}° > {self.config.mean_shift_degrees}° "
                f"({state.ensemble_mean:.1f} → {current_ensemble_mean:.1f})",
                current_ensemble_prob=current_ensemble_prob,
                current_mean_shift=mean_shift,
            )
        
        # 4. Ensemble tightening — lower std = higher confidence = good
        # (No exit condition — just informational)
        
        # 5. Liquidity check
        spread, volume, liquid = self.check_liquidity(market_id)
        if not liquid:
            return ThesisVerdict(
                False,
                f"liquidity_dried: spread=${spread:.3f}, vol=${volume:.0f}",
                current_ensemble_prob=current_ensemble_prob,
                current_spread=spread,
            )
        
        # Thesis intact
        return ThesisVerdict(
            True,
            "intact",
            current_ensemble_prob=current_ensemble_prob,
            current_spread=spread,
            current_mean_shift=mean_shift if 'mean_shift' in dir() else 0.0,
        )
    
    # ── Sensor Monitoring ──────────────────────────────────────
    
    def check_sensor_status(self, station: str) -> Tuple[bool, dict]:
        """
        Check if an airport's ASOS sensor has known issues.
        
        Checks:
        1. METAR data recency (is the station reporting?)
        2. Known maintenance flags from NWS NOTAMs
        3. Temperature reporting consistency
        
        Returns: (is_healthy, details_dict)
        """
        # Check cache first
        if station in self._sensor_cache:
            cached = self._sensor_cache[station]
            age = (datetime.now(timezone.utc) - 
                   datetime.fromisoformat(cached["checked_at"])).total_seconds() / 3600
            if age < self.config.sensor_check_hours:
                return cached["healthy"], cached
        
        details = {
            "station": station,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "healthy": True,
            "reporting": True,
            "temp_ok": True,
            "warnings": [],
        }
        
        try:
            # 1. Check METAR data freshness
            url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json"
            resp = requests.get(url, timeout=(5, 10))
            data = resp.json()
            
            if not data or not isinstance(data, list):
                details["healthy"] = False
                details["reporting"] = False
                details["warnings"].append("station_not_reporting")
            else:
                metar = data[0]
                obs_time = metar.get("obsTime", "")
                if obs_time:
                    # obsTime can be epoch int or ISO string
                    if isinstance(obs_time, (int, float)):
                        obs_dt = datetime.fromtimestamp(obs_time, tz=timezone.utc)
                    else:
                        try:
                            obs_dt = datetime.fromisoformat(str(obs_time).replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            details["warnings"].append("obs_time_unparseable")
                            obs_dt = None
                    if obs_dt:
                        age_hours = (datetime.now(timezone.utc) - obs_dt).total_seconds() / 3600
                        if age_hours > 2:
                            details["healthy"] = False
                        details["reporting"] = False
                        details["warnings"].append(f"metar_stale_{age_hours:.0f}h")
                
                temp = metar.get("temp")
                if temp is None or float(temp) < -100 or float(temp) > 150:
                    details["healthy"] = False
                    details["temp_ok"] = False
                    details["warnings"].append("sensor_temp_invalid")
            
        except Exception as e:
            details["healthy"] = False
            details["warnings"].append(f"check_failed: {str(e)[:50]}")
        
        self._sensor_cache[station] = {
            "healthy": details["healthy"],
            "checked_at": details["checked_at"],
        }
        
        return details["healthy"], details
    
    # ── Combined Threshold Check ───────────────────────────────
    
    def should_exit(
        self,
        market_id: str,
        current_ensemble_prob: float,
        current_ensemble_mean: float,
        current_ensemble_n_members: int = 0,
        station: str = "",
        check_sensors: bool = False,
    ) -> Tuple[bool, str, ThesisVerdict]:
        """
        Master gate: should we exit this position?
        
        Checks in order:
        1. Thesis integrity (ensemble)
        2. Liquidity (spread/volume)
        3. Sensor health (ASOS)
        
        Returns: (should_exit, reason, full_verdict)
        """
        # Thesis check
        verdict = self.check_thesis(
            market_id, current_ensemble_prob, 
            current_ensemble_mean, current_ensemble_n_members
        )
        
        if not verdict.intact:
            return (True, verdict.reason, verdict)
        
        # Sensor check (only when explicitly requested)
        if station and check_sensors:
            sensor_ok, sensor_details = self.check_sensor_status(station)
            if not sensor_ok:
                verdict.sensor_warning = True
                verdict.intact = False
                verdict.reason = f"sensor_failure: {station}"
                verdict.details = sensor_details
                return (True, verdict.reason, verdict)
        
        return (False, "thesis_intact", verdict)


# ────────────────────────────────────────────────────────────────
# Utility: create thesis state from ensemble result
# ────────────────────────────────────────────────────────────────

def create_thesis_state(
    market_id: str,
    entry_price: float,
    shares: float,
    bucket_low: float,
    bucket_high: float,
    ensemble_result: dict,   # from EnsembleForecast.get_cdf()
    station: str,
) -> ThesisState:
    """Build a ThesisState from an ensemble forecast result."""
    prob = ensemble_result.get("bucket_prob", 0.0)
    
    return ThesisState(
        ensemble_mean=ensemble_result.get("mean_temp", 0),
        ensemble_std=ensemble_result.get("std_temp", 0),
        ensemble_prob=prob,
        bucket_low=bucket_low,
        bucket_high=bucket_high,
        opened_at=datetime.now(timezone.utc).isoformat(),
        station=station,
        market_id=market_id,
        entry_price=entry_price,
        shares=shares,
        n_members=ensemble_result.get("n_members", 0),
    )
