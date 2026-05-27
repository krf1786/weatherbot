#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble.py — Ensemble Forecast CDF Calculator
================================================
Replaces deterministic single-model forecasts with probabilistic ensemble 
distributions. Queries Open-Meteo's Ensemble API (ECMWF IFS ENS, 51 members) 
and builds empirical CDFs/PDFs per airport coordinate.

Usage:
    from ensemble import EnsembleForecast
    ens = EnsembleForecast()
    cdf = ens.get_cdf("nyc", "2026-05-28")  # → {prob, buckets, cdf_values}
"""

import time
import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np


# ────────────────────────────────────────────────────────────────
# Ensemble model config
# ────────────────────────────────────────────────────────────────

ENSEMBLE_MODELS = {
    "ecmwf_ens": {
        "endpoint": "https://ensemble-api.open-meteo.com/v1/ensemble",
        "model": "ecmwf_ifs025",         # 50 members, 0.25° (~28km)
        "members": 50,
        "update_times": [0, 12],        # UTC hours — updated at 00Z and 12Z
    },
    "gefs": {
        "endpoint": "https://ensemble-api.open-meteo.com/v1/ensemble",
        "model": "gfs_ensemble",         # GEFS, 31 members, 0.25°
        "members": 31,
        "update_times": [0, 6, 12, 18],
    },
}

# How stale ensemble data can be before requiring a re-fetch (hours)
MAX_ENSEMBLE_AGE_HOURS = 6


# ────────────────────────────────────────────────────────────────
# Core class
# ────────────────────────────────────────────────────────────────

class EnsembleForecast:
    """
    Fetches ensemble members for an airport coordinate, builds empirical 
    CDFs for any target date, and calculates exact bucket probabilities 
    using ensemble membership counts.
    """
    
    def __init__(self, cache_dir: Path = Path("data/ensemble")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, dict] = {}
        self._load_cache()
    
    # ── Cache ──────────────────────────────────────────────────
    
    def _cache_key(self, lat: float, lon: float, ensemble: str) -> str:
        return f"{lat:.4f}_{lon:.4f}_{ensemble}"
    
    def _cache_path(self, lat: float, lon: float, ensemble: str) -> Path:
        return self.cache_dir / f"{self._cache_key(lat, lon, ensemble)}.json"
    
    def _load_cache(self):
        """Load all cached ensemble data from disk."""
        for f in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                key = f.stem
                fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01T00:00:00"))
                age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
                if age < MAX_ENSEMBLE_AGE_HOURS:
                    self._cache[key] = data
            except Exception:
                pass
    
    # ── Fetch ──────────────────────────────────────────────────
    
    def fetch_ensemble(
        self,
        lat: float,
        lon: float,
        ensemble: str = "ecmwf_ens",
        max_retries: int = 3,
    ) -> Optional[dict]:
        """
        Query Open-Meteo Ensemble API for all members at a coordinate.
        
        Returns dict with keys: members (list of member dicts), 
        fetched_at, model, lat, lon.
        Each member dict has: {member_id, dates: {date: temp}}
        """
        cache_key = self._cache_key(lat, lon, ensemble)
        cache_path = self._cache_path(lat, lon, ensemble)
        
        # Check cache
        if cache_key in self._cache:
            data = self._cache[cache_key]
            fetched = datetime.fromisoformat(data["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age < MAX_ENSEMBLE_AGE_HOURS:
                return data
        
        model_cfg = ENSEMBLE_MODELS.get(ensemble)
        if not model_cfg:
            raise ValueError(f"Unknown ensemble: {ensemble}")
        
        url = (
            f"{model_cfg['endpoint']}"
            f"?latitude={lat}&longitude={lon}"
            f"&models={model_cfg['model']}"
            f"&daily=temperature_2m_max"
            f"&forecast_days=7"
            f"&timezone=UTC"
        )
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=(10, 30))
                data = resp.json()
                
                if "error" in data:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return None
                
                # Parse ensemble members from response
                # Open-Meteo ensemble returns per-member daily data
                members = self._parse_ensemble_response(data, model_cfg["members"])
                
                result = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "model": model_cfg["model"],
                    "ensemble": ensemble,
                    "lat": lat,
                    "lon": lon,
                    "members": members,
                    "n_members": len(members),
                }
                
                # Cache to disk
                cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                self._cache[cache_key] = result
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"  [ENSEMBLE] fetch failed: {e}")
                    return None
        
        return None
    
    def _parse_ensemble_response(self, data: dict, expected_members: int) -> List[dict]:
        """
        Parse Open-Meteo's ensemble response format.
        
        Open-Meteo returns daily data with per-member arrays. 
        We extract each member's forecast as a list of (date, temp) pairs.
        """
        members = []
        daily = data.get("daily", {})
        
        if not daily:
            # Fall back to hourly if daily not available
            hourly = data.get("hourly", {})
            if hourly:
                return self._parse_hourly_ensemble(hourly, expected_members)
            return members
        
        dates = daily.get("time", [])
        if not dates:
            return members
        
        # Temperature data per member: temperature_2m_max_member01, ...member50
        for i in range(1, expected_members + 1):
            key = f"temperature_2m_max_member{i:02d}"
            temps = daily.get(key, [])
            if not temps:
                # Also try the mean if no per-member data
                continue
            
            if temps:
                member_dates = {}
                for date, temp in zip(dates, temps):
                    if temp is not None:
                        member_dates[date] = round(temp, 2)
                if member_dates:
                    members.append({
                        "member_id": i,
                        "dates": member_dates,
                    })
        
        return members
    
    def _parse_hourly_ensemble(self, hourly: dict, expected_members: int) -> List[dict]:
        """Fallback parser for hourly ensemble data."""
        members = []
        times = hourly.get("time", [])
        if not times:
            return members
        
        for i in range(1, expected_members + 1):
            key = f"member_{i:02d}_temperature_2m"
            temps = hourly.get(key, [])
            if temps:
                # Aggregate to daily max per date
                daily_max = {}
                for t, temp in zip(times, temps):
                    if temp is not None:
                        date = t[:10]  # YYYY-MM-DD
                        daily_max[date] = max(daily_max.get(date, -999), temp)
                members.append({
                    "member_id": i,
                    "dates": {d: round(v, 2) for d, v in daily_max.items()},
                })
        
        return members
    
    # ── CDF Computation ───────────────────────────────────────
    
    def get_ensemble_temps(
        self,
        members: List[dict],
        target_date: str,
    ) -> np.ndarray:
        """
        Extract all member forecasts for a specific date into a numpy array.
        Returns None if the date isn't covered.
        """
        temps = []
        for m in members:
            t = m["dates"].get(target_date)
            if t is not None:
                temps.append(t)
        
        if not temps:
            return None
        
        return np.array(temps, dtype=np.float64)
    
    def build_cdf(
        self,
        temps: np.ndarray,
        n_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an empirical CDF from ensemble member temperatures.
        
        Uses Gaussian KDE + empirical CDF hybrid:
        - KDE gives smooth probability density
        - Integration gives the CDF
        - For small ensembles (<30 members), uses smoother bandwidth
        
        Returns:
            x: temperature grid
            cdf: cumulative probability at each x
        """
        from scipy.stats import gaussian_kde
        
        n = len(temps)
        
        # Bandwidth selection — wider for small ensembles
        if n < 20:
            bw_method = "scott"
        elif n < 50:
            bw_method = 0.5  # Silverman's rule of thumb reduction
        else:
            bw_method = "scott"
        
        kde = gaussian_kde(temps, bw_method=bw_method)
        
        # Build grid spanning 4σ around the mean
        sigma = np.std(temps)
        mu = np.mean(temps)
        x_min = mu - 4 * sigma
        x_max = mu + 4 * sigma
        x = np.linspace(x_min, x_max, n_points)
        
        # PDF → CDF via cumulative integration
        pdf = kde.evaluate(x)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]  # normalize to [0, 1]
        
        return x, cdf
    
    def bucket_probability(
        self,
        temps: np.ndarray,
        bucket_low: float,
        bucket_high: float,
    ) -> float:
        """
        Calculate exact probability that temperature falls in [bucket_low, bucket_high]
        using ensemble membership count.
        
        For edge buckets (below N or above N):
          bucket_low = -999 → everything below bucket_high
          bucket_high = 999 → everything above bucket_low
        
        Returns probability in [0, 1].
        """
        n = len(temps)
        if n == 0:
            return 0.0
        
        if bucket_low == -999:
            count = np.sum(temps <= bucket_high)
        elif bucket_high == 999:
            count = np.sum(temps >= bucket_low)
        else:
            count = np.sum((temps >= bucket_low) & (temps <= bucket_high))
        
        return count / n
    
    def bucket_probability_kde(
        self,
        x: np.ndarray,
        cdf: np.ndarray,
        bucket_low: float,
        bucket_high: float,
    ) -> float:
        """
        Calculate probability using the KDE-smoothed CDF.
        More accurate for small ensembles where raw counts can be noisy.
        """
        n = len(cdf)
        if n == 0:
            return 0.0
        
        if bucket_low == -999:
            idx = np.searchsorted(x, bucket_high)
            return float(cdf[min(idx, n - 1)])
        
        if bucket_high == 999:
            idx = np.searchsorted(x, bucket_low)
            return float(1.0 - cdf[max(min(idx, n - 1), 0)])
        
        idx_high = min(np.searchsorted(x, bucket_high), n - 1)
        idx_low = max(np.searchsorted(x, bucket_low), 0)
        
        if idx_high <= idx_low:
            return 0.0
        
        return float(cdf[idx_high] - cdf[idx_low])
    
    # ── High-Level Interface ──────────────────────────────────
    
    def get_cdf(
        self,
        lat: float,
        lon: float,
        date: str,
        ensemble: str = "ecmwf_ens",
    ) -> Optional[dict]:
        """
        Main entry point: fetch ensemble, build CDF, return ready-to-use results.
        
        Returns dict:
        {
            "date": str,
            "n_members": int,
            "mean_temp": float,
            "std_temp": float,
            "x": np.ndarray,      # temperature grid
            "cdf": np.ndarray,     # CDF values
            "members_raw": np.ndarray,  # all member temps
            "fetched_at": str,
        }
        """
        data = self.fetch_ensemble(lat, lon, ensemble)
        if not data:
            return None
        
        temps = self.get_ensemble_temps(data["members"], date)
        if temps is None or len(temps) == 0:
            return None
        
        x, cdf = self.build_cdf(temps)
        
        return {
            "date": date,
            "n_members": len(temps),
            "mean_temp": round(float(np.mean(temps)), 2),
            "std_temp": round(float(np.std(temps)), 3),
            "min_temp": round(float(np.min(temps)), 2),
            "max_temp": round(float(np.max(temps)), 2),
            "x": x,
            "cdf": cdf,
            "members_raw": temps,
            "fetched_at": data["fetched_at"],
        }
    
    def get_bucket_prob(
        self,
        lat: float,
        lon: float,
        date: str,
        bucket_low: float,
        bucket_high: float,
        ensemble: str = "ecmwf_ens",
        use_kde: bool = True,
    ) -> Optional[float]:
        """
        Get probability of temperature hitting a specific bucket.
        Uses KDE-smoothed probability for small ensembles, 
        raw counts for large ones.
        """
        result = self.get_cdf(lat, lon, date, ensemble)
        if not result:
            return None
        
        if use_kde and result["n_members"] >= 10:
            p = self.bucket_probability_kde(
                result["x"], result["cdf"], bucket_low, bucket_high
            )
        else:
            p = self.bucket_probability(
                result["members_raw"], bucket_low, bucket_high
            )
        
        return round(float(p), 6)
    
    def is_ensemble_fresh(self, lat: float, lon: float, ensemble: str = "ecmwf_ens") -> bool:
        """Check if we have fresh ensemble data for these coordinates."""
        cache_key = self._cache_key(lat, lon, ensemble)
        if cache_key not in self._cache:
            return False
        
        fetched = datetime.fromisoformat(self._cache[cache_key]["fetched_at"])
        age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        return age < MAX_ENSEMBLE_AGE_HOURS


# ────────────────────────────────────────────────────────────────
# Utility: probability → EV and Kelly
# ────────────────────────────────────────────────────────────────

def ensemble_ev(prob_true: float, market_price: float) -> float:
    """Expected value given ensemble probability and market price."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    return round(prob_true * (1.0 / market_price - 1.0) - (1.0 - prob_true), 6)


def ensemble_kelly(prob_true: float, market_price: float, fraction: float = 0.25) -> float:
    """Fractional Kelly bet size given ensemble probability."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = 1.0 / market_price - 1.0
    f = (prob_true * b - (1.0 - prob_true)) / b
    return round(min(max(0.0, f) * fraction, 1.0), 6)


def ensemble_sharpe(prob_true: float, market_price: float) -> float:
    """
    Approximate Sharpe ratio for a binary bet.
    Sharpe = (expected_return) / (std_dev_of_return)
    For binary: expected = p*(1/p-1) - (1-p), var = p*(1-p)/p²
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    p = prob_true
    expected = p / market_price - 1.0
    variance = p * (1 - p) / (market_price ** 2)
    if variance <= 0:
        return 0.0
    return round(expected / math.sqrt(variance), 4)
