#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial.py — Spatial Correlation Penalty for Weather Bets
============================================================
Prevents over-concentration by detecting when multiple positions are 
exposed to the same synoptic weather system. Dynamically scales down 
Kelly fractions for correlated bets.

Mathematical Formula:
──────────────────────
For a new bet on city_new, the correlation penalty is:

    penalty = 1.0 - Σ(correlation_weight(city_i) for city_i in open_positions)

where correlation_weight(city_i) is computed from:

    1. DISTANCE_FACTOR:  exp(-d² / (2 * σ_d²))  
       where d = distance in km, σ_d = 800 km (synoptic scale)
       → two cities 400 km apart: factor = exp(-0.125) ≈ 0.88
       → two cities 1500 km apart: factor = exp(-1.76) ≈ 0.17

    2. WIND_ALIGNMENT_FACTOR:  dot(wind_new, wind_open) clamped to [0, 1]
       → same wind direction = 1.0 (same air mass)
       → opposite wind direction = 0.0 (different systems)

    3. PRESSURE_SIMILARITY:  1.0 - |p_new - p_open| / 20.0, clamped to [0, 1]
       → within 5 hPa = 0.75 correlation
       → within 20 hPa+ = 0.0 correlation

    correlation_weight = DISTANCE × 0.4 + WIND × 0.3 + PRESSURE × 0.3

The adjusted Kelly fraction:

    kelly_adjusted = kelly * max(0.3, penalty)

Floor at 0.3 so we never completely zero out — just heavily discount.

Synoptic Scale Justification:
──────────────────────────────
A cold front in North America can span 2000-3000 km. Betting "under" on 
Chicago, NYC, and Atlanta simultaneously means 3 bets on ONE cold front — 
not 3 independent edges. This penalty recognizes that clustered bets share 
the same underlying weather risk.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests


# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

SYNOPTIC_SIGMA_KM = 800.0   # Correlation decay distance (km)
PRESSURE_RANGE_HPA = 20.0   # 20 hPa range for pressure similarity
MIN_CORRELATION_FLOOR = 0.30  # Floor for adjusted Kelly fraction

# Simple wind classification by region (approximate prevailing patterns)
REGION_WINDS = {
    "us_northeast":   (270, 10),   # Westerly, 10 kt typical
    "us_midwest":     (270, 10),
    "us_southeast":   (225, 8),    # SW, 8 kt
    "us_west":        (315, 8),    # NW, 8 kt
    "eu_west":        (270, 15),   # Westerly, strong
    "eu_central":     (270, 10),
    "eu_east":        (225, 8),
    "asia_east":      (180, 8),    # Southerly monsoon influence
    "asia_se":        (180, 8),
    "asia_south":     (225, 5),
    "sa_east":        (90, 10),    # Easterly trades
    "oc":             (270, 15),   # Strong westerlies
    "ca_east":        (270, 12),
}


# ────────────────────────────────────────────────────────────────
# Data types
# ────────────────────────────────────────────────────────────────

@dataclass
class CityWeather:
    """Snapshot of weather conditions at a city for correlation calc."""
    lat: float
    lon: float
    wind_dir: float    # degrees (0-360)
    wind_speed: float  # knots
    pressure: float    # hPa
    temperature: float


@dataclass
class OpenPosition:
    """Minimal representation of an open position for correlation."""
    city: str
    lat: float
    lon: float
    bucket_low: float
    bucket_high: float
    shares: float
    entry_price: float


# ────────────────────────────────────────────────────────────────
# Distance calculation (Haversine)
# ────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ────────────────────────────────────────────────────────────────
# Correlation weight computation
# ────────────────────────────────────────────────────────────────

def distance_factor(distance_km: float, sigma_km: float = SYNOPTIC_SIGMA_KM) -> float:
    """
    Gaussian distance decay: closer = higher correlation.
    Two cities 400 km apart: exp(-0.125) ≈ 0.88
    Two cities 1500 km apart: exp(-1.76) ≈ 0.17
    """
    return math.exp(-(distance_km ** 2) / (2 * sigma_km ** 2))


def wind_alignment(wind_dir_a: float, wind_dir_b: float) -> float:
    """
    Cosine similarity of wind directions.
    1.0 = same direction (same air mass approach)
    0.0 = perpendicular
    -1.0 = opposite — but clamped to [0, 1] 
           (opposite winds still mean different systems, so 0)
    
    Formula: cos(θ₂ - θ₁) clamped to [0, 1]
    """
    delta = math.radians(wind_dir_b - wind_dir_a)
    return max(0.0, math.cos(delta))


def pressure_similarity(p_a: float, p_b: float, range_hpa: float = PRESSURE_RANGE_HPA) -> float:
    """
    Linear pressure similarity.
    1.0 = identical pressure (same system)
    0.0 = 20 hPa+ apart (different systems)
    
    Formula: max(0, 1 - |p₂ - p₁| / range)
    """
    return max(0.0, 1.0 - abs(p_b - p_a) / range_hpa)


def correlation_weight(
    w_a: CityWeather,
    w_b: CityWeather,
    dist_weight: float = 0.4,
    wind_weight: float = 0.3,
    press_weight: float = 0.3,
) -> float:
    """
    Composite correlation weight between two cities.
    
    Returns 0.0 (fully independent) to 1.0 (fully correlated — same system).
    
    Weights: distance 40%, wind 30%, pressure 30%.
    """
    d = haversine_km(w_a.lat, w_a.lon, w_b.lat, w_b.lon)
    
    d_factor = distance_factor(d)
    w_factor = wind_alignment(w_a.wind_dir, w_b.wind_dir)
    p_factor = pressure_similarity(w_a.pressure, w_b.pressure)
    
    return round(dist_weight * d_factor + wind_weight * w_factor + press_weight * p_factor, 4)


# ────────────────────────────────────────────────────────────────
# Wind estimation from fallback data
# ────────────────────────────────────────────────────────────────

def estimate_region_wind(lat: float, lon: float) -> Tuple[float, float]:
    """Crude wind estimate based on latitude/region. Use as fallback."""
    if lon < -30:
        if lat > 50:
            return (270, 10)  # Canada
        elif lat > 40:
            return (270, 10)  # US North
        elif lat > 30:
            return (225, 8)   # US South
        else:
            return (90, 10)   # Caribbean/tropics
    elif lon < 60:
        if lat > 50:
            return (270, 15)  # NW Europe
        elif lat > 40:
            return (270, 10)  # Central Europe
        else:
            return (225, 8)   # Mediterranean
    elif lon > 100:
        if lat > 30:
            return (270, 10)  # East Asia
        else:
            return (180, 8)   # SE Asia
    elif lon > 0:
        return (225, 5)       # Middle East/South Asia
    else:
        if lat < 0:
            return (90, 10)   # South America/SA
        else:
            return (270, 8)   # Default mid-latitude westerly


def fetch_current_weather(lat: float, lon: float) -> Optional[CityWeather]:
    """
    Get current wind and pressure from Open-Meteo for a city.
    
    Uses the current weather API, falling back to regional estimates.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=wind_speed_10m,wind_direction_10m,pressure_msl,temperature_2m"
            f"&timezone=UTC"
        )
        resp = requests.get(url, timeout=(5, 8))
        data = resp.json()
        
        current = data.get("current", {})
        
        return CityWeather(
            lat=lat,
            lon=lon,
            wind_dir=float(current.get("wind_direction_10m", 0) or 0),
            wind_speed=float(current.get("wind_speed_10m", 0) or 0),
            pressure=float(current.get("pressure_msl", 1013) or 1013),
            temperature=float(current.get("temperature_2m", 15) or 15),
        )
        
    except Exception:
        wd, ws = estimate_region_wind(lat, lon)
        return CityWeather(
            lat=lat, lon=lon,
            wind_dir=wd, wind_speed=ws,
            pressure=1013.0,
            temperature=15.0,
        )


# ────────────────────────────────────────────────────────────────
# Spatial risk manager
# ────────────────────────────────────────────────────────────────

class SpatialRiskManager:
    """
    Manages spatial correlation across the portfolio.
    
    Before opening a new position, checks correlation with all 
    open positions and adjusts the Kelly fraction.
    """
    
    def __init__(self, max_total_correlation: float = 1.5):
        """
        Args:
            max_total_correlation: if sum of correlations exceeds this,
                                   reject the new bet entirely
        """
        self.max_total_correlation = max_total_correlation
        self._weather_cache: Dict[str, CityWeather] = {}
    
    def get_city_weather(self, lat: float, lon: float, city: str) -> CityWeather:
        """Get weather for a city, with caching."""
        if city not in self._weather_cache:
            self._weather_cache[city] = fetch_current_weather(lat, lon)
        return self._weather_cache[city]
    
    def compute_penalty(
        self,
        new_city: str,
        new_lat: float,
        new_lon: float,
        open_positions: List[OpenPosition],
    ) -> Tuple[float, List[dict]]:
        """
        Compute Kelly adjustment penalty for a potential new position.
        
        Returns:
            penalty: float in [MIN_CORRELATION_FLOOR, 1.0]
                → 1.0 = fully independent, no discount
                → 0.3 = MAX correlation — floor, not zero
            details: per-position correlation breakdown
        """
        if not open_positions:
            return (1.0, [])
        
        new_wx = self.get_city_weather(new_lat, new_lon, new_city)
        
        total_corr = 0.0
        details = []
        
        for pos in open_positions:
            pos_wx = self.get_city_weather(pos.lat, pos.lon, pos.city)
            corr = correlation_weight(new_wx, pos_wx)
            
            details.append({
                "city": pos.city,
                "distance_km": round(haversine_km(new_lat, new_lon, pos.lat, pos.lon), 0),
                "wind_alignment": round(wind_alignment(new_wx.wind_dir, pos_wx.wind_dir), 3),
                "pressure_similarity": round(pressure_similarity(new_wx.pressure, pos_wx.pressure), 3),
                "correlation": corr,
            })
            
            total_corr += corr
        
        # Hard cap: reject if total correlation too high
        if total_corr > self.max_total_correlation:
            return (0.0, details)
        
        penalty = max(MIN_CORRELATION_FLOOR, 1.0 - total_corr)
        return (round(penalty, 4), details)
    
    def adjusted_kelly(
        self,
        base_kelly: float,
        penalty: float,
    ) -> float:
        """
        Apply spatial correlation penalty to Kelly fraction.
        
        Formula: kelly_adj = base_kelly * penalty
        Where penalty = max(0.3, 1.0 - Σ(correlation_weights))
        
        Example:
          NYC alone: kelly=0.25, penalty=1.0 → kelly_adj=0.25
          NYC + Chicago (400km): penalty≈0.85 → kelly_adj≈0.21
          NYC + Chicago + Atlanta (clustered): penalty≈0.4 → kelly_adj≈0.10
          NYC + London (6000km): penalty≈1.0 → kelly_adj=0.25 (independent)
        """
        return round(base_kelly * penalty, 6)
    
    def invalidate_cache(self):
        """Clear weather cache to force re-fetch on next use."""
        self._weather_cache.clear()


# ────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # NYC and Chicago — correlated (1200 km, same westerly flow)
    nyc = fetch_current_weather(40.7772, -73.8726)
    chi = fetch_current_weather(41.9742, -87.9073)
    
    corr = correlation_weight(nyc, chi)
    d = haversine_km(nyc.lat, nyc.lon, chi.lat, chi.lon)
    
    print(f"NYC ↔ Chicago: {d:.0f} km, correlation={corr:.3f}")
    print(f"  Distance factor:  {distance_factor(d):.3f}")
    print(f"  Wind alignment:   {wind_alignment(nyc.wind_dir, chi.wind_dir):.3f}")
    print(f"  Pressure sim:     {pressure_similarity(nyc.pressure, chi.pressure):.3f}")
    print(f"\nKelly penalty: {max(0.3, 1.0 - corr):.3f}")
    print(f"  0.25 Kelly → {0.25 * max(0.3, 1.0 - corr):.3f} adjusted")
