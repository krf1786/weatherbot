#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensor.py — ASOS/NWS Sensor Failure Monitor
==============================================
Monitors airport weather sensors for failures and maintenance.
Blacklists markets when sensors are unreliable (resolution risk).

Polymarket resolves based on specific ASOS sensors. If the designated
sensor breaks, the resolution can switch to a backup station 20+ miles 
away — different microclimate, different temperature reading, 
potentially invalidating the bet.

Data sources:
  - METAR data freshness from aviationweather.gov
  - NWS ASOS maintenance status (scraped)
  - Temperature reporting consistency checks

Usage:
    from sensor import SensorMonitor
    monitor = SensorMonitor()
    is_ok = monitor.is_station_healthy("KLGA")
    if not is_ok:
        print(f"KLGA blacklisted: {monitor.get_blacklist_reason('KLGA')}")
"""

import time
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple

import requests


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

# Stations known to be unreliable / under maintenance
KNOWN_BAD_STATIONS: Dict[str, str] = {
    # station → reason
    # "KXYZ": "ASOS temp sensor under maintenance June 2026",
}

# Maximum age of METAR observation before station is considered stale
MAX_METAR_AGE_MINUTES = 120  # 2 hours

# Temperature sanity bounds (°C) — outside these = sensor fault
TEMP_LOW_BOUND = -80.0   # Colder than Antarctica
TEMP_HIGH_BOUND = 60.0   # Hotter than Death Valley record

# How often to refresh station status (minutes)
REFRESH_INTERVAL_MINUTES = 60

# NWS ASOS maintenance page
NWS_ASOS_URL = "https://www.weather.gov/asos/asosstatus"


# ────────────────────────────────────────────────────────────────
# Core Monitor
# ────────────────────────────────────────────────────────────────

class SensorMonitor:
    """
    Monitors ASOS/AWOS sensor health for Polymarket resolution stations.
    Maintains a blacklist of unreliable stations.
    """
    
    def __init__(
        self,
        blacklist_file: Path = Path("data/sensor_blacklist.json"),
        cache_dir: Path = Path("data/sensor_cache"),
    ):
        self.blacklist_file = blacklist_file
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._blacklist: Dict[str, dict] = {}
        self._last_refresh: Dict[str, datetime] = {}
        self._load_blacklist()
    
    def _load_blacklist(self):
        """Load persisted blacklist from disk."""
        if self.blacklist_file.exists():
            try:
                self._blacklist = json.loads(self.blacklist_file.read_text())
            except Exception:
                self._blacklist = {}
    
    def _save_blacklist(self):
        """Persist blacklist to disk."""
        self.blacklist_file.write_text(json.dumps(self._blacklist, indent=2))
    
    def _cache_path(self, station: str) -> Path:
        return self.cache_dir / f"{station}_status.json"
    
    # ── METAR Freshness Check ──────────────────────────────────
    
    def check_metar_freshness(self, station: str) -> Tuple[bool, dict]:
        """
        Check if the station is actively reporting METAR data.
        
        Returns: (is_reporting, details)
        """
        details = {
            "station": station,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "reporting": False,
            "age_minutes": None,
            "temp_ok": False,
            "raw_temp_c": None,
            "warnings": [],
        }
        
        try:
            url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json"
            resp = requests.get(url, timeout=(5, 10))
            
            if resp.status_code != 200:
                details["warnings"].append(f"http_{resp.status_code}")
                return (False, details)
            
            data = resp.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                details["warnings"].append("no_data_returned")
                return (False, details)
            
            metar = data[0]
            
            # Check observation time freshness
            obs_time = metar.get("obsTime", "")
            if obs_time:
                obs_dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - obs_dt).total_seconds() / 60
                details["age_minutes"] = round(age, 1)
                
                if age > MAX_METAR_AGE_MINUTES:
                    details["warnings"].append(f"metar_stale_{age:.0f}min")
                    return (False, details)
                
                details["reporting"] = True
            
            # Check temperature validity
            temp_c = metar.get("temp")
            if temp_c is None:
                details["warnings"].append("temp_missing")
                return (False, details)
            
            temp_c = float(temp_c)
            details["raw_temp_c"] = temp_c
            
            if temp_c < TEMP_LOW_BOUND or temp_c > TEMP_HIGH_BOUND:
                details["warnings"].append(f"temp_out_of_bounds_{temp_c}")
                details["temp_ok"] = False
                return (False, details)
            
            details["temp_ok"] = True
            return (True, details)
            
        except Exception as e:
            details["warnings"].append(f"check_failed: {str(e)[:80]}")
            return (False, details)
    
    # ── NWS ASOS Status Scrape ─────────────────────────────────
    
    def check_nws_maintenance(self, station: str) -> Tuple[bool, str]:
        """
        Scrape NWS ASOS status page for station maintenance flags.
        
        Returns: (is_ok, message)
        """
        cache_path = self.cache_path(f"nws_{station}")
        
        # Check cache
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            age = (datetime.now(timezone.utc) - 
                   datetime.fromisoformat(cached["checked_at"])).total_seconds() / 3600
            if age < 24:  # NWS status changes daily at most
                return cached["is_ok"], cached["message"]
        
        try:
            resp = requests.get(NWS_ASOS_URL, timeout=(10, 20))
            if resp.status_code != 200:
                return (True, "nws_page_unreachable")
            
            text = resp.text.lower()
            
            # Search for station-specific maintenance notices
            station_lower = station.lower()
            
            # Maintenance keywords
            maintenance_patterns = [
                rf"{station_lower}.*?(?:maintenance|inoperative|not reporting|out of service|degraded|failed|repair)",
                rf"(?:maintenance|inoperative|not reporting).*?{station_lower}",
            ]
            
            for pattern in maintenance_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    result = (False, f"nws_maintenance_flagged")
                    cache_path.write_text(json.dumps({
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "is_ok": False,
                        "message": "nws_maintenance_flagged",
                    }))
                    return result
            
            # No issues found
            result = (True, "no_nws_issues")
            cache_path.write_text(json.dumps({
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "is_ok": True,
                "message": "no_nws_issues",
            }))
            return result
            
        except Exception as e:
            return (True, f"nws_check_unavailable: {str(e)[:50]}")
    
    # ── Temperature Consistency ────────────────────────────────
    
    def check_temp_consistency(
        self, 
        station: str, 
        current_temp_c: float,
        ensemble_mean_c: float,
        max_deviation_c: float = 15.0,
    ) -> Tuple[bool, str]:
        """
        Check if current temperature is consistent with ensemble forecast.
        
        A large deviation between METAR observation and ensemble mean 
        can indicate a sensor issue (or a truly extreme event — 
        check manually before blacklisting).
        
        Returns: (is_consistent, message)
        """
        deviation = abs(current_temp_c - ensemble_mean_c)
        
        if deviation > max_deviation_c:
            return (False, f"temp_deviation_{deviation:.1f}C_vs_ensemble_{ensemble_mean_c:.1f}C")
        
        return (True, f"consistent_{deviation:.1f}C_deviation")
    
    # ── Full Health Check ──────────────────────────────────────
    
    def is_station_healthy(
        self,
        station: str,
        ensemble_mean_c: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Full health check for a station.
        
        Runs: METAR freshness → NWS maintenance → temperature sanity.
        Caches result for REFRESH_INTERVAL_MINUTES.
        
        Returns: (is_healthy, list_of_warnings)
        """
        # Check known blacklist first
        if station in self._blacklist:
            bl = self._blacklist[station]
            # Check if blacklist has expired
            if bl.get("expires_at"):
                expires = datetime.fromisoformat(bl["expires_at"])
                if datetime.now(timezone.utc) > expires:
                    self._blacklist.pop(station, None)
                    self._save_blacklist()
                else:
                    return (False, [f"blacklisted: {bl.get('reason', 'unknown')}"])
            else:
                return (False, [f"blacklisted: {bl.get('reason', 'unknown')}"])
        
        # Check cache freshness
        if station in self._last_refresh:
            age = (datetime.now(timezone.utc) - self._last_refresh[station]).total_seconds() / 60
            if age < REFRESH_INTERVAL_MINUTES:
                return (True, [])  # Assume cached pass was fine
        
        warnings = []
        is_ok = True
        
        # 1. METAR freshness
        reporting, metar_details = self.check_metar_freshness(station)
        if not reporting:
            is_ok = False
            warnings.extend(metar_details.get("warnings", []))
        
        # 2. NWS maintenance check (only if METAR passed)
        if is_ok:
            nws_ok, nws_msg = self.check_nws_maintenance(station)
            if not nws_ok:
                is_ok = False
                warnings.append(nws_msg)
        
        # 3. Temperature consistency (optional)
        if is_ok and ensemble_mean_c is not None and metar_details.get("raw_temp_c"):
            temp_ok, temp_msg = self.check_temp_consistency(
                station, 
                metar_details["raw_temp_c"],
                ensemble_mean_c,
            )
            if not temp_ok:
                warnings.append(temp_msg)
                # Don't blacklist on temp deviation alone — could be real extreme weather
        
        # Update last refresh
        self._last_refresh[station] = datetime.now(timezone.utc)
        
        return (is_ok, warnings)
    
    def blacklist_station(
        self,
        station: str,
        reason: str,
        duration_hours: int = 24,
    ):
        """Add a station to the blacklist."""
        expires = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        self._blacklist[station] = {
            "reason": reason,
            "blacklisted_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires.isoformat(),
        }
        self._save_blacklist()
    
    def unblacklist_station(self, station: str):
        """Remove a station from the blacklist."""
        self._blacklist.pop(station, None)
        self._save_blacklist()
    
    def get_blacklist_reason(self, station: str) -> Optional[str]:
        """Get why a station is blacklisted, or None."""
        bl = self._blacklist.get(station)
        if bl:
            return bl.get("reason")
        return None
    
    def is_blacklisted(self, station: str) -> bool:
        """Quick check if station is currently blacklisted."""
        if station not in self._blacklist:
            return False
        
        bl = self._blacklist[station]
        if bl.get("expires_at"):
            expires = datetime.fromisoformat(bl["expires_at"])
            if datetime.now(timezone.utc) > expires:
                self._blacklist.pop(station, None)
                self._save_blacklist()
                return False
        
        return True
    
    def get_blacklist(self) -> Dict[str, dict]:
        """Get entire blacklist (for dashboard display)."""
        return dict(self._blacklist)
    
    # ── Batch Check ────────────────────────────────────────────
    
    def check_market_stations(
        self,
        stations: Dict[str, Optional[float]],  # station → ensemble_mean_c
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Check multiple stations at once.
        Returns {station: (is_healthy, warnings)} for each.
        """
        results = {}
        for station, ensemble_mean in stations.items():
            results[station] = self.is_station_healthy(station, ensemble_mean)
        return results


# ────────────────────────────────────────────────────────────────
# Station → backup station mapping (for NWS resolution rules)
# ────────────────────────────────────────────────────────────────

# When primary ASOS fails, markets often resolve against the backup
STATION_BACKUPS: Dict[str, str] = {
    "KLGA": "KJFK",   # LaGuardia → JFK (12 miles)
    "KJFK": "KLGA",
    "KEWR": "KEWR",   # Newark stays Newark (separate market)
    "KORD": "KMDW",   # O'Hare → Midway (15 miles)
    "KMDW": "KORD",
    "KATL": "KPDK",   # Hartsfield-Jackson → DeKalb-Peachtree
    "KMIA": "KOPF",   # Miami Intl → Opa-locka
    "KSEA": "KBFI",   # SeaTac → Boeing Field
    "KDAL": "KDFW",   # Love Field → DFW
    "KSFO": "KOAK",   # SFO → Oakland
    "KLAX": "KBUR",   # LAX → Burbank
}

def get_backup_station(station: str) -> Optional[str]:
    """Get the backup station for a given airport. Returns None if unknown."""
    return STATION_BACKUPS.get(station.upper())
