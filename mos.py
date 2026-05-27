#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mos.py — Model Output Statistics (MOS) Downscaling Pipeline
==============================================================
Corrects systematic biases in global weather models for specific
airport locations. Trains XGBoost models to predict the adjustment 
from raw grid forecast → actual ASOS observation.

Architecture:
  Inputs:  ECMWF ensemble mean, time of year (sin/cos encoding),
           wind speed, wind direction (sin/cos), elevation delta
  Target:  Actual METAR observed temperature at the station
  Output:  Bias-corrected forecast (°F or °C)

Training requires: historical ECMWF forecasts + METAR observations.
Minimum ~90 days of data for initial training.

Directory structure:
  data/mos/
    models/           ← trained model files (*.json, *.pkl)
    training_data/    ← historical pairs (forecast, actual)
    by_station/       ← per-station MOS models
      KLGA.json
      KORD.json
      ...

Usage:
    from mos import MOSCorrecter
    correcter = MOSCorrecter()
    corrected_temp = correcter.correct("KLGA", raw_ecmwf=82.3, month=5, wind_speed=10)
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

class MOSConfig:
    """Configuration for MOS training and inference."""
    
    # Minimum training samples before model is considered usable
    min_samples: int = 90  # ~3 months of daily data
    
    # Features used for prediction
    features: List[str] = [
        "raw_forecast",       # ECMWF ensemble mean temperature
        "sin_month",          # sin(2π * month / 12) — seasonal cycle
        "cos_month",          # cos(2π * month / 12)
        "wind_speed_kt",      # surface wind speed
        "sin_wind_dir",       # sin(wind direction in radians)
        "cos_wind_dir",       # cos(wind direction in radians)
        "hour_of_day",        # 0-23 UTC hour
        "day_of_year_sin",    # finner seasonal cycle
        "day_of_year_cos",
    ]
    
    # Target
    target: str = "actual_temp"
    
    # XGBoost hyperparameters (conservative defaults)
    xgb_params: dict = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }
    
    # Whether to use ensemble spread as an additional feature
    use_ensemble_spread: bool = True
    
    # Whether to weight recent data more heavily
    recency_weighting: bool = True
    recency_half_life_days: float = 90.0  # exponential decay half-life


# ────────────────────────────────────────────────────────────────
# Feature engineering
# ────────────────────────────────────────────────────────────────

def encode_time_features(
    month: int,
    day_of_year: Optional[int] = None,
    hour: int = 12,  # default noon UTC
) -> Dict[str, float]:
    """Create cyclical time encodings for the model."""
    if day_of_year is None:
        day_of_year = month * 30  # rough approximation
    
    return {
        "sin_month": math.sin(2 * math.pi * month / 12),
        "cos_month": math.cos(2 * math.pi * month / 12),
        "hour_of_day": float(hour),
        "day_of_year_sin": math.sin(2 * math.pi * day_of_year / 365),
        "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 365),
    }


def encode_wind_features(
    wind_dir_deg: float,
    wind_speed_kt: float,
) -> Dict[str, float]:
    """Encode wind direction cyclically and speed linearly."""
    wind_rad = math.radians(wind_dir_deg)
    return {
        "sin_wind_dir": math.sin(wind_rad),
        "cos_wind_dir": math.cos(wind_rad),
        "wind_speed_kt": wind_speed_kt,
    }


def build_feature_vector(
    raw_forecast: float,
    month: int,
    wind_dir_deg: float = 0.0,
    wind_speed_kt: float = 0.0,
    hour: int = 12,
    day_of_year: Optional[int] = None,
    ensemble_spread: Optional[float] = None,
) -> np.ndarray:
    """
    Build the feature vector for a single prediction.
    Order must match MOSConfig.features.
    """
    time_feats = encode_time_features(month, day_of_year, hour)
    wind_feats = encode_wind_features(wind_dir_deg, wind_speed_kt)
    
    feats = [
        raw_forecast,
        time_feats["sin_month"],
        time_feats["cos_month"],
        wind_feats["wind_speed_kt"],
        wind_feats["sin_wind_dir"],
        wind_feats["cos_wind_dir"],
        time_feats["hour_of_day"],
        time_feats["day_of_year_sin"],
        time_feats["day_of_year_cos"],
    ]
    
    if ensemble_spread is not None:
        feats.append(ensemble_spread)
    
    return np.array(feats, dtype=np.float32)


# ────────────────────────────────────────────────────────────────
# MOS Model
# ────────────────────────────────────────────────────────────────

class MOSModel:
    """
    A single station's MOS bias-correction model.
    
    Trains on historical pairs: (raw_forecast_features → actual_observation)
    Predicts the bias-corrected temperature for new forecasts.
    """
    
    def __init__(self, station: str, config: Optional[MOSConfig] = None):
        self.station = station
        self.config = config or MOSConfig()
        self.model = None
        self.bias_mean = 0.0
        self.bias_std = 0.0
        self.n_samples = 0
        self.last_trained = None
        self.mae = None  # mean absolute error on validation
    
    def train(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Train an XGBoost model on historical data.
        
        Args:
            X: feature matrix (n_samples, n_features)
            y: target vector (n_samples,) — actual observed temperatures
            weights: optional sample weights (recency weighting)
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print(f"  [MOS] XGBoost not installed — using baseline bias correction")
            self._train_baseline(X, y)
            return
        
        self.n_samples = len(y)
        
        # Compute bias for fallback
        raw_forecasts = X[:, 0] if X.shape[1] > 0 else np.zeros(len(y))
        self.bias_mean = float(np.mean(y - raw_forecasts))
        self.bias_std = float(np.std(y - raw_forecasts))
        
        if self.n_samples < self.config.min_samples:
            print(f"  [MOS] {self.station}: {self.n_samples} samples < {self.config.min_samples} min — baseline only")
            return
        
        # Train XGBoost
        params = dict(self.config.xgb_params)
        
        self.model = XGBRegressor(**params)
        self.model.fit(X, y, sample_weight=weights)
        
        self.last_trained = datetime.now(timezone.utc).isoformat()
        
        # Quick validation
        y_pred = self.model.predict(X)
        self.mae = float(np.mean(np.abs(y - y_pred)))
        
        print(f"  [MOS] {self.station}: trained on {self.n_samples} samples, MAE={self.mae:.2f}°")
    
    def _train_baseline(self, X: np.ndarray, y: np.ndarray):
        """Fallback: simple mean bias correction when XGBoost unavailable."""
        self.n_samples = len(y)
        raw_forecasts = X[:, 0] if len(X.shape) > 1 and X.shape[1] > 0 else np.zeros(len(y))
        self.bias_mean = float(np.mean(y - raw_forecasts))
        self.bias_std = float(np.std(y - raw_forecasts))
        self.last_trained = datetime.now(timezone.utc).isoformat()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict bias-corrected temperatures.
        
        Returns the model prediction if XGBoost is trained,
        otherwise applies the mean bias correction.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.model is not None:
            return self.model.predict(X)
        
        # Fallback: mean bias correction
        raw_forecast = X[:, 0]
        return raw_forecast + self.bias_mean
    
    def predict_single(
        self,
        raw_forecast: float,
        month: int,
        wind_dir_deg: float = 0.0,
        wind_speed_kt: float = 0.0,
        hour: int = 12,
        ensemble_spread: Optional[float] = None,
    ) -> float:
        """
        Convenience method: predict from raw inputs.
        
        Returns:
            Corrected temperature in the same units as raw_forecast.
        """
        features = []
        if self.config.use_ensemble_spread and ensemble_spread is not None:
            features = MOSConfig.features  # includes ensemble_spread
        else:
            features = [f for f in MOSConfig.features if f != "ensemble_spread_deg"]
        
        X = build_feature_vector(
            raw_forecast, month, wind_dir_deg, wind_speed_kt,
            hour, ensemble_spread=ensemble_spread,
        )
        
        if len(features) < len(X):
            X = X[:len(features)]
        
        return float(self.predict(X.reshape(1, -1))[0])
    
    def save(self, path: Path):
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "station": self.station,
            "bias_mean": self.bias_mean,
            "bias_std": self.bias_std,
            "n_samples": self.n_samples,
            "last_trained": self.last_trained,
            "mae": self.mae,
        }
        
        # XGBoost model saved separately if trained
        if self.model is not None:
            model_path = path.with_suffix(".json")
            self.model.save_model(str(model_path))
            data["model_path"] = str(model_path)
        
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path, config: Optional[MOSConfig] = None) -> "MOSModel":
        """Load model from disk."""
        data = json.loads(path.read_text())
        
        model = cls(data["station"], config)
        model.bias_mean = data["bias_mean"]
        model.bias_std = data["bias_std"]
        model.n_samples = data["n_samples"]
        model.last_trained = data.get("last_trained")
        model.mae = data.get("mae")
        
        model_path = data.get("model_path")
        if model_path and Path(model_path).exists():
            try:
                from xgboost import XGBRegressor
                model.model = XGBRegressor()
                model.model.load_model(model_path)
            except Exception:
                pass  # XGBoost model unavailable, use baseline
        
        return model


# ────────────────────────────────────────────────────────────────
# MOS Manager (multi-station)
# ────────────────────────────────────────────────────────────────

class MOSCorrecter:
    """
    Manages MOS models for all stations.
    
    Directory structure:
      data/mos/
        by_station/
          KLGA.json     ← MOSModel save
          KORD.json
          ...
        training_data/
          KLGA_pairs.jsonl  ← historical (forecast, actual) pairs
    """
    
    def __init__(
        self,
        base_dir: Path = Path("data/mos"),
        config: Optional[MOSConfig] = None,
    ):
        self.base_dir = base_dir
        self.config = config or MOSConfig()
        self.models_dir = base_dir / "by_station"
        self.training_dir = base_dir / "training_data"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        self._models: Dict[str, MOSModel] = {}
        self._load_all()
    
    def _model_path(self, station: str) -> Path:
        return self.models_dir / f"{station}.json"
    
    def _training_path(self, station: str) -> Path:
        return self.training_dir / f"{station}_pairs.jsonl"
    
    def _load_all(self):
        """Load all trained MOS models from disk."""
        for f in self.models_dir.glob("*.json"):
            try:
                station = f.stem
                self._models[station] = MOSModel.load(f, self.config)
            except Exception:
                pass
    
    def add_training_pair(
        self,
        station: str,
        raw_forecast: float,
        actual_temp: float,
        month: int,
        wind_dir_deg: float = 0.0,
        wind_speed_kt: float = 0.0,
        hour: int = 12,
        ensemble_spread: Optional[float] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Record a (forecast → actual) pair for future training.
        Appends to the station's JSONL training file.
        """
        pair = {
            "raw_forecast": raw_forecast,
            "actual_temp": actual_temp,
            "month": month,
            "wind_dir_deg": wind_dir_deg,
            "wind_speed_kt": wind_speed_kt,
            "hour": hour,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if ensemble_spread is not None:
            pair["ensemble_spread"] = ensemble_spread
        
        tp = self._training_path(station)
        with open(tp, "a") as f:
            f.write(json.dumps(pair) + "\n")
    
    def train_station(self, station: str) -> Optional[MOSModel]:
        """
        Train a MOS model for a specific station from accumulated data.
        """
        tp = self._training_path(station)
        if not tp.exists():
            return None
        
        pairs = []
        with open(tp) as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    continue
        
        if len(pairs) < self.config.min_samples:
            print(f"  [MOS] {station}: only {len(pairs)} pairs, need {self.config.min_samples}")
            return None
        
        # Build feature matrix
        X_list = []
        y_list = []
        weights = []
        
        now = datetime.now(timezone.utc)
        
        for p in pairs:
            feats = build_feature_vector(
                p["raw_forecast"],
                p["month"],
                p.get("wind_dir_deg", 0),
                p.get("wind_speed_kt", 0),
                p.get("hour", 12),
                ensemble_spread=p.get("ensemble_spread"),
            )
            X_list.append(feats)
            y_list.append(p["actual_temp"])
            
            # Recency weighting
            if self.config.recency_weighting and p.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(p["timestamp"])
                    age_days = (now - ts).total_seconds() / 86400
                    w = math.exp(-age_days / self.config.recency_half_life_days * math.log(2))
                except Exception:
                    w = 1.0
            else:
                w = 1.0
            weights.append(w)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        
        model = MOSModel(station, self.config)
        model.train(X, y, w)
        model.save(self._model_path(station))
        
        self._models[station] = model
        return model
    
    def correct(
        self,
        station: str,
        raw_forecast: float,
        month: int,
        wind_dir_deg: float = 0.0,
        wind_speed_kt: float = 0.0,
        hour: int = 12,
        ensemble_spread: Optional[float] = None,
    ) -> float:
        """
        Apply MOS correction to a raw forecast.
        
        If no trained model exists for the station, applies a simple
        bias correction from available data if any. Otherwise returns
        the raw forecast unchanged.
        
        Returns:
            Bias-corrected temperature forecast.
        """
        model = self._models.get(station)
        
        if model and model.n_samples >= self.config.min_samples:
            return model.predict_single(
                raw_forecast, month, wind_dir_deg, wind_speed_kt,
                hour, ensemble_spread,
            )
        
        # Fallback: quick bias check from training data
        tp = self._training_path(station)
        if tp.exists():
            pairs = []
            with open(tp) as f:
                for line in f:
                    try:
                        pairs.append(json.loads(line))
                    except Exception:
                        pass
            
            if pairs:
                biases = [p["actual_temp"] - p["raw_forecast"] for p in pairs]
                mean_bias = sum(biases) / len(biases)
                return raw_forecast + mean_bias
        
        # No data — return uncorrected
        return raw_forecast
    
    def get_station_mae(self, station: str) -> Optional[float]:
        """Get the model's validation MAE for a station."""
        model = self._models.get(station)
        if model:
            return model.mae
        return None
    
    def get_trained_stations(self) -> List[str]:
        """List stations with trained models."""
        return [
            s for s, m in self._models.items()
            if m.n_samples >= self.config.min_samples
        ]
    
    def train_all(self) -> Dict[str, Optional[MOSModel]]:
        """Train models for all stations with sufficient data."""
        results = {}
        for tp in self.training_dir.glob("*_pairs.jsonl"):
            station = tp.stem.replace("_pairs", "")
            model = self.train_station(station)
            results[station] = model
            time.sleep(0.1)  # rate limit
        return results
