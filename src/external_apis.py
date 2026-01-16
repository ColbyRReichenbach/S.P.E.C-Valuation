"""
S.P.E.C. Valuation Engine - External API Integrations
======================================================
Integrations with external data providers for enhanced features.
V3.0: Walk Score API for walkability/transit scores.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache

import requests
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


# ====================================
# WALK SCORE API
# ====================================
WALKSCORE_API_URL = "https://api.walkscore.com/score"

# Cache timeout (24 hours in seconds)
WALKSCORE_CACHE_TIMEOUT = 86400

# Rate limiting (Walk Score allows 5,000 calls/day on free tier)
WALKSCORE_RATE_LIMIT_DELAY = 0.2  # 200ms between calls


def get_walkscore_api_key() -> Optional[str]:
    """
    Get Walk Score API key from environment.
    
    Free registration at: https://www.walkscore.com/professional/api-sign-up.php
    
    Returns:
        API key string or None if not configured.
    """
    api_key = os.getenv("WALKSCORE_API_KEY")
    if not api_key:
        logger.warning(
            "WALKSCORE_API_KEY not set. Walk Score features will be unavailable. "
            "Get a free key at: https://www.walkscore.com/professional/api-sign-up.php"
        )
    return api_key


def fetch_walkscore(
    lat: float,
    lon: float,
    address: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Optional[int]]:
    """
    Fetch Walk Score, Transit Score, and Bike Score for a location.
    
    Walk Score measures:
    - Walk Score (0-100): How walkable is this location?
    - Transit Score (0-100): How good is public transit access?
    - Bike Score (0-100): How bikeable is this location?
    
    Args:
        lat: Latitude of the property.
        lon: Longitude of the property.
        address: Optional street address (improves accuracy).
        api_key: Walk Score API key (defaults to env var).
    
    Returns:
        Dict with 'walk_score', 'transit_score', 'bike_score' (or None if unavailable).
    """
    if api_key is None:
        api_key = get_walkscore_api_key()
    
    if api_key is None:
        return {"walk_score": None, "transit_score": None, "bike_score": None}
    
    # Build request parameters
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "transit": 1,  # Include transit score
        "bike": 1,     # Include bike score
        "wsapikey": api_key,
    }
    
    if address:
        params["address"] = address
    
    try:
        # Rate limiting
        time.sleep(WALKSCORE_RATE_LIMIT_DELAY)
        
        response = requests.get(WALKSCORE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if data.get("status") != 1:
            status_messages = {
                2: "Score being calculated, try again later",
                30: "Invalid latitude/longitude",
                31: "Walk Score not available for this location",
                40: "Invalid API key",
                41: "API quota exceeded",
                42: "IP address blocked",
            }
            status = data.get("status", 0)
            logger.warning(f"Walk Score API error: {status_messages.get(status, f'Unknown status {status}')}")
            return {"walk_score": None, "transit_score": None, "bike_score": None}
        
        # Extract scores
        result = {
            "walk_score": data.get("walkscore"),
            "transit_score": data.get("transit", {}).get("score") if isinstance(data.get("transit"), dict) else None,
            "bike_score": data.get("bike", {}).get("score") if isinstance(data.get("bike"), dict) else None,
        }
        
        logger.debug(f"Walk Score for ({lat:.4f}, {lon:.4f}): {result}")
        return result
        
    except requests.exceptions.Timeout:
        logger.warning(f"Walk Score API timeout for ({lat:.4f}, {lon:.4f})")
        return {"walk_score": None, "transit_score": None, "bike_score": None}
    except requests.exceptions.RequestException as e:
        logger.error(f"Walk Score API error: {e}")
        return {"walk_score": None, "transit_score": None, "bike_score": None}
    except Exception as e:
        logger.error(f"Unexpected error fetching Walk Score: {e}")
        return {"walk_score": None, "transit_score": None, "bike_score": None}


def add_walkscore_features(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    address_col: Optional[str] = None,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Add Walk Score features to a DataFrame of properties.
    
    Uses caching to avoid redundant API calls for the same locations.
    
    Args:
        df: DataFrame with latitude/longitude columns.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        address_col: Optional address column for better accuracy.
        cache_path: Path to cache file (default: data/processed/walkscore_cache.csv).
    
    Returns:
        DataFrame with added walk_score, transit_score, bike_score columns.
    """
    from config.settings import PROCESSED_DATA_DIR
    
    df = df.copy()
    
    # Check for API key first
    api_key = get_walkscore_api_key()
    if api_key is None:
        logger.warning("No Walk Score API key. Adding placeholder values.")
        df["walk_score"] = None
        df["transit_score"] = None
        df["bike_score"] = None
        return df
    
    # Set up cache
    if cache_path is None:
        cache_path = PROCESSED_DATA_DIR / "walkscore_cache.csv"
    
    # Load existing cache
    cache = {}
    if cache_path.exists():
        try:
            cache_df = pd.read_csv(cache_path)
            for _, row in cache_df.iterrows():
                key = (round(row["lat"], 4), round(row["lon"], 4))
                cache[key] = {
                    "walk_score": row.get("walk_score"),
                    "transit_score": row.get("transit_score"),
                    "bike_score": row.get("bike_score"),
                }
            logger.info(f"Loaded {len(cache)} cached Walk Scores")
        except Exception as e:
            logger.warning(f"Could not load Walk Score cache: {e}")
    
    # Fetch scores for each property
    walk_scores = []
    transit_scores = []
    bike_scores = []
    new_cache_entries = []
    
    total = len(df)
    cached_hits = 0
    api_calls = 0
    
    logger.info(f"Fetching Walk Scores for {total} properties...")
    
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        if pd.isna(lat) or pd.isna(lon):
            walk_scores.append(None)
            transit_scores.append(None)
            bike_scores.append(None)
            continue
        
        # Check cache
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in cache:
            cached = cache[cache_key]
            walk_scores.append(cached["walk_score"])
            transit_scores.append(cached["transit_score"])
            bike_scores.append(cached["bike_score"])
            cached_hits += 1
            continue
        
        # Fetch from API
        address = row[address_col] if address_col and address_col in row.index else None
        scores = fetch_walkscore(lat, lon, address=address, api_key=api_key)
        
        walk_scores.append(scores["walk_score"])
        transit_scores.append(scores["transit_score"])
        bike_scores.append(scores["bike_score"])
        
        # Add to cache
        cache[cache_key] = scores
        new_cache_entries.append({
            "lat": lat,
            "lon": lon,
            "walk_score": scores["walk_score"],
            "transit_score": scores["transit_score"],
            "bike_score": scores["bike_score"],
        })
        
        api_calls += 1
        
        # Progress logging
        if api_calls % 50 == 0:
            logger.info(f"  Fetched {api_calls} scores ({cached_hits} from cache)...")
    
    # Add columns to DataFrame
    df["walk_score"] = walk_scores
    df["transit_score"] = transit_scores
    df["bike_score"] = bike_scores
    
    # Save updated cache
    if new_cache_entries:
        try:
            new_cache_df = pd.DataFrame(new_cache_entries)
            if cache_path.exists():
                existing = pd.read_csv(cache_path)
                combined = pd.concat([existing, new_cache_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["lat", "lon"], keep="last")
            else:
                combined = new_cache_df
            combined.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(new_cache_entries)} new entries to Walk Score cache")
        except Exception as e:
            logger.warning(f"Could not save Walk Score cache: {e}")
    
    # Summary
    non_null = df["walk_score"].notna().sum()
    logger.info(f"Walk Score Results:")
    logger.info(f"  API calls: {api_calls}")
    logger.info(f"  Cache hits: {cached_hits}")
    logger.info(f"  Properties with scores: {non_null}/{total}")
    
    if non_null > 0:
        logger.info(f"  Mean Walk Score: {df['walk_score'].mean():.1f}")
        logger.info(f"  Mean Transit Score: {df['transit_score'].mean():.1f}")
        logger.info(f"  Mean Bike Score: {df['bike_score'].mean():.1f}")
    
    return df


def simulate_walkscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simulated Walk Score features when API is unavailable.
    
    Uses distance-based heuristics (closer to downtown = higher walkability).
    This is for development/testing when no API key is available.
    
    Args:
        df: DataFrame with distance_to_downtown_km column.
    
    Returns:
        DataFrame with simulated walk_score, transit_score, bike_score.
    """
    import numpy as np
    
    df = df.copy()
    
    logger.info("Generating simulated Walk Score features (no API key available)")
    
    # Base walkability on distance to downtown
    # Closer = more walkable (typical for urban areas)
    if "distance_to_downtown_km" in df.columns:
        # Inverse relationship: closer = higher score
        # Score = 100 - (distance * 8), clamped to 20-95
        base_score = 100 - (df["distance_to_downtown_km"] * 8)
        base_score = base_score.clip(20, 95)
        
        # Add some random variation (+/- 10 points)
        np.random.seed(42)
        noise = np.random.normal(0, 5, len(df))
        
        df["walk_score"] = (base_score + noise).clip(10, 100).round().astype(int)
        df["transit_score"] = (base_score + noise * 0.8 - 5).clip(0, 100).round().astype(int)
        df["bike_score"] = (base_score + noise * 1.2 + 5).clip(10, 100).round().astype(int)
    else:
        # Fallback: use SF average scores
        df["walk_score"] = 85  # SF is very walkable
        df["transit_score"] = 80  # Good transit
        df["bike_score"] = 75  # Bike-friendly
    
    logger.info(f"Simulated Walk Scores:")
    logger.info(f"  Mean Walk Score: {df['walk_score'].mean():.1f}")
    logger.info(f"  Mean Transit Score: {df['transit_score'].mean():.1f}")
    logger.info(f"  Mean Bike Score: {df['bike_score'].mean():.1f}")
    
    return df


def ensure_walkscore_features(
    df: pd.DataFrame,
    use_simulation_fallback: bool = True,
) -> pd.DataFrame:
    """
    Ensure Walk Score features exist in DataFrame.
    
    Tries API first, falls back to simulation if unavailable.
    
    Args:
        df: DataFrame with property data.
        use_simulation_fallback: If True, use simulated scores when API unavailable.
    
    Returns:
        DataFrame with walk_score, transit_score, bike_score columns.
    """
    # Check if features already exist
    if all(col in df.columns for col in ["walk_score", "transit_score", "bike_score"]):
        if df["walk_score"].notna().any():
            logger.info("Walk Score features already present")
            return df
    
    # Try API first
    api_key = get_walkscore_api_key()
    if api_key:
        return add_walkscore_features(df)
    
    # Fall back to simulation
    if use_simulation_fallback:
        return simulate_walkscore_features(df)
    
    # Just add None columns
    df = df.copy()
    df["walk_score"] = None
    df["transit_score"] = None
    df["bike_score"] = None
    return df
