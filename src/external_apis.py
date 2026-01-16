"""
S.P.E.C. Valuation Engine - External API Integrations
======================================================
Integrations with external data providers for enhanced features.
V3.0: Walk Score API for walkability/transit scores.

IMPORTANT: Walk Score API Compliance
- Free tier: 5,000 calls/day
- Rate limit: ~5 requests/second recommended
- Caching/storing scores: REQUIRES Walk Score Professional subscription
- Branding: Must display Walk Score logo and link back to ws_link

Terms: https://www.walkscore.com/tile-terms-of-use.shtml
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import json

import requests
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# ====================================
# WALK SCORE API CONFIGURATION
# ====================================
WALKSCORE_API_URL = "https://api.walkscore.com/score"

# Rate limiting: Walk Score recommends not hitting their server too hard
# 200ms between calls = max 5 req/sec = 300 req/min = 18,000 req/hour
# This is well under the 5,000/day limit for reasonable batch sizes
WALKSCORE_RATE_LIMIT_DELAY = 0.2  # seconds between API calls

# Daily limit tracking
WALKSCORE_DAILY_LIMIT = 5000  # Free tier limit
WALKSCORE_DAILY_LIMIT_BUFFER = 100  # Stop early to be safe


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


def _get_daily_call_count() -> int:
    """
    Get the number of Walk Score API calls made today.
    
    Tracks calls to prevent exceeding daily limit.
    """
    from config.settings import PROCESSED_DATA_DIR
    
    count_file = PROCESSED_DATA_DIR / "walkscore_daily_count.json"
    
    if not count_file.exists():
        return 0
    
    try:
        with open(count_file, "r") as f:
            data = json.load(f)
        
        # Check if count is from today
        if data.get("date") == date.today().isoformat():
            return data.get("count", 0)
        else:
            # New day, reset count
            return 0
    except Exception:
        return 0


def _increment_daily_call_count() -> int:
    """
    Increment and return the daily API call count.
    """
    from config.settings import PROCESSED_DATA_DIR
    
    count_file = PROCESSED_DATA_DIR / "walkscore_daily_count.json"
    current_count = _get_daily_call_count()
    new_count = current_count + 1
    
    try:
        with open(count_file, "w") as f:
            json.dump({
                "date": date.today().isoformat(),
                "count": new_count,
            }, f)
    except Exception as e:
        logger.warning(f"Could not save daily call count: {e}")
    
    return new_count


def _check_daily_limit() -> bool:
    """
    Check if we're approaching the daily API limit.
    
    Returns:
        True if we can make more calls, False if at limit.
    """
    current = _get_daily_call_count()
    remaining = WALKSCORE_DAILY_LIMIT - WALKSCORE_DAILY_LIMIT_BUFFER - current
    
    if remaining <= 0:
        logger.warning(
            f"Walk Score daily limit approaching ({current} calls today). "
            f"Limit: {WALKSCORE_DAILY_LIMIT}. Wait until tomorrow or upgrade to Professional."
        )
        return False
    
    return True


def fetch_walkscore(
    lat: float,
    lon: float,
    address: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Optional[any]]:
    """
    Fetch Walk Score, Transit Score, and Bike Score for a location.
    
    Walk Score measures:
    - Walk Score (0-100): How walkable is this location?
    - Transit Score (0-100): How good is public transit access?
    - Bike Score (0-100): How bikeable is this location?
    
    NOTE: Per Walk Score Terms of Service (Section 3), scores should NOT be 
    cached or stored without a Walk Score Professional subscription. This 
    function fetches fresh data each time.
    
    Args:
        lat: Latitude of the property.
        lon: Longitude of the property.
        address: Optional street address (improves accuracy).
        api_key: Walk Score API key (defaults to env var).
    
    Returns:
        Dict with scores and branding info for compliance:
        - walk_score, transit_score, bike_score: int or None
        - ws_link: URL to Walk Score page (REQUIRED for display per ToS)
        - logo_url: Walk Score logo URL (REQUIRED for display per ToS)
        - description: Walk Score interpretation (e.g., "Walker's Paradise")
    """
    if api_key is None:
        api_key = get_walkscore_api_key()
    
    if api_key is None:
        return {
            "walk_score": None, 
            "transit_score": None, 
            "bike_score": None,
            "ws_link": None,
            "logo_url": None,
            "description": None,
        }
    
    # Check daily limit
    if not _check_daily_limit():
        return {
            "walk_score": None, 
            "transit_score": None, 
            "bike_score": None,
            "ws_link": None,
            "logo_url": None,
            "description": None,
            "error": "daily_limit_exceeded",
        }
    
    # Build request parameters per API docs
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
        # Rate limiting - be a good API citizen
        time.sleep(WALKSCORE_RATE_LIMIT_DELAY)
        
        response = requests.get(WALKSCORE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Increment daily counter
        _increment_daily_call_count()
        
        # Check for API errors
        if data.get("status") != 1:
            status_messages = {
                2: "Score being calculated, try again later",
                30: "Invalid latitude/longitude",
                31: "Walk Score not available for this location",
                40: "Invalid API key",
                41: "API quota exceeded (5,000/day limit)",
                42: "IP address blocked - contact Walk Score support",
            }
            status = data.get("status", 0)
            error_msg = status_messages.get(status, f"Unknown status {status}")
            logger.warning(f"Walk Score API error: {error_msg}")
            
            return {
                "walk_score": None, 
                "transit_score": None, 
                "bike_score": None,
                "ws_link": data.get("ws_link"),  # May still be provided
                "logo_url": None,
                "description": None,
                "error": error_msg,
            }
        
        # Extract scores AND branding info (required by ToS Section 4)
        result = {
            "walk_score": data.get("walkscore"),
            "transit_score": data.get("transit", {}).get("score") if isinstance(data.get("transit"), dict) else None,
            "bike_score": data.get("bike", {}).get("score") if isinstance(data.get("bike"), dict) else None,
            # REQUIRED for branding compliance:
            "ws_link": data.get("ws_link"),  # Must link to this when displaying score
            "logo_url": data.get("logo_url"),  # Must display this logo
            "description": data.get("description"),  # e.g., "Walker's Paradise"
            "transit_description": data.get("transit", {}).get("description") if isinstance(data.get("transit"), dict) else None,
            "bike_description": data.get("bike", {}).get("description") if isinstance(data.get("bike"), dict) else None,
        }
        
        logger.debug(f"Walk Score for ({lat:.4f}, {lon:.4f}): {result['walk_score']} ({result['description']})")
        return result
        
    except requests.exceptions.Timeout:
        logger.warning(f"Walk Score API timeout for ({lat:.4f}, {lon:.4f})")
        return {"walk_score": None, "transit_score": None, "bike_score": None, "ws_link": None, "logo_url": None, "description": None, "error": "timeout"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Walk Score API request error: {e}")
        return {"walk_score": None, "transit_score": None, "bike_score": None, "ws_link": None, "logo_url": None, "description": None, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error fetching Walk Score: {e}")
        return {"walk_score": None, "transit_score": None, "bike_score": None, "ws_link": None, "logo_url": None, "description": None, "error": str(e)}


def add_walkscore_features_live(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    address_col: Optional[str] = None,
    max_calls: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add Walk Score features by fetching LIVE from API (no caching).
    
    Per Walk Score Terms of Service (Section 3), caching/storing scores 
    requires a Walk Score Professional subscription. This function fetches
    fresh data each time - suitable for:
    - One-time property analysis
    - Small batches (< 5000 properties/day)
    
    For larger datasets, consider Walk Score Professional:
    https://www.walkscore.com/professional/
    
    Args:
        df: DataFrame with latitude/longitude columns.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        address_col: Optional address column for better accuracy.
        max_calls: Maximum API calls to make (default: unlimited up to daily limit).
    
    Returns:
        DataFrame with added walk_score, transit_score, bike_score columns.
    """
    df = df.copy()
    
    # Check for API key first
    api_key = get_walkscore_api_key()
    if api_key is None:
        logger.warning("No Walk Score API key. Adding None values.")
        df["walk_score"] = None
        df["transit_score"] = None
        df["bike_score"] = None
        df["ws_link"] = None
        return df
    
    # Check daily limit at start
    current_calls = _get_daily_call_count()
    remaining = WALKSCORE_DAILY_LIMIT - WALKSCORE_DAILY_LIMIT_BUFFER - current_calls
    
    if remaining <= 0:
        logger.error(f"Daily API limit reached ({current_calls} calls today). Cannot fetch Walk Scores.")
        df["walk_score"] = None
        df["transit_score"] = None
        df["bike_score"] = None
        df["ws_link"] = None
        return df
    
    # Limit calls if needed
    total_needed = len(df)
    if max_calls:
        total_needed = min(total_needed, max_calls)
    if total_needed > remaining:
        logger.warning(f"Only {remaining} API calls remaining today. Processing first {remaining} rows.")
        total_needed = remaining
    
    # Fetch scores for each property
    walk_scores = []
    transit_scores = []
    bike_scores = []
    ws_links = []
    api_calls = 0
    
    logger.info(f"Fetching Walk Scores for up to {total_needed} properties...")
    logger.info(f"  Daily API calls so far: {current_calls}/{WALKSCORE_DAILY_LIMIT}")
    
    for idx, row in df.iterrows():
        if api_calls >= total_needed:
            # Add None for remaining rows
            walk_scores.append(None)
            transit_scores.append(None)
            bike_scores.append(None)
            ws_links.append(None)
            continue
            
        lat = row[lat_col]
        lon = row[lon_col]
        
        if pd.isna(lat) or pd.isna(lon):
            walk_scores.append(None)
            transit_scores.append(None)
            bike_scores.append(None)
            ws_links.append(None)
            continue
        
        # Fetch from API (live, no cache)
        address = row[address_col] if address_col and address_col in row.index else None
        scores = fetch_walkscore(lat, lon, address=address, api_key=api_key)
        
        walk_scores.append(scores.get("walk_score"))
        transit_scores.append(scores.get("transit_score"))
        bike_scores.append(scores.get("bike_score"))
        ws_links.append(scores.get("ws_link"))
        
        api_calls += 1
        
        # Progress logging
        if api_calls % 100 == 0:
            logger.info(f"  Fetched {api_calls}/{total_needed} scores...")
    
    # Add columns to DataFrame
    df["walk_score"] = walk_scores
    df["transit_score"] = transit_scores
    df["bike_score"] = bike_scores
    df["ws_link"] = ws_links  # Required for branding compliance
    
    # Summary
    non_null = df["walk_score"].notna().sum()
    logger.info(f"Walk Score Results:")
    logger.info(f"  API calls made: {api_calls}")
    logger.info(f"  Properties with scores: {non_null}/{len(df)}")
    logger.info(f"  Daily total: {_get_daily_call_count()}/{WALKSCORE_DAILY_LIMIT}")
    
    if non_null > 0:
        logger.info(f"  Mean Walk Score: {df['walk_score'].mean():.1f}")
    
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
    df = df.copy()
    
    logger.info("Generating simulated Walk Score features (no API key available)")
    logger.info("  NOTE: For production, get a free API key at walkscore.com")
    
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
    
    # No ws_link for simulated data
    df["ws_link"] = None
    
    logger.info(f"Simulated Walk Scores (for development only):")
    logger.info(f"  Mean Walk Score: {df['walk_score'].mean():.1f}")
    logger.info(f"  Mean Transit Score: {df['transit_score'].mean():.1f}")
    logger.info(f"  Mean Bike Score: {df['bike_score'].mean():.1f}")
    
    return df


def ensure_walkscore_features(
    df: pd.DataFrame,
    use_simulation_fallback: bool = True,
    max_api_calls: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ensure Walk Score features exist in DataFrame.
    
    Tries API first (live fetch), falls back to simulation if unavailable.
    
    Args:
        df: DataFrame with property data.
        use_simulation_fallback: If True, use simulated scores when API unavailable.
        max_api_calls: Limit on API calls (to preserve daily quota).
    
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
        return add_walkscore_features_live(df, max_calls=max_api_calls)
    
    # Fall back to simulation
    if use_simulation_fallback:
        return simulate_walkscore_features(df)
    
    # Just add None columns
    df = df.copy()
    df["walk_score"] = None
    df["transit_score"] = None
    df["bike_score"] = None
    df["ws_link"] = None
    return df


# ====================================
# DEPRECATED: Caching functions removed per Walk Score ToS Section 3
# "You will not...cache or otherwise store any Walk Score content 
# including but not limited to Walk Score, Bike Score, and Transit Score 
# ratings...without WS's prior written consent."
#
# For caching capability, upgrade to Walk Score Professional:
# https://www.walkscore.com/professional/
# ====================================
