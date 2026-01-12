"""
S.P.E.C. Valuation Engine - Global Configuration
=================================================
Centralized settings for colors, paths, and constants.
Finance-grade dark theme with professional aesthetics.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any


# ====================================
# PATH CONFIGURATION
# ====================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Specific file paths
HOUSING_CSV = RAW_DATA_DIR / "housing.csv"
HOUSING_PARQUET = PROCESSED_DATA_DIR / "housing.parquet"
DATABASE_PATH = DATA_DIR / "real_estate.db"
MODEL_PATH = ASSETS_DIR / "model.pkl"


# ====================================
# COLOR PALETTE (Finance Dark Mode)
# ====================================
@dataclass(frozen=True)
class ColorPalette:
    """Professional finance-grade color scheme."""
    
    # Primary Background (Gunmetal Grey spectrum)
    BG_PRIMARY: str = "#1A1D23"       # Main background
    BG_SECONDARY: str = "#22262E"     # Card/panel background
    BG_TERTIARY: str = "#2A2F38"      # Elevated surfaces
    
    # Accent Colors
    EMERALD_GREEN: str = "#00D47E"    # Profit / Undervalued
    EMERALD_LIGHT: str = "#4AE59E"    # Hover states
    CRIMSON_RED: str = "#FF4757"      # Risk / Overvalued
    CRIMSON_LIGHT: str = "#FF6B7A"    # Hover states
    
    # Neutral Palette
    TEXT_PRIMARY: str = "#FFFFFF"     # Headers
    TEXT_SECONDARY: str = "#A0A4AB"   # Body text
    TEXT_MUTED: str = "#6B7280"       # Captions
    
    # Chart Colors
    CHART_BLUE: str = "#3B82F6"       # Primary series
    CHART_PURPLE: str = "#8B5CF6"     # Secondary series
    CHART_AMBER: str = "#F59E0B"      # Tertiary series
    CHART_CYAN: str = "#06B6D4"       # Quaternary series
    
    # Status Colors
    WARNING: str = "#FBBF24"
    INFO: str = "#60A5FA"


COLORS = ColorPalette()


# ====================================
# SHAP WATERFALL COLORS
# ====================================
SHAP_POSITIVE_COLOR = COLORS.EMERALD_GREEN
SHAP_NEGATIVE_COLOR = COLORS.CRIMSON_RED
SHAP_BASELINE_COLOR = COLORS.TEXT_MUTED


# ====================================
# MAP CONFIGURATION
# ====================================
MAP_STYLE = "dark"
MAP_CENTER_LAT = 37.7749  # San Francisco default
MAP_CENTER_LON = -122.4194
MAP_ZOOM = 11


# ====================================
# MODEL CONFIGURATION
# ====================================
MODEL_FEATURES = [
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
]
TARGET_COLUMN = "price"

XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


# ====================================
# STREAMLIT CONFIGURATION
# ====================================
PAGE_TITLE = "S.P.E.C. Valuation Engine"
PAGE_ICON = "🏠"
LAYOUT = "wide"

# Caching TTL (seconds)
CACHE_TTL_DATA = 3600      # 1 hour for data
CACHE_TTL_MODEL = 86400    # 24 hours for model


# ====================================
# SYNTHETIC DATA GENERATION
# ====================================
SYNTHETIC_DATA_SIZE = 500
ZIP_CODES = [
    "94102", "94103", "94104", "94105", "94107",
    "94108", "94109", "94110", "94111", "94112",
    "94114", "94115", "94116", "94117", "94118",
]

CONDITION_SCALE = {
    1: "Poor",
    2: "Fair",
    3: "Average",
    4: "Good",
    5: "Excellent",
}
