"""
S.P.E.C. Valuation Engine - Global Configuration
=================================================
Centralized settings for colors, paths, and constants.
Finance-grade dark theme with professional aesthetics.
V2.0 Production-Grade Configuration
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import datetime


# ====================================
# PATH CONFIGURATION
# ====================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"
TESTS_DIR = PROJECT_ROOT / "tests"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# Specific file paths
HOUSING_CSV = RAW_DATA_DIR / "housing.csv"
HOUSING_PARQUET = PROCESSED_DATA_DIR / "housing.parquet"
DATABASE_PATH = DATA_DIR / "real_estate.db"
MODEL_PATH = ASSETS_DIR / "model.pkl"
MARKET_REPORTS_DIR = DATA_DIR / "market_reports"


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
# GEOSPATIAL / H3 CONFIGURATION
# ====================================
H3_RESOLUTION: int = 9  # Resolution 9 ~= 0.1 km² hexagons
CITY_CENTER_LAT: float = 37.7749  # San Francisco City Hall
CITY_CENTER_LON: float = -122.4194

# BART Station Coordinates (SF stations)
BART_STATIONS: List[tuple] = [
    ("Embarcadero", 37.7929, -122.3968),
    ("Montgomery", 37.7894, -122.4013),
    ("Powell", 37.7844, -122.4078),
    ("Civic Center", 37.7796, -122.4139),
    ("16th St Mission", 37.7650, -122.4194),
    ("24th St Mission", 37.7522, -122.4182),
    ("Glen Park", 37.7329, -122.4343),
    ("Balboa Park", 37.7219, -122.4474),
]


# ====================================
# MODEL CONFIGURATION
# ====================================
# V2.1 - Core features only
MODEL_FEATURES: List[str] = [
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
]

# V2 - With spatial (legacy)
MODEL_FEATURES_V2: List[str] = [
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
    "h3_index",
    "distance_to_center_km",
]

# V2.2 - Enhanced features (10 total)
MODEL_FEATURES_V2_2: List[str] = [
    # Core (4)
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
    # Derived (3)
    "property_age",
    "sqft_per_bedroom",
    "is_newer_construction",
    # Spatial (3)
    "distance_to_downtown_km",
    "distance_to_nearest_bart_km",
    "neighborhood_price_tier",
]

# V3.0 - With H3 Spatial Lag Features (14 total)
MODEL_FEATURES_V3: List[str] = [
    # Core (4)
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
    # Derived (3)
    "property_age",
    "sqft_per_bedroom",
    "is_newer_construction",
    # Spatial - Distance (3)
    "distance_to_downtown_km",
    "distance_to_nearest_bart_km",
    "neighborhood_price_tier",
    # Spatial - H3 Lags (4)
    "h3_median_ppsf",            # Median $/sqft in same H3 cell
    "h3_neighbor_median_ppsf",   # Median $/sqft in 6 adjacent cells
    "h3_listing_density",        # Number of listings in H3 cell
    "h3_price_percentile",       # Where this property ranks (0-100)
]

# V3.0 Full - With Walk Score Features (17 total) - CURRENT
MODEL_FEATURES_V3_FULL: List[str] = [
    # Core (4)
    "sqft",
    "bedrooms",
    "year_built",
    "condition",
    # Derived (3)
    "property_age",
    "sqft_per_bedroom",
    "is_newer_construction",
    # Spatial - Distance (3)
    "distance_to_downtown_km",
    "distance_to_nearest_bart_km",
    "neighborhood_price_tier",
    # Spatial - H3 Lags (4)
    "h3_median_ppsf",
    "h3_neighbor_median_ppsf",
    "h3_listing_density",
    "h3_price_percentile",
    # Walk Score API (3) - V3.0 NEW
    "walk_score",                # Walkability (0-100)
    "transit_score",             # Public transit access (0-100)
    "bike_score",                # Bikeability (0-100)
]

TARGET_COLUMN: str = "price"

XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


# ====================================
# OPTUNA HYPERPARAMETER SEARCH SPACE
# ====================================
OPTUNA_N_TRIALS: int = 50
OPTUNA_TIMEOUT_SECONDS: int = 600  # 10 minutes max

OPTUNA_SEARCH_SPACE: Dict[str, Any] = {
    "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
    "max_depth": {"low": 3, "high": 10},
    "subsample": {"low": 0.6, "high": 1.0},
    "n_estimators": {"low": 50, "high": 300},
    "min_child_weight": {"low": 1, "high": 10},
    "colsample_bytree": {"low": 0.5, "high": 1.0},
}


# ====================================
# MLFLOW CONFIGURATION
# ====================================
MLFLOW_EXPERIMENT_NAME: str = "spec_valuation_engine"
MLFLOW_TRACKING_URI: str = str(MLFLOW_DIR)


# ====================================
# RAG / VECTOR STORE CONFIGURATION
# ====================================
VECTOR_COLLECTION_NAME: str = "market_reports"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 500  # Characters per chunk
CHUNK_OVERLAP: int = 50  # Overlap between chunks
TOP_K_RETRIEVAL: int = 3  # Number of chunks to retrieve


# ====================================
# DATA VALIDATION CONSTRAINTS
# ====================================
CURRENT_YEAR: int = datetime.datetime.now().year

DATA_VALIDATION_RULES: Dict[str, Any] = {
    "price_min": 0,
    "price_max": 100_000_000,
    "sqft_min": 100,
    "sqft_max": 50_000,
    "bedrooms_min": 0,
    "bedrooms_max": 20,
    "year_built_min": 1800,
    "year_built_max": CURRENT_YEAR,
    "condition_min": 1,
    "condition_max": 5,
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
# API CONNECTOR CONFIGURATION
# ====================================
API_TIMEOUT_SECONDS: int = 30
API_RETRY_ATTEMPTS: int = 3

# Zillow API (placeholder - requires real credentials)
ZILLOW_API_BASE_URL: str = "https://zillow-com1.p.rapidapi.com"
REDFIN_API_BASE_URL: str = "https://redfin-com-data.p.rapidapi.com"


# ====================================
# ZIP CODES & SYNTHETIC DATA (Legacy)
# ====================================
SYNTHETIC_DATA_SIZE = 500
ZIP_CODES: List[str] = [
    "94102", "94103", "94104", "94105", "94107",
    "94108", "94109", "94110", "94111", "94112",
    "94114", "94115", "94116", "94117", "94118",
]

CONDITION_SCALE: Dict[int, str] = {
    1: "Poor",
    2: "Fair",
    3: "Average",
    4: "Good",
    5: "Excellent",
}


# ====================================
# DOCKER / PRODUCTION CONFIGURATION
# ====================================
DOCKER_PYTHON_VERSION: str = "3.9"
STREAMLIT_PORT: int = 8501
