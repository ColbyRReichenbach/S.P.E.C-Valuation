"""
S.P.E.C. Valuation Engine - Spatial Utilities
==============================================
Geospatial utilities for property mapping and analysis.
V2.0 with H3 Hexagonal Indexing for spatial features.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# Import from sibling config module
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    H3_RESOLUTION,
    CITY_CENTER_LAT,
    CITY_CENTER_LON,
    COLORS,
)

# Configure logging
logger = logging.getLogger(__name__)


# ====================================
# H3 HEXAGONAL INDEXING
# ====================================
def _get_h3():
    """
    Lazy import of h3 library.
    
    Returns:
        h3 module or None if not installed.
    """
    try:
        import h3
        return h3
    except ImportError:
        logger.warning("h3 library not installed. H3 features disabled.")
        return None


def lat_lon_to_h3(
    lat: float,
    lon: float,
    resolution: int = H3_RESOLUTION
) -> Optional[str]:
    """
    Convert latitude/longitude to H3 hexagonal index.
    
    H3 is Uber's hierarchical spatial indexing system that divides
    the world into hexagonal cells. Resolution 9 gives ~0.1 km² cells.
    
    Args:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        resolution: H3 resolution (0-15, higher = smaller cells).
    
    Returns:
        H3 index string or None if h3 not available.
    """
    h3 = _get_h3()
    if h3 is None:
        return None
    
    try:
        return h3.latlng_to_cell(lat, lon, resolution)
    except Exception as e:
        logger.error(f"H3 conversion failed for ({lat}, {lon}): {e}")
        return None


def h3_to_lat_lon(h3_index: str) -> Optional[Tuple[float, float]]:
    """
    Convert H3 index back to latitude/longitude (cell center).
    
    Args:
        h3_index: H3 cell index.
    
    Returns:
        Tuple of (lat, lon) or None if invalid.
    """
    h3 = _get_h3()
    if h3 is None:
        return None
    
    try:
        lat, lon = h3.cell_to_latlng(h3_index)
        return (lat, lon)
    except Exception as e:
        logger.error(f"H3 reverse conversion failed for {h3_index}: {e}")
        return None


def add_h3_index(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    resolution: int = H3_RESOLUTION
) -> pd.DataFrame:
    """
    Add H3 hexagonal index column to DataFrame.
    
    Args:
        df: DataFrame with lat/lon columns.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        resolution: H3 resolution level.
    
    Returns:
        DataFrame with added 'h3_index' column.
    """
    df = df.copy()
    
    h3 = _get_h3()
    if h3 is None:
        # Fallback: use rounded coordinates as pseudo-index
        logger.warning("H3 not available. Using coordinate-based pseudo-index.")
        df["h3_index"] = df.apply(
            lambda row: f"{round(row[lat_col], 3)}_{round(row[lon_col], 3)}",
            axis=1
        )
        return df
    
    def get_h3(row):
        try:
            return h3.latlng_to_cell(row[lat_col], row[lon_col], resolution)
        except Exception:
            return None
    
    df["h3_index"] = df.apply(get_h3, axis=1)
    
    # Log coverage
    valid_count = df["h3_index"].notna().sum()
    logger.info(
        f"Added H3 index (resolution {resolution}) to {valid_count}/{len(df)} records."
    )
    
    return df


def h3_index_to_numeric(h3_index: str) -> int:
    """
    Convert H3 index string to numeric value for ML models.
    
    This is useful because tree-based models can work with
    the numeric representation of H3 indices.
    
    Args:
        h3_index: H3 cell index string.
    
    Returns:
        Integer representation of the H3 index.
    """
    if h3_index is None:
        return 0
    
    try:
        # H3 indices are hexadecimal strings - convert to int
        return int(h3_index, 16)
    except (ValueError, TypeError):
        return 0


def add_h3_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numeric representation of H3 index for ML features.
    
    Args:
        df: DataFrame with 'h3_index' column.
    
    Returns:
        DataFrame with added 'h3_numeric' column.
    """
    df = df.copy()
    
    if "h3_index" not in df.columns:
        df = add_h3_index(df)
    
    df["h3_numeric"] = df["h3_index"].apply(h3_index_to_numeric)
    
    return df


# ====================================
# DISTANCE CALCULATIONS
# ====================================
def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the Haversine distance between two points in kilometers.
    
    Args:
        lat1, lon1: Coordinates of the first point.
        lat2, lon2: Coordinates of the second point.
    
    Returns:
        Distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (
        np.sin(delta_lat / 2) ** 2 +
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_distance_to_center(
    lat: float,
    lon: float,
    center_lat: float = CITY_CENTER_LAT,
    center_lon: float = CITY_CENTER_LON
) -> float:
    """
    Calculate distance from a point to city center.
    
    Args:
        lat: Property latitude.
        lon: Property longitude.
        center_lat: City center latitude (default: SF City Hall).
        center_lon: City center longitude.
    
    Returns:
        Distance to city center in kilometers.
    """
    return calculate_distance(lat, lon, center_lat, center_lon)


def add_distance_to_center(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    center_lat: float = CITY_CENTER_LAT,
    center_lon: float = CITY_CENTER_LON
) -> pd.DataFrame:
    """
    Add distance to city center column to DataFrame.
    
    This is a powerful spatial feature for property valuation
    as properties closer to downtown typically command premiums.
    
    Args:
        df: DataFrame with coordinate columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        center_lat: City center latitude.
        center_lon: City center longitude.
    
    Returns:
        DataFrame with added 'distance_to_center_km' column.
    """
    df = df.copy()
    
    df["distance_to_center_km"] = df.apply(
        lambda row: calculate_distance_to_center(
            row[lat_col], row[lon_col], center_lat, center_lon
        ),
        axis=1
    )
    
    logger.info(
        f"Added distance_to_center_km. "
        f"Range: {df['distance_to_center_km'].min():.2f} - "
        f"{df['distance_to_center_km'].max():.2f} km"
    )
    
    return df


# ====================================
# SPATIAL FEATURE ENGINEERING
# ====================================
def add_all_spatial_features(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon"
) -> pd.DataFrame:
    """
    Add all spatial features to DataFrame.
    
    Features added:
    - h3_index: H3 hexagonal cell index (string)
    - h3_numeric: Numeric representation for ML
    - distance_to_center_km: Distance to city center
    
    Args:
        df: DataFrame with coordinate columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
    
    Returns:
        DataFrame with all spatial features added.
    """
    logger.info("Adding spatial features...")
    
    # Add H3 index
    df = add_h3_index(df, lat_col, lon_col)
    
    # Add numeric H3 for ML
    df = add_h3_numeric(df)
    
    # Add distance to center
    df = add_distance_to_center(df, lat_col, lon_col)
    
    logger.info("Spatial feature engineering complete.")
    
    return df


def get_h3_neighbors(h3_index: str, k: int = 1) -> List[str]:
    """
    Get neighboring H3 cells within k rings.
    
    Useful for creating neighborhood-level features.
    
    Args:
        h3_index: Center cell index.
        k: Number of rings (1 = immediate neighbors).
    
    Returns:
        List of neighbor H3 indices.
    """
    h3 = _get_h3()
    if h3 is None:
        return []
    
    try:
        return list(h3.grid_disk(h3_index, k))
    except Exception as e:
        logger.error(f"Failed to get H3 neighbors: {e}")
        return []


def calculate_h3_neighborhood_stats(
    df: pd.DataFrame,
    target_col: str = "price",
    k_rings: int = 1
) -> pd.DataFrame:
    """
    Calculate neighborhood statistics using H3 spatial indexing.
    
    For each property, computes statistics from neighboring H3 cells.
    
    Args:
        df: DataFrame with 'h3_index' column.
        target_col: Column to compute statistics for.
        k_rings: Number of H3 rings to include.
    
    Returns:
        DataFrame with neighborhood statistics.
    """
    h3 = _get_h3()
    if h3 is None or "h3_index" not in df.columns:
        logger.warning("Cannot compute H3 neighborhood stats.")
        return df
    
    df = df.copy()
    
    # Create lookup of h3_index -> values
    h3_values = df.groupby("h3_index")[target_col].agg(["mean", "median", "count"])
    
    def get_neighborhood_stats(h3_idx):
        if pd.isna(h3_idx):
            return None, None
        
        neighbors = get_h3_neighbors(h3_idx, k_rings)
        neighbor_stats = h3_values.loc[h3_values.index.isin(neighbors)]
        
        if len(neighbor_stats) == 0:
            return None, None
        
        return (
            neighbor_stats["mean"].mean(),
            neighbor_stats["count"].sum()
        )
    
    # Apply to each row
    stats = df["h3_index"].apply(get_neighborhood_stats)
    df["neighborhood_avg_price"] = [s[0] for s in stats]
    df["neighborhood_count"] = [s[1] for s in stats]
    
    return df


# ====================================
# VALUATION STATUS & MAP UTILITIES
# ====================================
def add_valuation_status(
    df: pd.DataFrame,
    list_price_col: str = "price",
    model_price_col: str = "model_price"
) -> pd.DataFrame:
    """
    Add valuation status (Undervalued/Overvalued) to property DataFrame.
    
    Args:
        df: DataFrame with property data.
        list_price_col: Column name for list price.
        model_price_col: Column name for model-predicted price.
    
    Returns:
        DataFrame with added 'valuation_status' and 'price_delta' columns.
    """
    df = df.copy()
    
    # Calculate price difference
    df["price_delta"] = df[model_price_col] - df[list_price_col]
    df["price_delta_pct"] = (df["price_delta"] / df[list_price_col]) * 100
    
    # Determine valuation status
    df["valuation_status"] = df.apply(
        lambda row: "Undervalued" if row["price_delta"] > 0 else "Overvalued",
        axis=1
    )
    
    return df


def prepare_map_data(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Prepare DataFrame for Streamlit map display.
    
    Streamlit's st.map expects columns named 'lat' and 'lon'.
    
    Args:
        df: DataFrame with coordinate columns.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
    
    Returns:
        DataFrame with standardized coordinate columns.
    """
    map_df = df.copy()
    
    # Ensure correct column names
    if lat_col != "lat":
        map_df["lat"] = map_df[lat_col]
    if lon_col != "lon":
        map_df["lon"] = map_df[lon_col]
    
    return map_df


def get_neighborhood_stats(df: pd.DataFrame, zip_code: str) -> Dict[str, Any]:
    """
    Calculate neighborhood statistics for a given zip code.
    
    Args:
        df: DataFrame with property data.
        zip_code: Zip code to analyze.
    
    Returns:
        Dictionary with neighborhood metrics.
    """
    neighborhood = df[df["zip_code"] == zip_code]
    
    if len(neighborhood) == 0:
        return {
            "count": 0,
            "avg_price": 0,
            "median_price": 0,
            "avg_sqft": 0,
            "avg_price_per_sqft": 0,
        }
    
    return {
        "count": len(neighborhood),
        "avg_price": neighborhood["price"].mean(),
        "median_price": neighborhood["price"].median(),
        "avg_sqft": neighborhood["sqft"].mean(),
        "avg_price_per_sqft": (neighborhood["price"] / neighborhood["sqft"]).mean(),
    }


def create_color_scale(
    values: pd.Series,
    green_hex: str = "#00D47E",
    red_hex: str = "#FF4757",
) -> List[str]:
    """
    Create a color scale from red (negative) to green (positive).
    
    Args:
        values: Series of numeric values.
        green_hex: Hex color for positive values.
        red_hex: Hex color for negative values.
    
    Returns:
        List of hex color strings.
    """
    colors = []
    for val in values:
        if val > 0:
            colors.append(green_hex)
        else:
            colors.append(red_hex)
    return colors


# ====================================
# H3 VISUALIZATION UTILITIES
# ====================================
def get_h3_boundary(h3_index: str) -> Optional[List[Tuple[float, float]]]:
    """
    Get the boundary coordinates of an H3 cell for visualization.
    
    Args:
        h3_index: H3 cell index.
    
    Returns:
        List of (lat, lon) tuples defining the hexagon boundary.
    """
    h3 = _get_h3()
    if h3 is None:
        return None
    
    try:
        boundary = h3.cell_to_boundary(h3_index)
        return list(boundary)
    except Exception as e:
        logger.error(f"Failed to get H3 boundary: {e}")
        return None


def create_h3_geojson(
    df: pd.DataFrame,
    value_col: str = "price"
) -> Dict[str, Any]:
    """
    Create GeoJSON from H3-indexed DataFrame for map visualization.
    
    Args:
        df: DataFrame with 'h3_index' column.
        value_col: Column to use for cell coloring.
    
    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    h3 = _get_h3()
    if h3 is None or "h3_index" not in df.columns:
        return {"type": "FeatureCollection", "features": []}
    
    # Aggregate by H3 cell
    h3_stats = df.groupby("h3_index").agg({
        value_col: ["mean", "count"]
    }).reset_index()
    h3_stats.columns = ["h3_index", "value_mean", "count"]
    
    features = []
    for _, row in h3_stats.iterrows():
        try:
            boundary = h3.cell_to_boundary(row["h3_index"])
            # GeoJSON uses [lon, lat] order
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])  # Close the polygon
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "h3_index": row["h3_index"],
                    "value": row["value_mean"],
                    "count": row["count"]
                }
            })
        except Exception:
            continue
    
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    # Test spatial utilities
    print("Testing Spatial Utilities v2.0")
    print("=" * 50)
    
    # Test H3
    test_lat, test_lon = 37.7749, -122.4194
    h3_idx = lat_lon_to_h3(test_lat, test_lon)
    print(f"\nH3 Index for SF City Hall: {h3_idx}")
    
    # Test distance
    distance = calculate_distance_to_center(37.7599, -122.4284)
    print(f"Distance from Sunset to City Center: {distance:.2f} km")
    
    # Test with DataFrame
    test_df = pd.DataFrame({
        "id": [1, 2, 3],
        "lat": [37.7749, 37.7599, 37.7849],
        "lon": [-122.4194, -122.4284, -122.4094],
        "price": [1000000, 800000, 1200000],
    })
    
    print("\nOriginal DataFrame:")
    print(test_df)
    
    enriched_df = add_all_spatial_features(test_df)
    print("\nWith Spatial Features:")
    print(enriched_df)
