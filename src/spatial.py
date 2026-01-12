"""
S.P.E.C. Valuation Engine - Spatial Utilities
==============================================
Geospatial utilities for property mapping and analysis.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


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


def get_neighborhood_stats(df: pd.DataFrame, zip_code: str) -> Dict:
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
