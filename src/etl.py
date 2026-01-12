"""
S.P.E.C. Valuation Engine - ETL Pipeline
=========================================
Extract, Transform, Load pipeline for housing data.
Handles synthetic data generation, cleaning, and persistence.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import from sibling config module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    HOUSING_CSV,
    HOUSING_PARQUET,
    DATABASE_PATH,
    DATA_DIR,
    SYNTHETIC_DATA_SIZE,
    ZIP_CODES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_housing_data(
    n_samples: int = SYNTHETIC_DATA_SIZE,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic housing data for the San Francisco Bay Area.
    
    Args:
        n_samples: Number of housing records to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with synthetic housing data.
    """
    np.random.seed(seed)
    logger.info(f"Generating {n_samples} synthetic housing records...")
    
    # Generate base features
    sqft = np.random.normal(1800, 600, n_samples).clip(600, 5000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.35, 0.2, 0.1])
    year_built = np.random.randint(1920, 2024, n_samples)
    condition = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.10])
    zip_codes = np.random.choice(ZIP_CODES, n_samples)
    
    # Generate lat/lon (San Francisco area)
    lat = np.random.uniform(37.70, 37.82, n_samples)
    lon = np.random.uniform(-122.52, -122.35, n_samples)
    
    # Calculate price based on features (realistic model)
    base_price = 400_000
    price = (
        base_price
        + sqft * 450                          # $/sqft
        + bedrooms * 75_000                   # Bedroom premium
        + (year_built - 1950) * 1_500         # Age adjustment
        + condition * 40_000                  # Condition premium
        + np.random.normal(0, 50_000, n_samples)  # Market noise
    )
    price = price.clip(200_000, 4_000_000).astype(int)
    
    # Add market metrics
    days_on_market = np.random.exponential(30, n_samples).clip(1, 180).astype(int)
    
    df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "lat": lat.round(6),
        "lon": lon.round(6),
        "price": price,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "year_built": year_built,
        "zip_code": zip_codes,
        "condition": condition,
        "days_on_market": days_on_market,
    })
    
    logger.info(f"Generated {len(df)} records. Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to snake_case.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with cleaned column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df


def ensure_directories() -> None:
    """Create necessary data directories if they don't exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def load_or_generate_raw_data() -> pd.DataFrame:
    """
    Load housing data from CSV or generate synthetic data if not found.
    
    Returns:
        DataFrame with housing data.
    """
    ensure_directories()
    
    if HOUSING_CSV.exists():
        logger.info(f"Loading existing data from {HOUSING_CSV}")
        df = pd.read_csv(HOUSING_CSV)
    else:
        logger.warning(f"Raw data not found at {HOUSING_CSV}. Generating synthetic data...")
        df = generate_synthetic_housing_data()
        
        # Save to raw directory for reference (though it's .gitignored)
        df.to_csv(HOUSING_CSV, index=False)
        logger.info(f"Saved synthetic data to {HOUSING_CSV}")
    
    return df


def save_to_parquet(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """
    Save DataFrame to Parquet format for fast loading.
    
    Args:
        df: DataFrame to save.
        path: Output path (defaults to HOUSING_PARQUET).
    
    Returns:
        Path to saved file.
    """
    path = path or HOUSING_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df)} records to {path}")
    
    return path


def save_to_sqlite(df: pd.DataFrame, table_name: str = "sales") -> Path:
    """
    Save DataFrame to SQLite database for SQL querying.
    
    Args:
        df: DataFrame to save.
        table_name: Name of the table to create.
    
    Returns:
        Path to database file.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        # Drop existing table and recreate
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Create indexes for common queries
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_zip_code ON {table_name}(zip_code)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_price ON {table_name}(price)")
        conn.commit()
        
        logger.info(f"Saved {len(df)} records to SQLite table '{table_name}' at {DATABASE_PATH}")
    finally:
        conn.close()
    
    return DATABASE_PATH


def run_etl_pipeline() -> pd.DataFrame:
    """
    Execute the full ETL pipeline.
    
    1. Load or generate raw data
    2. Clean column names
    3. Save to Parquet (for speed)
    4. Save to SQLite (for SQL queries)
    
    Returns:
        Cleaned DataFrame.
    """
    logger.info("=" * 50)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 50)
    
    # Extract
    df = load_or_generate_raw_data()
    
    # Transform
    df = clean_column_names(df)
    
    # Load
    save_to_parquet(df)
    save_to_sqlite(df)
    
    logger.info("=" * 50)
    logger.info("ETL Pipeline Complete")
    logger.info("=" * 50)
    
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed Parquet file for the application.
    If not found, run the ETL pipeline first.
    
    Returns:
        DataFrame with processed housing data.
    """
    if not HOUSING_PARQUET.exists():
        logger.info("Processed data not found. Running ETL pipeline...")
        return run_etl_pipeline()
    
    return pd.read_parquet(HOUSING_PARQUET)


def execute_sql_query(query: str) -> pd.DataFrame:
    """
    Execute a SQL query against the real estate database.
    
    Args:
        query: SQL query string.
    
    Returns:
        DataFrame with query results.
    """
    if not DATABASE_PATH.exists():
        logger.info("Database not found. Running ETL pipeline...")
        run_etl_pipeline()
    
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


if __name__ == "__main__":
    # Run ETL pipeline when executed directly
    df = run_etl_pipeline()
    print(f"\nData Summary:")
    print(df.describe())
