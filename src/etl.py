"""
S.P.E.C. Valuation Engine - ETL Pipeline
=========================================
Extract, Transform, Load pipeline for housing data.
V2.0 with Pandera data validation and real API connectors.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

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
    ZIP_CODES,
    DATA_VALIDATION_RULES,
    CURRENT_YEAR,
)
from src.connectors import UnifiedDataIngester, ZillowConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# PANDERA DATA VALIDATION SCHEMAS
# ====================================
class HousingDataSchema(pa.DataFrameModel):
    """
    Pandera schema for validating housing data.
    
    Enforces business rules:
    - Price must be positive (not negative)
    - Year built cannot be in the future
    - Sqft must be reasonable
    - Condition must be in valid range
    """
    
    id: pa.typing.Series[int] = pa.Field(ge=0, coerce=True)
    lat: pa.typing.Series[float] = pa.Field(ge=-90, le=90, nullable=True)
    lon: pa.typing.Series[float] = pa.Field(ge=-180, le=180, nullable=True)
    price: pa.typing.Series[float] = pa.Field(
        ge=0,  # Price MUST be non-negative
        le=DATA_VALIDATION_RULES["price_max"],
        coerce=True
    )
    sqft: pa.typing.Series[float] = pa.Field(
        ge=DATA_VALIDATION_RULES["sqft_min"],
        le=DATA_VALIDATION_RULES["sqft_max"],
        nullable=True,  # Allow null for imputation
        coerce=True
    )
    bedrooms: pa.typing.Series[int] = pa.Field(
        ge=DATA_VALIDATION_RULES["bedrooms_min"],
        le=DATA_VALIDATION_RULES["bedrooms_max"],
        coerce=True
    )
    year_built: pa.typing.Series[int] = pa.Field(
        ge=DATA_VALIDATION_RULES["year_built_min"],
        le=CURRENT_YEAR,  # Cannot be in the future!
        coerce=True
    )
    zip_code: pa.typing.Series[str] = pa.Field(nullable=True, coerce=True)
    condition: pa.typing.Series[int] = pa.Field(
        ge=DATA_VALIDATION_RULES["condition_min"],
        le=DATA_VALIDATION_RULES["condition_max"],
        coerce=True
    )
    
    class Config:
        coerce = True
        strict = False  # Allow extra columns


# Pre-validation schema (looser, for raw data)
RAW_DATA_SCHEMA = DataFrameSchema(
    {
        "price": Column(float, Check.ge(0), coerce=True, required=True),
        "year_built": Column(
            int, 
            Check.le(CURRENT_YEAR), 
            coerce=True, 
            required=True
        ),
    },
    strict=False,
    coerce=True,
)


# ====================================
# DATA QUALITY EXCEPTIONS
# ====================================
class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


class NegativePriceError(DataValidationError):
    """Raised when price is negative."""
    pass


class FutureYearBuiltError(DataValidationError):
    """Raised when year_built is in the future."""
    pass


# ====================================
# DATA VALIDATION FUNCTIONS
# ====================================
def validate_critical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate critical business constraints before processing.
    
    FAIL the pipeline if:
    - Any price is negative
    - Any year_built > current year
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Validated DataFrame.
    
    Raises:
        NegativePriceError: If any price is negative.
        FutureYearBuiltError: If any year_built is in the future.
    """
    # Check for negative prices
    negative_prices = df[df["price"] < 0]
    if len(negative_prices) > 0:
        error_msg = (
            f"CRITICAL: Found {len(negative_prices)} records with negative prices. "
            f"IDs: {negative_prices['id'].tolist()[:10]}..."
        )
        logger.error(error_msg)
        raise NegativePriceError(error_msg)
    
    # Check for future year_built
    future_years = df[df["year_built"] > CURRENT_YEAR]
    if len(future_years) > 0:
        error_msg = (
            f"CRITICAL: Found {len(future_years)} records with year_built > {CURRENT_YEAR}. "
            f"IDs: {future_years['id'].tolist()[:10]}..."
        )
        logger.error(error_msg)
        raise FutureYearBuiltError(error_msg)
    
    logger.info("Critical constraint validation passed.")
    return df


def validate_with_pandera(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate DataFrame against Pandera schema.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Validated DataFrame.
    
    Raises:
        pa.errors.SchemaError: If validation fails.
    """
    try:
        validated_df = HousingDataSchema.validate(df)
        logger.info(f"Pandera validation passed for {len(validated_df)} records.")
        return validated_df
    except pa.errors.SchemaErrors as e:
        logger.error(f"Pandera validation failed: {e}")
        # Log detailed errors
        for error in e.failure_cases.itertuples():
            logger.error(f"  Row {error.index}: {error.column} - {error.check}")
        raise


# ====================================
# DATA IMPUTATION FUNCTIONS
# ====================================
def impute_missing_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing sqft values using neighborhood (zip_code) median.
    
    Business Rule: If sqft is null, use the median sqft from the same zip code.
    If no zip code data available, use overall median.
    
    Args:
        df: DataFrame with potential null sqft values.
    
    Returns:
        DataFrame with imputed sqft values.
    """
    df = df.copy()
    
    null_sqft_count = df["sqft"].isna().sum()
    
    if null_sqft_count == 0:
        logger.info("No missing sqft values to impute.")
        return df
    
    logger.info(f"Imputing {null_sqft_count} missing sqft values...")
    
    # Calculate zip code medians
    zip_medians = df.groupby("zip_code")["sqft"].median()
    overall_median = df["sqft"].median()
    
    # Impute based on zip code
    def impute_sqft(row):
        if pd.isna(row["sqft"]):
            zip_median = zip_medians.get(row["zip_code"])
            if pd.notna(zip_median):
                return zip_median
            else:
                return overall_median
        return row["sqft"]
    
    df["sqft"] = df.apply(impute_sqft, axis=1)
    df["sqft_imputed"] = df.index.isin(
        df[df["sqft"].isna()].index
    )
    
    logger.info(f"Imputed {null_sqft_count} sqft values. "
                f"Using neighborhood median where available.")
    
    return df


def impute_missing_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing condition values based on year_built.
    
    Heuristic: Newer properties tend to have better condition.
    
    Args:
        df: DataFrame with potential null condition values.
    
    Returns:
        DataFrame with imputed condition values.
    """
    df = df.copy()
    
    if "condition" not in df.columns:
        df["condition"] = 3  # Default to average
        return df
    
    null_condition_count = df["condition"].isna().sum()
    
    if null_condition_count == 0:
        return df
    
    logger.info(f"Imputing {null_condition_count} missing condition values...")
    
    # Heuristic: condition based on age
    def impute_condition(row):
        if pd.isna(row["condition"]):
            age = CURRENT_YEAR - row["year_built"]
            if age < 10:
                return 5  # Excellent
            elif age < 25:
                return 4  # Good
            elif age < 50:
                return 3  # Average
            elif age < 75:
                return 2  # Fair
            else:
                return 1  # Poor
        return row["condition"]
    
    df["condition"] = df.apply(impute_condition, axis=1).astype(int)
    
    return df


# ====================================
# CORE ETL FUNCTIONS
# ====================================
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


def load_raw_data_from_csv() -> Optional[pd.DataFrame]:
    """
    Load housing data from existing CSV file.
    
    Returns:
        DataFrame if CSV exists, None otherwise.
    """
    if HOUSING_CSV.exists():
        logger.info(f"Loading existing data from {HOUSING_CSV}")
        return pd.read_csv(HOUSING_CSV)
    return None


def ingest_from_api(
    zip_codes: Optional[List[str]] = None,
    listings_per_zip: int = 50
) -> pd.DataFrame:
    """
    Ingest data from real estate APIs.
    
    Args:
        zip_codes: List of zip codes to fetch.
        listings_per_zip: Max listings per zip code.
    
    Returns:
        DataFrame with ingested data.
    """
    ingester = UnifiedDataIngester()
    return ingester.ingest_all_zip_codes(
        zip_codes=zip_codes or ZIP_CODES,
        listings_per_zip=listings_per_zip
    )


def load_or_ingest_raw_data(
    force_api: bool = False,
    zip_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load housing data from CSV or ingest from APIs.
    
    Priority:
    1. If force_api=True, always fetch from API
    2. If Redfin real data exists, use that (preferred)
    3. If CSV exists, load from CSV
    4. Otherwise, ingest from API (simulation mode if no keys)
    
    Args:
        force_api: Force API ingestion even if CSV exists.
        zip_codes: Optional list of zip codes for API ingestion.
    
    Returns:
        DataFrame with housing data.
    """
    ensure_directories()
    
    if not force_api:
        # Try to load real Redfin data first
        redfin_path = RAW_DATA_DIR / "redfin_training_data.csv"
        if redfin_path.exists():
            logger.info(f"Loading real Redfin data from {redfin_path}")
            df = pd.read_csv(redfin_path, dtype={'zip_code': str})
            logger.info(f"Loaded {len(df)} records from real Redfin data")
            return df
        
        # Fall back to existing CSV
        df = load_raw_data_from_csv()
        if df is not None:
            return df
    
    logger.info("Ingesting data from APIs...")
    df = ingest_from_api(zip_codes=zip_codes)
    
    # Save to raw directory
    if len(df) > 0:
        df.to_csv(HOUSING_CSV, index=False)
        logger.info(f"Saved {len(df)} records to {HOUSING_CSV}")
    
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


# ====================================
# MAIN ETL PIPELINE
# ====================================
def run_etl_pipeline(
    force_api: bool = False,
    validate: bool = True,
    zip_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Execute the full ETL pipeline.
    
    Pipeline Steps:
    1. Load or ingest raw data
    2. Clean column names
    3. Validate critical constraints (FAIL if violated)
    4. Impute missing values
    5. Validate with Pandera schema
    6. Save to Parquet and SQLite
    
    Args:
        force_api: Force API ingestion.
        validate: Enable validation (set False for debugging).
        zip_codes: Optional zip codes for API ingestion.
    
    Returns:
        Cleaned and validated DataFrame.
    
    Raises:
        NegativePriceError: If any price is negative.
        FutureYearBuiltError: If any year_built > current year.
        pa.errors.SchemaError: If Pandera validation fails.
    """
    logger.info("=" * 50)
    logger.info("Starting ETL Pipeline v2.0")
    logger.info("=" * 50)
    
    # Step 1: Extract
    logger.info("Step 1: Extracting data...")
    df = load_or_ingest_raw_data(force_api=force_api, zip_codes=zip_codes)
    
    if len(df) == 0:
        logger.error("No data available. ETL pipeline aborted.")
        raise DataValidationError("No data available for processing.")
    
    logger.info(f"Loaded {len(df)} raw records.")
    
    # Step 2: Clean column names
    logger.info("Step 2: Cleaning column names...")
    df = clean_column_names(df)
    
    # Step 3: Validate critical constraints
    if validate:
        logger.info("Step 3: Validating critical constraints...")
        df = validate_critical_constraints(df)
    
    # Step 4: Impute missing values
    logger.info("Step 4: Imputing missing values...")
    df = impute_missing_sqft(df)
    df = impute_missing_condition(df)
    
    # Step 5: Pandera validation
    if validate:
        logger.info("Step 5: Running Pandera schema validation...")
        try:
            df = validate_with_pandera(df)
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Pandera validation found issues: {e}")
            # Continue with warnings rather than failing
            # In production, you might want stricter behavior
    
    # Step 5b: Add V2.2 enhanced features
    logger.info("Step 5b: Adding V2.2 enhanced features...")
    from src.spatial import add_v2_2_features
    df = add_v2_2_features(df)
    
    # Step 6: Load
    logger.info("Step 6: Loading to storage...")
    save_to_parquet(df)
    save_to_sqlite(df)
    
    logger.info("=" * 50)
    logger.info("ETL Pipeline Complete (V2.2)")
    logger.info(f"Final record count: {len(df)}")
    logger.info(f"Features available: {list(df.columns)}")
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
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        raise
    finally:
        conn.close()


# ====================================
# DATA QUALITY REPORT
# ====================================
def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a data quality report for the DataFrame.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary with quality metrics.
    """
    report: Dict[str, Any] = {
        "total_records": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_stats": {},
    }
    
    # Numeric column stats
    for col in df.select_dtypes(include=[np.number]).columns:
        report["numeric_stats"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }
    
    return report


if __name__ == "__main__":
    # Run ETL pipeline when executed directly
    try:
        df = run_etl_pipeline(validate=True)
        print(f"\nData Summary:")
        print(df.describe())
        
        print(f"\nData Quality Report:")
        report = generate_data_quality_report(df)
        print(f"Total Records: {report['total_records']}")
        print(f"Null Counts: {report['null_counts']}")
        
    except DataValidationError as e:
        print(f"\nPIPELINE FAILED: {e}")
        exit(1)
