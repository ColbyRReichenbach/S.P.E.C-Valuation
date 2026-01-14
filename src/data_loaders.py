"""
S.P.E.C. Valuation Engine - Real Data Loaders
==============================================
Load real market data from Redfin Data Center.
No API keys required - uses downloaded CSV/TSV files.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

# Import from sibling config module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, ZIP_CODES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# REDFIN DATA LOADERS
# ====================================
class RedfinDataLoader:
    """
    Load real market data from Redfin Data Center downloads.
    
    Supports:
    - Zip code level market data
    - Metro level data
    - Historical trends
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory containing Redfin data files.
        """
        self.data_dir = data_dir or RAW_DATA_DIR / "redfin"
        self.sf_data_path = self.data_dir / "sf_bay_area.csv"
        self.ca_data_path = self.data_dir / "ca_latest.csv"
        
    def load_sf_market_data(self) -> pd.DataFrame:
        """
        Load SF Bay Area market data.
        
        Returns:
            DataFrame with zip-level market metrics.
        """
        if not self.sf_data_path.exists():
            raise FileNotFoundError(
                f"SF data not found at {self.sf_data_path}. "
                "Run the data download script first."
            )
        
        df = pd.read_csv(self.sf_data_path, dtype={'zip_code': str})
        logger.info(f"Loaded {len(df)} SF Bay Area zip codes")
        return df
    
    def load_california_data(self) -> pd.DataFrame:
        """
        Load all California market data.
        
        Returns:
            DataFrame with CA zip-level market metrics.
        """
        if not self.ca_data_path.exists():
            raise FileNotFoundError(
                f"CA data not found at {self.ca_data_path}. "
                "Run the data download script first."
            )
        
        df = pd.read_csv(self.ca_data_path, dtype={'REGION': str})
        # Extract zip code from REGION
        df['zip_code'] = df['REGION'].str.extract(r'(\d{5})')
        logger.info(f"Loaded {len(df)} California zip codes")
        return df
    
    def get_zip_market_stats(self, zip_code: str) -> Optional[Dict[str, Any]]:
        """
        Get market statistics for a specific zip code.
        
        Args:
            zip_code: Target zip code.
        
        Returns:
            Dictionary with market metrics or None.
        """
        try:
            df = self.load_sf_market_data()
            row = df[df['zip_code'] == zip_code]
            
            if len(row) == 0:
                return None
            
            row = row.iloc[0]
            return {
                'zip_code': zip_code,
                'median_sale_price': row.get('MEDIAN_SALE_PRICE'),
                'median_ppsf': row.get('MEDIAN_PPSF'),
                'homes_sold': row.get('HOMES_SOLD'),
            }
        except Exception as e:
            logger.error(f"Error getting stats for {zip_code}: {e}")
            return None


def generate_training_data_from_redfin(
    n_samples_per_zip: int = 50,
    zip_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate realistic training data based on real Redfin market stats.
    
    Uses actual median prices and price-per-sqft from Redfin data
    to create property-level training records with realistic distributions.
    
    Args:
        n_samples_per_zip: Number of samples to generate per zip code.
        zip_codes: List of zip codes to use. Defaults to all available.
    
    Returns:
        DataFrame with property-level training data.
    """
    loader = RedfinDataLoader()
    
    try:
        market_data = loader.load_sf_market_data()
    except FileNotFoundError:
        logger.warning("Redfin data not found. Using simulation fallback.")
        return None
    
    records = []
    
    for _, row in market_data.iterrows():
        zip_code = row['zip_code']
        
        if zip_codes and zip_code not in zip_codes:
            continue
        
        median_price = row.get('MEDIAN_SALE_PRICE')
        median_ppsf = row.get('MEDIAN_PPSF')
        
        if pd.isna(median_price) or pd.isna(median_ppsf):
            continue
        
        # Set random seed for reproducibility per zip
        np.random.seed(hash(zip_code) % 2**32)
        
        for i in range(n_samples_per_zip):
            # Generate sqft based on median price and ppsf
            base_sqft = median_price / median_ppsf if median_ppsf > 0 else 1500
            sqft = int(np.clip(np.random.normal(base_sqft, base_sqft * 0.3), 600, 5000))
            
            # Generate bedrooms correlated with sqft
            if sqft < 1000:
                bedrooms = np.random.choice([1, 2], p=[0.7, 0.3])
            elif sqft < 1500:
                bedrooms = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            elif sqft < 2500:
                bedrooms = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
            else:
                bedrooms = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            
            # Year built distribution skewed toward older SF housing
            year_built = int(np.random.choice(
                [1920, 1940, 1960, 1980, 2000, 2020],
                p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
            ) + np.random.randint(-10, 10))
            year_built = max(1850, min(2025, year_built))
            
            # Condition correlated with age
            age = 2025 - year_built
            if age < 10:
                condition = np.random.choice([4, 5], p=[0.3, 0.7])
            elif age < 30:
                condition = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            elif age < 60:
                condition = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
            else:
                condition = np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.2])
            
            # Price based on sqft and real ppsf, plus variance
            estimated_price = sqft * median_ppsf
            # Add variance based on condition and bedrooms
            condition_multiplier = 0.85 + (condition * 0.05)  # 0.9 - 1.1
            bedroom_adjustment = (bedrooms - 3) * 25000  # +/- $25K per bedroom from 3
            noise = np.random.normal(0, median_price * 0.1)
            
            price = int(np.clip(
                estimated_price * condition_multiplier + bedroom_adjustment + noise,
                median_price * 0.4, median_price * 2.5
            ))
            
            # Generate coordinates based on REAL zip code neighborhood centers
            # These are actual SF neighborhood centroids (no more water markers!)
            ZIP_COORDINATES = {
                '94102': (37.7810, -122.4188),  # Tenderloin/Civic Center
                '94103': (37.7726, -122.4110),  # SoMa
                '94104': (37.7914, -122.4018),  # Financial District
                '94105': (37.7898, -122.3928),  # Rincon Hill
                '94107': (37.7647, -122.3976),  # Potrero Hill/Dogpatch
                '94108': (37.7923, -122.4078),  # Chinatown
                '94109': (37.7929, -122.4213),  # Nob Hill/Russian Hill
                '94110': (37.7485, -122.4155),  # Mission District
                '94111': (37.7988, -122.3996),  # Embarcadero
                '94112': (37.7203, -122.4421),  # Ingleside
                '94114': (37.7609, -122.4350),  # Castro
                '94115': (37.7863, -122.4377),  # Pacific Heights
                '94116': (37.7435, -122.4866),  # Sunset (inner)
                '94117': (37.7700, -122.4437),  # Haight-Ashbury
                '94118': (37.7825, -122.4615),  # Inner Richmond
                '94121': (37.7767, -122.4943),  # Outer Richmond
                '94122': (37.7585, -122.4825),  # Sunset
                '94123': (37.8003, -122.4359),  # Marina
                '94124': (37.7327, -122.3907),  # Bayview
                '94127': (37.7357, -122.4578),  # St. Francis Wood
                '94130': (37.8221, -122.3698),  # Treasure Island
                '94131': (37.7423, -122.4372),  # Twin Peaks/Glen Park
                '94132': (37.7211, -122.4783),  # Lake Merced
                '94133': (37.8033, -122.4100),  # North Beach
                '94134': (37.7191, -122.4108),  # Visitacion Valley
                '94142': (37.7590, -122.4060),  # SoMa (PO)
                '94158': (37.7695, -122.3885),  # Mission Bay
                '94159': (37.7870, -122.4615),  # Presidio
            }
            
            # Get base coordinates for this zip, or default to downtown
            base_lat, base_lon = ZIP_COORDINATES.get(zip_code, (37.7749, -122.4194))
            
            # Add small random offset (within ~0.5 mile radius)
            lat = base_lat + np.random.uniform(-0.005, 0.005)
            lon = base_lon + np.random.uniform(-0.005, 0.005)
            
            record_id = len(records) + 1  # Start from 1
            records.append({
                'id': record_id,
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'price': price,
                'sqft': sqft,
                'bedrooms': bedrooms,
                'year_built': year_built,
                'zip_code': zip_code,
                'condition': condition,
                'days_on_market': int(np.clip(np.random.exponential(30), 1, 180)),
                'source': 'redfin_derived',
                'area_median_price': median_price,
                'area_median_ppsf': median_ppsf,
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} training records from Redfin data")
    return df


def load_or_generate_real_data(
    n_samples_per_zip: int = 50,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Load existing real data or generate from Redfin stats.
    
    Args:
        n_samples_per_zip: Samples per zip for generation.
        force_regenerate: Force regeneration even if data exists.
    
    Returns:
        DataFrame with training data.
    """
    cache_path = RAW_DATA_DIR / "redfin_training_data.csv"
    
    if cache_path.exists() and not force_regenerate:
        logger.info(f"Loading cached training data from {cache_path}")
        return pd.read_csv(cache_path, dtype={'zip_code': str})
    
    df = generate_training_data_from_redfin(n_samples_per_zip)
    
    if df is not None and len(df) > 0:
        df.to_csv(cache_path, index=False)
        logger.info(f"Saved training data to {cache_path}")
        return df
    
    logger.warning("Could not generate real data, returning None")
    return None


if __name__ == "__main__":
    # Test the loaders
    print("=" * 50)
    print("Testing Redfin Data Loaders")
    print("=" * 50)
    
    loader = RedfinDataLoader()
    
    try:
        sf_data = loader.load_sf_market_data()
        print(f"\nSF Market Data: {len(sf_data)} zip codes")
        print(sf_data[['zip_code', 'MEDIAN_SALE_PRICE', 'MEDIAN_PPSF']].head(10))
        
        # Get stats for a specific zip
        stats = loader.get_zip_market_stats('94107')
        print(f"\n94107 Stats: {stats}")
        
        # Generate training data
        print("\nGenerating training data...")
        train_df = generate_training_data_from_redfin(n_samples_per_zip=20)
        if train_df is not None:
            print(f"Generated {len(train_df)} training records")
            print(train_df[['zip_code', 'price', 'sqft', 'bedrooms', 'condition']].head(10))
            
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
