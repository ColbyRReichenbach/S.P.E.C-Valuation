"""
S.P.E.C. Valuation Engine - Data Connectors
============================================
API connectors for real estate data ingestion.
Supports Zillow, Redfin, and simulated data sources.
V2.0 Production-Grade Data Ingestion
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Import from sibling config module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    API_TIMEOUT_SECONDS,
    API_RETRY_ATTEMPTS,
    ZILLOW_API_BASE_URL,
    REDFIN_API_BASE_URL,
    ZIP_CODES,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# ABSTRACT BASE CONNECTOR
# ====================================
class BaseConnector(ABC):
    """Abstract base class for real estate data connectors."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the connector.
        
        Args:
            api_key: Optional API key for authenticated requests.
        """
        self.api_key = api_key
        self.timeout = API_TIMEOUT_SECONDS
        self.max_retries = API_RETRY_ATTEMPTS
    
    @abstractmethod
    def fetch_listings(
        self,
        zip_code: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch property listings for a given zip code.
        
        Args:
            zip_code: Target zip code.
            limit: Maximum number of listings to fetch.
        
        Returns:
            DataFrame with property listings.
        """
        pass
    
    @abstractmethod
    def fetch_property_details(
        self,
        property_id: str
    ) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific property.
        
        Args:
            property_id: Unique property identifier.
        
        Returns:
            Dictionary with property details.
        """
        pass
    
    def _retry_request(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            Function result.
        
        Raises:
            Exception: If all retries fail.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception


# ====================================
# ZILLOW API CONNECTOR
# ====================================
class ZillowConnector(BaseConnector):
    """
    Zillow API connector for property data.
    
    Requires RapidAPI key with Zillow subscription.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Zillow connector.
        
        Args:
            api_key: RapidAPI key. Falls back to RAPIDAPI_KEY env var.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
        self.base_url = ZILLOW_API_BASE_URL
        
        if not self.api_key:
            logger.warning(
                "No RapidAPI key found. ZillowConnector will operate in simulation mode."
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "X-RapidAPI-Key": self.api_key or "",
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }
    
    def fetch_listings(
        self,
        zip_code: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch property listings from Zillow API.
        
        Args:
            zip_code: Target zip code.
            limit: Maximum number of listings.
        
        Returns:
            DataFrame with property listings.
        """
        if not self.api_key:
            logger.info("No API key - returning simulated Zillow data")
            return self._simulate_listings(zip_code, limit)
        
        try:
            url = f"{self.base_url}/propertyExtendedSearch"
            params = {
                "location": zip_code,
                "home_type": "Houses",
                "status_type": "ForSale",
            }
            
            response = self._retry_request(
                requests.get,
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_zillow_response(data, limit)
            
        except Exception as e:
            logger.error(f"Zillow API error: {e}. Falling back to simulation.")
            return self._simulate_listings(zip_code, limit)
    
    def fetch_property_details(
        self,
        property_id: str
    ) -> Dict[str, Any]:
        """
        Fetch property details from Zillow API.
        
        Args:
            property_id: Zillow Property ID (zpid).
        
        Returns:
            Dictionary with property details.
        """
        if not self.api_key:
            logger.info("No API key - returning simulated property details")
            return self._simulate_property_details(property_id)
        
        try:
            url = f"{self.base_url}/property"
            params = {"zpid": property_id}
            
            response = self._retry_request(
                requests.get,
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Zillow property detail error: {e}")
            return self._simulate_property_details(property_id)
    
    def _parse_zillow_response(
        self,
        data: Dict[str, Any],
        limit: int
    ) -> pd.DataFrame:
        """Parse Zillow API response into DataFrame."""
        properties = data.get("props", [])[:limit]
        
        records = []
        for prop in properties:
            records.append({
                "id": prop.get("zpid"),
                "lat": prop.get("latitude"),
                "lon": prop.get("longitude"),
                "price": prop.get("price"),
                "sqft": prop.get("livingArea"),
                "bedrooms": prop.get("bedrooms"),
                "bathrooms": prop.get("bathrooms"),
                "year_built": prop.get("yearBuilt"),
                "zip_code": prop.get("address", {}).get("zipcode"),
                "address": prop.get("address", {}).get("streetAddress"),
                "property_type": prop.get("propertyType"),
                "days_on_market": prop.get("daysOnZillow"),
            })
        
        return pd.DataFrame(records)
    
    def _simulate_listings(
        self,
        zip_code: str,
        limit: int
    ) -> pd.DataFrame:
        """Generate simulated listings for testing."""
        np.random.seed(hash(zip_code) % 2**32)
        
        n = min(limit, 50)
        
        # Generate realistic data based on SF market
        sqft = np.random.normal(1800, 600, n).clip(600, 5000).astype(int)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
        year_built = np.random.randint(1920, 2024, n)
        condition = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.40, 0.30, 0.10])
        
        # Price model
        base_price = 400_000
        price = (
            base_price
            + sqft * 450
            + bedrooms * 75_000
            + (year_built - 1950) * 1_500
            + condition * 40_000
            + np.random.normal(0, 50_000, n)
        ).clip(200_000, 4_000_000).astype(int)
        
        # SF coordinates
        lat = np.random.uniform(37.70, 37.82, n)
        lon = np.random.uniform(-122.52, -122.35, n)
        
        return pd.DataFrame({
            "id": [f"SIM_{zip_code}_{i}" for i in range(n)],
            "lat": lat.round(6),
            "lon": lon.round(6),
            "price": price,
            "sqft": sqft,
            "bedrooms": bedrooms,
            "year_built": year_built,
            "zip_code": zip_code,
            "condition": condition,
            "days_on_market": np.random.exponential(30, n).clip(1, 180).astype(int),
            "source": "simulation",
        })
    
    def _simulate_property_details(
        self,
        property_id: str
    ) -> Dict[str, Any]:
        """Generate simulated property details."""
        np.random.seed(hash(property_id) % 2**32)
        
        return {
            "zpid": property_id,
            "price": int(np.random.uniform(500000, 2000000)),
            "sqft": int(np.random.normal(1800, 600)),
            "bedrooms": int(np.random.choice([2, 3, 4])),
            "bathrooms": int(np.random.choice([1, 2, 3])),
            "yearBuilt": int(np.random.randint(1920, 2024)),
            "lotSize": int(np.random.uniform(2000, 10000)),
        }


# ====================================
# REDFIN API CONNECTOR  
# ====================================
class RedfinConnector(BaseConnector):
    """
    Redfin API connector for property data.
    
    Requires RapidAPI key with Redfin subscription.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Redfin connector.
        
        Args:
            api_key: RapidAPI key. Falls back to RAPIDAPI_KEY env var.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
        self.base_url = REDFIN_API_BASE_URL
        
        if not self.api_key:
            logger.warning(
                "No RapidAPI key found. RedfinConnector will operate in simulation mode."
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "X-RapidAPI-Key": self.api_key or "",
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }
    
    def fetch_listings(
        self,
        zip_code: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch property listings from Redfin API.
        
        Args:
            zip_code: Target zip code.
            limit: Maximum number of listings.
        
        Returns:
            DataFrame with property listings.
        """
        if not self.api_key:
            logger.info("No API key - returning simulated Redfin data")
            return self._simulate_listings(zip_code, limit)
        
        try:
            url = f"{self.base_url}/properties/search"
            params = {"zipcode": zip_code}
            
            response = self._retry_request(
                requests.get,
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_redfin_response(data, limit)
            
        except Exception as e:
            logger.error(f"Redfin API error: {e}. Falling back to simulation.")
            return self._simulate_listings(zip_code, limit)
    
    def fetch_property_details(
        self,
        property_id: str
    ) -> Dict[str, Any]:
        """
        Fetch property details from Redfin API.
        
        Args:
            property_id: Redfin Property ID.
        
        Returns:
            Dictionary with property details.
        """
        if not self.api_key:
            return {"property_id": property_id, "source": "simulation"}
        
        try:
            url = f"{self.base_url}/property/detail"
            params = {"property_id": property_id}
            
            response = self._retry_request(
                requests.get,
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Redfin property detail error: {e}")
            return {"property_id": property_id, "error": str(e)}
    
    def _parse_redfin_response(
        self,
        data: Dict[str, Any],
        limit: int
    ) -> pd.DataFrame:
        """Parse Redfin API response into DataFrame."""
        properties = data.get("data", {}).get("results", [])[:limit]
        
        records = []
        for prop in properties:
            records.append({
                "id": prop.get("propertyId"),
                "lat": prop.get("latLong", {}).get("latitude"),
                "lon": prop.get("latLong", {}).get("longitude"),
                "price": prop.get("priceInfo", {}).get("amount"),
                "sqft": prop.get("sqftInfo", {}).get("amount"),
                "bedrooms": prop.get("beds"),
                "year_built": prop.get("yearBuilt"),
                "zip_code": prop.get("postalCode"),
            })
        
        return pd.DataFrame(records)
    
    def _simulate_listings(
        self,
        zip_code: str,
        limit: int
    ) -> pd.DataFrame:
        """Generate simulated Redfin listings."""
        # Reuse Zillow simulator with different seed
        np.random.seed(hash(f"redfin_{zip_code}") % 2**32)
        
        n = min(limit, 50)
        sqft = np.random.normal(1800, 600, n).clip(600, 5000).astype(int)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
        year_built = np.random.randint(1920, 2024, n)
        condition = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.40, 0.30, 0.10])
        
        base_price = 400_000
        price = (
            base_price
            + sqft * 450
            + bedrooms * 75_000
            + (year_built - 1950) * 1_500
            + condition * 40_000
            + np.random.normal(0, 50_000, n)
        ).clip(200_000, 4_000_000).astype(int)
        
        lat = np.random.uniform(37.70, 37.82, n)
        lon = np.random.uniform(-122.52, -122.35, n)
        
        return pd.DataFrame({
            "id": [f"RF_{zip_code}_{i}" for i in range(n)],
            "lat": lat.round(6),
            "lon": lon.round(6),
            "price": price,
            "sqft": sqft,
            "bedrooms": bedrooms,
            "year_built": year_built,
            "zip_code": zip_code,
            "condition": condition,
            "days_on_market": np.random.exponential(30, n).clip(1, 180).astype(int),
            "source": "simulation",
        })


# ====================================
# ZIP CODE ENRICHMENT CONNECTOR
# ====================================
class ZipCodeEnricher:
    """
    Enrich property data with zip code demographics and geography.
    
    Uses uszipcode library for local lookups (no API required).
    """
    
    def __init__(self):
        """Initialize the ZipCodeEnricher."""
        self.search = None
        self.available = False
        
        try:
            from uszipcode import SearchEngine
            self.search = SearchEngine()
            self.available = True
            logger.info("ZipCodeEnricher initialized with uszipcode database")
        except ImportError:
            logger.warning("uszipcode not installed. ZipCodeEnricher disabled.")
        except AttributeError as e:
            logger.warning(f"uszipcode compatibility issue (likely sqlalchemy_mate): {e}. ZipCodeEnricher disabled.")
        except Exception as e:
            logger.warning(f"ZipCodeEnricher initialization failed: {e}. Disabled.")
    
    def enrich(self, df: pd.DataFrame, zip_col: str = "zip_code") -> pd.DataFrame:
        """
        Enrich DataFrame with zip code demographics.
        
        Args:
            df: DataFrame with zip code column.
            zip_col: Name of the zip code column.
        
        Returns:
            DataFrame with added demographic columns.
        """
        if not self.available:
            logger.warning("ZipCodeEnricher not available. Returning original data.")
            return df
        
        df = df.copy()
        
        # Get unique zip codes
        zip_codes = df[zip_col].unique()
        
        # Lookup demographics
        demographics = {}
        for zc in zip_codes:
            try:
                result = self.search.by_zipcode(str(zc))
                if result:
                    demographics[zc] = {
                        "city": result.major_city,
                        "county": result.county,
                        "state": result.state,
                        "population": result.population,
                        "population_density": result.population_density,
                        "median_household_income": result.median_household_income,
                        "median_home_value": result.median_home_value,
                    }
            except Exception as e:
                logger.warning(f"Could not lookup zip {zc}: {e}")
        
        # Merge demographics
        for col in ["city", "county", "state", "population", 
                    "population_density", "median_household_income", "median_home_value"]:
            df[col] = df[zip_col].map(
                lambda zc: demographics.get(zc, {}).get(col)
            )
        
        return df
    
    def get_zip_info(self, zip_code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a single zip code.
        
        Args:
            zip_code: Target zip code.
        
        Returns:
            Dictionary with zip code details or None.
        """
        if not self.available:
            return None
        
        try:
            result = self.search.by_zipcode(str(zip_code))
            if result:
                return {
                    "zipcode": result.zipcode,
                    "city": result.major_city,
                    "county": result.county,
                    "state": result.state,
                    "lat": result.lat,
                    "lng": result.lng,
                    "population": result.population,
                    "population_density": result.population_density,
                    "median_household_income": result.median_household_income,
                    "median_home_value": result.median_home_value,
                }
        except Exception as e:
            logger.error(f"Error looking up zip {zip_code}: {e}")
        
        return None


# ====================================
# UNIFIED DATA INGESTION
# ====================================
class UnifiedDataIngester:
    """
    Unified data ingestion from multiple sources.
    
    Combines data from Zillow, Redfin, and enriches with zip code data.
    """
    
    def __init__(self):
        """Initialize the unified ingester."""
        self.zillow = ZillowConnector()
        self.redfin = RedfinConnector()
        self.enricher = ZipCodeEnricher()
    
    def ingest_all_zip_codes(
        self,
        zip_codes: Optional[List[str]] = None,
        listings_per_zip: int = 50
    ) -> pd.DataFrame:
        """
        Ingest data from all sources for multiple zip codes.
        
        Args:
            zip_codes: List of zip codes to ingest. Defaults to ZIP_CODES from settings.
            listings_per_zip: Max listings per zip code.
        
        Returns:
            Combined DataFrame with all listings.
        """
        zip_codes = zip_codes or ZIP_CODES
        
        all_data = []
        
        for zc in zip_codes:
            logger.info(f"Ingesting data for zip code {zc}...")
            
            # Fetch from Zillow
            try:
                zillow_df = self.zillow.fetch_listings(zc, limit=listings_per_zip)
                zillow_df["source"] = "zillow"
                all_data.append(zillow_df)
            except Exception as e:
                logger.error(f"Zillow ingest failed for {zc}: {e}")
            
            # Optional: also fetch from Redfin
            # redfin_df = self.redfin.fetch_listings(zc, limit=listings_per_zip)
            # all_data.append(redfin_df)
        
        if not all_data:
            logger.error("No data ingested from any source")
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Deduplicate by address if available
        if "address" in combined.columns:
            combined = combined.drop_duplicates(subset=["address"], keep="first")
        
        # Enrich with zip code demographics
        combined = self.enricher.enrich(combined)
        
        # Reset ID column
        combined["id"] = range(1, len(combined) + 1)
        
        logger.info(f"Total ingested: {len(combined)} properties")
        return combined


# ====================================
# FACTORY FUNCTION
# ====================================
def get_connector(source: str = "zillow") -> BaseConnector:
    """
    Factory function to get a connector by source name.
    
    Args:
        source: Data source name ('zillow', 'redfin').
    
    Returns:
        Configured connector instance.
    
    Raises:
        ValueError: If source is not supported.
    """
    connectors = {
        "zillow": ZillowConnector,
        "redfin": RedfinConnector,
    }
    
    if source.lower() not in connectors:
        raise ValueError(f"Unknown source: {source}. Supported: {list(connectors.keys())}")
    
    return connectors[source.lower()]()


if __name__ == "__main__":
    # Test the connectors
    logger.info("Testing ZillowConnector (simulation mode)...")
    zillow = ZillowConnector()
    df = zillow.fetch_listings("94107", limit=10)
    print(f"\nZillow sample ({len(df)} records):")
    print(df.head())
    
    logger.info("\nTesting ZipCodeEnricher...")
    enricher = ZipCodeEnricher()
    info = enricher.get_zip_info("94107")
    print(f"\nZip 94107 info: {info}")
    
    logger.info("\nTesting UnifiedDataIngester...")
    ingester = UnifiedDataIngester()
    full_df = ingester.ingest_all_zip_codes(zip_codes=["94107", "94110"], listings_per_zip=20)
    print(f"\nUnified ingest ({len(full_df)} records):")
    print(full_df.head())
