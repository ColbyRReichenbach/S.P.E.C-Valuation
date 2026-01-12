"""
S.P.E.C. Valuation Engine - Valuation Model
============================================
XGBoost-based property valuation with SHAP explainability.
The "Black Box" prediction meets "White Box" transparency.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    MODEL_PATH,
    MODEL_FEATURES,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
    ASSETS_DIR,
)
from src.etl import load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ValuationModel:
    """
    XGBoost-based Automated Valuation Model with SHAP explainability.
    
    Provides both "Black Box" predictions and "White Box" explanations
    for real estate property valuations.
    
    Attributes:
        model: Trained XGBRegressor instance.
        explainer: SHAP TreeExplainer for feature importance.
        features: List of feature names used for training.
        is_trained: Boolean indicating if model is ready for inference.
    """
    
    def __init__(self):
        """Initialize the ValuationModel."""
        self.model: Optional[XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.features: List[str] = MODEL_FEATURES
        self.is_trained: bool = False
        self._training_data: Optional[pd.DataFrame] = None
    
    def train(self, df: Optional[pd.DataFrame] = None, force: bool = False) -> "ValuationModel":
        """
        Train the XGBoost model on housing data.
        
        Args:
            df: Training DataFrame. If None, loads from processed data.
            force: If True, retrain even if saved model exists.
        
        Returns:
            Self for method chaining.
        """
        # Check for existing model
        if not force and MODEL_PATH.exists():
            logger.info(f"Loading existing model from {MODEL_PATH}")
            self.load()
            return self
        
        # Load data if not provided
        if df is None:
            df = load_processed_data()
        
        self._training_data = df.copy()
        
        # Prepare features and target
        X = df[self.features].copy()
        y = df[TARGET_COLUMN].copy()
        
        logger.info(f"Training XGBoost model on {len(X)} samples...")
        logger.info(f"Features: {self.features}")
        
        # Initialize and train model
        self.model = XGBRegressor(**XGBOOST_PARAMS)
        self.model.fit(X, y)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        self.is_trained = True
        logger.info("Model training complete.")
        
        # Save for future use
        self.save()
        
        return self
    
    def predict(
        self,
        sqft: float,
        bedrooms: int,
        year_built: int,
        condition: int,
    ) -> float:
        """
        Predict property price (Black Box output).
        
        Args:
            sqft: Square footage of the property.
            bedrooms: Number of bedrooms.
            year_built: Year the property was built.
            condition: Condition rating (1-5).
        
        Returns:
            Predicted price as float.
        
        Raises:
            RuntimeError: If model is not trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Create feature vector
        X = pd.DataFrame([{
            "sqft": sqft,
            "bedrooms": bedrooms,
            "year_built": year_built,
            "condition": condition,
        }])[self.features]
        
        prediction = self.model.predict(X)[0]
        return float(prediction)
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices for multiple properties.
        
        Args:
            df: DataFrame with feature columns.
        
        Returns:
            Array of predicted prices.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = df[self.features].copy()
        return self.model.predict(X)
    
    def explain(
        self,
        sqft: float,
        bedrooms: int,
        year_built: int,
        condition: int,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction (White Box output).
        
        Args:
            sqft: Square footage of the property.
            bedrooms: Number of bedrooms.
            year_built: Year the property was built.
            condition: Condition rating (1-5).
        
        Returns:
            Dictionary containing:
                - predicted_price: The model's prediction
                - base_value: Expected value (average prediction)
                - shap_values: Dict of feature -> SHAP contribution
                - feature_values: Dict of feature -> input value
        
        Raises:
            RuntimeError: If model is not trained.
        """
        if not self.is_trained or self.explainer is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Create feature vector
        X = pd.DataFrame([{
            "sqft": sqft,
            "bedrooms": bedrooms,
            "year_built": year_built,
            "condition": condition,
        }])[self.features]
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        
        # Handle different SHAP output formats
        if isinstance(shap_values, np.ndarray):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
        
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])
        else:
            base_value = float(base_value)
        
        # Create readable output
        shap_dict = {
            feature: float(shap_array[i])
            for i, feature in enumerate(self.features)
        }
        
        feature_dict = {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "year_built": year_built,
            "condition": condition,
        }
        
        return {
            "predicted_price": float(prediction),
            "base_value": base_value,
            "shap_values": shap_dict,
            "feature_values": feature_dict,
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from the model.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.features,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        return df
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            path: Output path (defaults to MODEL_PATH).
        
        Returns:
            Path to saved model.
        """
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "features": self.features,
            "is_trained": self.is_trained,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load(self, path: Optional[Path] = None) -> "ValuationModel":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to model file (defaults to MODEL_PATH).
        
        Returns:
            Self for method chaining.
        """
        path = path or MODEL_PATH
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.features = model_data["features"]
        self.is_trained = model_data["is_trained"]
        
        # Reinitialize SHAP explainer
        if self.model is not None:
            self.explainer = shap.TreeExplainer(self.model)
        
        logger.info(f"Model loaded from {path}")
        return self


def get_model_instance() -> ValuationModel:
    """
    Get a trained ValuationModel instance.
    
    Factory function that either loads an existing model or trains a new one.
    
    Returns:
        Trained ValuationModel instance.
    """
    model = ValuationModel()
    model.train()  # Will load if exists, train if not
    return model


if __name__ == "__main__":
    # Test the model
    model = get_model_instance()
    
    # Test prediction
    test_sqft = 1500
    test_bedrooms = 3
    test_year = 1980
    test_condition = 4
    
    price = model.predict(test_sqft, test_bedrooms, test_year, test_condition)
    print(f"\nPredicted Price: ${price:,.0f}")
    
    # Test explanation
    explanation = model.explain(test_sqft, test_bedrooms, test_year, test_condition)
    print(f"\nSHAP Explanation:")
    print(f"  Base Value: ${explanation['base_value']:,.0f}")
    for feature, value in explanation["shap_values"].items():
        print(f"  {feature}: ${value:+,.0f}")
