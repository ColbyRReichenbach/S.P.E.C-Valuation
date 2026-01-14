"""
S.P.E.C. Valuation Engine - Valuation Model
============================================
XGBoost-based property valuation with SHAP explainability.
V2.0 with Optuna hyperparameter optimization and MLflow tracking.
"""

import pickle
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    MODEL_PATH,
    MODEL_FEATURES,
    TARGET_COLUMN,
    XGBOOST_PARAMS,
    ASSETS_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
    OPTUNA_SEARCH_SPACE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_DIR,
)
from src.etl import load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# MLFLOW UTILITIES
# ====================================
def _get_mlflow():
    """Lazy import of mlflow."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return None


def _setup_mlflow():
    """Configure MLflow tracking."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return None
    
    try:
        # Create mlruns directory
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set tracking URI
        mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        logger.info(f"MLflow configured. Tracking URI: {MLFLOW_TRACKING_URI}")
        return mlflow
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return None


# ====================================
# OPTUNA UTILITIES
# ====================================
def _get_optuna():
    """Lazy import of optuna."""
    try:
        import optuna
        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError:
        logger.warning("Optuna not installed. Hyperparameter tuning disabled.")
        return None


# ====================================
# VALUATION MODEL CLASS
# ====================================
class ValuationModel:
    """
    XGBoost-based Automated Valuation Model with SHAP explainability.
    
    V2.0 Features:
    - Optuna hyperparameter optimization
    - MLflow experiment tracking
    - Cross-validation metrics
    - SHAP summary plots
    
    Attributes:
        model: Trained XGBRegressor instance.
        explainer: SHAP TreeExplainer for feature importance.
        features: List of feature names used for training.
        is_trained: Boolean indicating if model is ready for inference.
        best_params: Optimized hyperparameters from Optuna.
        metrics: Training metrics (RMSE, MAE, R²).
    """
    
    def __init__(self):
        """Initialize the ValuationModel."""
        self.model: Optional[XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.features: List[str] = MODEL_FEATURES
        self.is_trained: bool = False
        self.best_params: Optional[Dict[str, Any]] = None
        self.metrics: Dict[str, float] = {}
        self._training_data: Optional[pd.DataFrame] = None
        self._mlflow = None
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = OPTUNA_N_TRIALS,
        timeout: int = OPTUNA_TIMEOUT_SECONDS
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Searches over:
        - learning_rate: 0.01 - 0.3 (log scale)
        - max_depth: 3 - 10
        - subsample: 0.6 - 1.0
        - n_estimators: 50 - 300
        - min_child_weight: 1 - 10
        - colsample_bytree: 0.5 - 1.0
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            n_trials: Number of optimization trials.
            timeout: Maximum optimization time in seconds.
        
        Returns:
            Best hyperparameters dictionary.
        """
        optuna = _get_optuna()
        
        if optuna is None:
            logger.warning("Optuna not available. Using default parameters.")
            return XGBOOST_PARAMS
        
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    OPTUNA_SEARCH_SPACE["learning_rate"]["low"],
                    OPTUNA_SEARCH_SPACE["learning_rate"]["high"],
                    log=OPTUNA_SEARCH_SPACE["learning_rate"].get("log", False)
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    OPTUNA_SEARCH_SPACE["max_depth"]["low"],
                    OPTUNA_SEARCH_SPACE["max_depth"]["high"]
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    OPTUNA_SEARCH_SPACE["subsample"]["low"],
                    OPTUNA_SEARCH_SPACE["subsample"]["high"]
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    OPTUNA_SEARCH_SPACE["n_estimators"]["low"],
                    OPTUNA_SEARCH_SPACE["n_estimators"]["high"]
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    OPTUNA_SEARCH_SPACE["min_child_weight"]["low"],
                    OPTUNA_SEARCH_SPACE["min_child_weight"]["high"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    OPTUNA_SEARCH_SPACE["colsample_bytree"]["low"],
                    OPTUNA_SEARCH_SPACE["colsample_bytree"]["high"]
                ),
                "random_state": 42,
                "n_jobs": -1,
            }
            
            model = XGBRegressor(**params)
            
            # 5-fold cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            
            return -scores.mean()  # Return RMSE
        
        # Create study
        study = optuna.create_study(direction="minimize")
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
        except Exception as e:
            logger.error(f"Optuna optimization error: {e}")
            return XGBOOST_PARAMS
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_params["random_state"] = 42
        self.best_params["n_jobs"] = -1
        
        logger.info(f"Optimization complete. Best RMSE: ${study.best_value:,.0f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        force: bool = False,
        optimize: bool = False,
        n_trials: int = OPTUNA_N_TRIALS
    ) -> "ValuationModel":
        """
        Train the XGBoost model with MLflow tracking.
        
        Args:
            df: Training DataFrame. If None, loads from processed data.
            force: If True, retrain even if saved model exists.
            optimize: If True, run Optuna hyperparameter optimization.
            n_trials: Number of Optuna trials if optimize=True.
        
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
        
        # Train/test split for validation metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training XGBoost model on {len(X_train)} samples...")
        logger.info(f"Features: {self.features}")
        
        # Setup MLflow
        self._mlflow = _setup_mlflow()
        
        # Hyperparameter optimization
        if optimize:
            params = self.optimize_hyperparameters(X_train, y_train, n_trials=n_trials)
        else:
            params = self.best_params or XGBOOST_PARAMS
        
        # Start MLflow run
        if self._mlflow:
            run = self._mlflow.start_run(
                run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        try:
            # Log parameters
            if self._mlflow:
                self._mlflow.log_params(params)
                self._mlflow.log_param("n_samples", len(X_train))
                self._mlflow.log_param("n_features", len(self.features))
                self._mlflow.log_param("features", str(self.features))
            
            # Initialize and train model
            self.model = XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            self.metrics = {
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
            }
            
            logger.info(f"Training Metrics:")
            logger.info(f"  RMSE (test): ${self.metrics['test_rmse']:,.0f}")
            logger.info(f"  MAE (test): ${self.metrics['test_mae']:,.0f}")
            logger.info(f"  R² (test): {self.metrics['test_r2']:.4f}")
            
            # Log metrics to MLflow
            if self._mlflow:
                self._mlflow.log_metrics(self.metrics)
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            # Generate and log SHAP summary plot
            if self._mlflow:
                self._log_shap_plot(X_test)
                self._log_model_artifact()
            
            self.is_trained = True
            logger.info("Model training complete.")
            
        finally:
            if self._mlflow:
                self._mlflow.end_run()
        
        # Save for future use
        self.save()
        
        return self
    
    def _log_shap_plot(self, X: pd.DataFrame) -> None:
        """Generate and log SHAP summary plot to MLflow."""
        if self._mlflow is None or self.explainer is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Sample data for SHAP (max 500 for performance)
            X_sample = X.sample(min(500, len(X)), random_state=42)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, X_sample,
                show=False,
                plot_type="bar"
            )
            plt.tight_layout()
            
            # Save to file
            plot_path = ASSETS_DIR / "shap_summary.png"
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            # Log to MLflow
            self._mlflow.log_artifact(str(plot_path))
            logger.info(f"SHAP summary plot logged to MLflow")
            
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {e}")
    
    def _log_model_artifact(self) -> None:
        """Log trained model to MLflow."""
        if self._mlflow is None or self.model is None:
            return
        
        try:
            # Log the XGBoost model
            self._mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                registered_model_name="spec_valuation_model"
            )
            logger.info("Model artifact logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")
    
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
            "best_params": self.best_params,
            "metrics": self.metrics,
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
        self.best_params = model_data.get("best_params")
        self.metrics = model_data.get("metrics", {})
        
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


# ====================================
# INVARIANCE TESTS FOR MODEL
# ====================================
def run_invariance_tests(model: ValuationModel) -> Dict[str, bool]:
    """
    Run invariance tests on the model.
    
    Tests that increasing sqft should not decrease price (all else equal).
    
    Args:
        model: Trained ValuationModel.
    
    Returns:
        Dictionary with test results.
    """
    results = {}
    
    # Base case
    base_price = model.predict(sqft=1500, bedrooms=3, year_built=1980, condition=3)
    
    # Test: Increasing sqft should not decrease price
    larger_price = model.predict(sqft=2000, bedrooms=3, year_built=1980, condition=3)
    results["sqft_monotonicity"] = larger_price >= base_price
    
    # Test: Better condition should not decrease price
    better_condition_price = model.predict(sqft=1500, bedrooms=3, year_built=1980, condition=5)
    results["condition_monotonicity"] = better_condition_price >= base_price
    
    # Test: More bedrooms should not decrease price (generally)
    more_bedrooms_price = model.predict(sqft=1500, bedrooms=4, year_built=1980, condition=3)
    results["bedroom_monotonicity"] = more_bedrooms_price >= base_price
    
    # Log results
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Invariance Test - {test}: {status}")
    
    return results


if __name__ == "__main__":
    # Test the model
    print("=" * 50)
    print("S.P.E.C. Valuation Model v2.0")
    print("=" * 50)
    
    # Train with optimization (set optimize=True for Optuna)
    model = ValuationModel()
    model.train(force=True, optimize=False)  # Set optimize=True for full optimization
    
    # Print metrics
    print(f"\nTraining Metrics:")
    for metric, value in model.metrics.items():
        print(f"  {metric}: {value:,.2f}")
    
    # Test prediction
    test_sqft = 1500
    test_bedrooms = 3
    test_year = 1980
    test_condition = 4
    
    price = model.predict(test_sqft, test_bedrooms, test_year, test_condition)
    print(f"\nTest Prediction: ${price:,.0f}")
    
    # Test explanation
    explanation = model.explain(test_sqft, test_bedrooms, test_year, test_condition)
    print(f"\nSHAP Explanation:")
    print(f"  Base Value: ${explanation['base_value']:,.0f}")
    for feature, value in explanation["shap_values"].items():
        print(f"  {feature}: ${value:+,.0f}")
    
    # Run invariance tests
    print(f"\nInvariance Tests:")
    test_results = run_invariance_tests(model)
