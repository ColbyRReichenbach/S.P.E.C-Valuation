"""
S.P.E.C. Valuation Engine - Source Package
===========================================
Spatial, Predictive, Explainable, Conversational
Automated Valuation Model Dashboard

V2.0 Production-Grade Refactor
- Pandera data validation
- H3 Hexagonal Indexing
- Optuna hyperparameter optimization
- MLflow experiment tracking
- ChromaDB RAG vector store
"""

__version__ = "2.0.0"
__author__ = "S.P.E.C. Team"

# Module exports
from src.etl import (
    run_etl_pipeline,
    load_processed_data,
    execute_sql_query,
    validate_critical_constraints,
)

from src.model import (
    ValuationModel,
    get_model_instance,
    run_invariance_tests,
)

from src.spatial import (
    add_h3_index,
    add_distance_to_center,
    add_all_spatial_features,
    add_valuation_status,
    prepare_map_data,
)

from src.oracle import (
    generate_investment_memo,
    get_market_context,
    get_vector_store,
    ingest_market_report,
)

from src.connectors import (
    ZillowConnector,
    RedfinConnector,
    ZipCodeEnricher,
    UnifiedDataIngester,
    get_connector,
)

__all__ = [
    # ETL
    "run_etl_pipeline",
    "load_processed_data",
    "execute_sql_query",
    "validate_critical_constraints",
    # Model
    "ValuationModel",
    "get_model_instance",
    "run_invariance_tests",
    # Spatial
    "add_h3_index",
    "add_distance_to_center",
    "add_all_spatial_features",
    "add_valuation_status",
    "prepare_map_data",
    # Oracle
    "generate_investment_memo",
    "get_market_context",
    "get_vector_store",
    "ingest_market_report",
    # Connectors
    "ZillowConnector",
    "RedfinConnector",
    "ZipCodeEnricher",
    "UnifiedDataIngester",
    "get_connector",
]
