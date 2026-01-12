# S.P.E.C. Valuation Engine

**Spatial · Predictive · Explainable · Conversational**

A full-stack Automated Valuation Model (AVM) for residential real estate, featuring machine learning price prediction, SHAP-based explainability, and LLM-powered investment analysis.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-6C5CE7?style=flat)

---

## Project Overview

This project demonstrates an end-to-end data science workflow: from ETL pipeline and feature engineering to model training, interpretability, and deployment as an interactive dashboard. The system identifies undervalued and overvalued properties by comparing list prices against ML predictions, with full transparency into the pricing drivers.

**Key Capabilities:**
- XGBoost regression model for property valuation
- SHAP TreeExplainer for feature attribution and model transparency
- Interactive geospatial visualization with color-coded valuation status
- Real-time "what-if" renovation simulator
- LLM-generated investment memos with market context

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| Machine Learning | XGBoost, scikit-learn, SHAP |
| Data Processing | Pandas, NumPy, SQLite, Parquet |
| Visualization | Plotly, Pydeck, Streamlit |
| LLM Integration | OpenAI API (GPT-4o-mini) |
| Frontend | Streamlit with custom CSS |

---

## Architecture

```
spec-valuation-engine/
├── app.py                 # Streamlit dashboard application
├── config/
│   └── settings.py        # Configuration and constants
├── src/
│   ├── etl.py             # Data pipeline (extract, transform, load)
│   ├── model.py           # XGBoost training + SHAP explainability
│   ├── spatial.py         # Geospatial utilities
│   └── oracle.py          # LLM integration for investment memos
├── data/
│   └── processed/         # Parquet and SQLite outputs
└── assets/
    └── model.pkl          # Serialized trained model
```

---

## Data Pipeline

The ETL module generates synthetic San Francisco Bay Area housing data for demonstration, then processes it through a standard pipeline:

1. **Extract**: Load raw property records
2. **Transform**: Normalize column names, handle missing values, engineer features
3. **Load**: Persist to Parquet (analytics) and SQLite (SQL queries)

Data schema includes: `lat`, `lon`, `price`, `sqft`, `bedrooms`, `year_built`, `condition`, `zip_code`, `days_on_market`

---

## Model Training & Explainability

**Algorithm**: XGBoost Regressor with hyperparameter tuning

**Features**:
- Square footage
- Bedroom count
- Year built
- Condition rating (1-5)

**Explainability**: SHAP TreeExplainer computes feature attributions for each prediction, enabling a waterfall visualization that breaks down how each feature contributes to the final price estimate.

```python
# Example: Get prediction with explanation
model = ValuationModel()
explanation = model.explain(sqft=2000, bedrooms=3, year_built=1985, condition=4)
# Returns: base_value, shap_values per feature, predicted_price
```

---

## Dashboard Features

| Section | Description |
|---------|-------------|
| **Market Pulse** | Aggregate metrics via SQL queries (avg days on market, total volume, avg price) |
| **Property Screener** | Filter by price, zip code, valuation status; interactive map with color-coded markers |
| **Valuation Breakdown** | SHAP waterfall chart showing price decomposition |
| **Renovation Simulator** | Adjust property features and see real-time value predictions |
| **Investment Analysis** | AI-generated memo with market context, risk factors, and recommendation |

---

## Getting Started

```bash
# Clone and navigate to project
cd spec-valuation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The first run will generate synthetic data and train the model. Subsequent runs use cached artifacts.

**Optional**: Add your OpenAI API key to `.env` for AI-powered investment memos:
```
OPENAI_API_KEY=sk-...
```

---

## Performance Optimizations

- **Caching**: Streamlit's `@st.cache_data` and `@st.cache_resource` decorators minimize redundant computation
- **Data Format**: Parquet for fast columnar reads
- **Model Persistence**: Pickled model avoids retraining on each session

---

## License

MIT License
