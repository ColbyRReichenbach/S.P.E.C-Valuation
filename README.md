# S.P.E.C. Valuation Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Spatial • Predictive • Explainable • Conversational**

A professional-grade Automated Valuation Model (AVM) dashboard for Real Estate Analysts. This tool bridges the gap between "Black Box" ML predictions and "White Box" explainability.

---

## 🎯 Overview

The S.P.E.C. Valuation Engine is an internal tool designed to help analysts validate AVM predictions with full transparency. It combines:

- **Spatial Analysis**: Interactive property maps with valuation overlays
- **Predictive Models**: XGBoost-powered price predictions
- **Explainability**: SHAP-based feature attribution (the "Why" behind predictions)
- **Conversational AI**: GPT-powered investment memo generation

---

## 🏗️ Architecture

```
spec-valuation-engine/
├── .gitignore               # Security: blocks secrets and data
├── .env.example             # Template for API keys
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── config/
│   └── settings.py          # Colors, paths, constants
├── data/
│   ├── raw/                 # Source CSVs (gitignored)
│   └── processed/           # Parquet & SQLite
├── src/
│   ├── __init__.py
│   ├── etl.py               # ETL pipeline
│   ├── model.py             # XGBoost + SHAP
│   ├── spatial.py           # Geospatial utilities
│   └── oracle.py            # AI/LLM integration
├── assets/
│   └── model.pkl            # Trained model (gitignored)
└── app.py                   # Streamlit frontend
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Navigate to project
cd spec-valuation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

For AI-powered investment memos:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

> ⚠️ **Note**: The app works without an API key (uses template memos instead).

### 3. Run the Application

```bash
streamlit run app.py
```

The app will:
1. Generate synthetic housing data (if none exists)
2. Train the XGBoost model (first run only)
3. Launch the dashboard at `http://localhost:8501`

---

## 📊 Features

### Market Pulse (Header)
Three key metrics computed via **raw SQL queries**:
- Average Days on Market
- Total Market Volume
- Average Price

SQL queries are viewable in expandable sections to demonstrate data skills.

### The Screener (Left Panel)
- **Filters**: Price range, Zip code, Valuation status
- **Interactive Map**: Property locations with valuation context
- **Property List**: Sortable, filterable property selection

### The Inspector (Right Panel)
- **Waterfall Chart**: SHAP-based price decomposition
- **Renovation Simulator**: Real-time value prediction as you adjust features
- **AI Memo**: GPT-generated bearish investment analysis

---

## 🔧 Technical Details

### Data Pipeline (ETL)
- Generates realistic synthetic SF Bay Area housing data
- Cleans column names to snake_case
- Persists to Parquet (fast loading) and SQLite (SQL queries)

### Valuation Model
- **Algorithm**: XGBoost Regressor
- **Features**: sqft, bedrooms, year_built, condition
- **Explainability**: SHAP TreeExplainer for feature attribution
- **Persistence**: Pickled model avoids retraining

### AI Oracle
- **Provider**: OpenAI GPT-4o-mini (configurable)
- **Persona**: Bearish Investment Analyst
- **Fallback**: Template-based memos when API unavailable
- **Context**: Mock news database per zip code

---

## 🎨 Design System

Finance-grade dark theme:

| Token | Color | Usage |
|-------|-------|-------|
| BG_PRIMARY | `#1A1D23` | Main background |
| BG_SECONDARY | `#22262E` | Cards, panels |
| EMERALD_GREEN | `#00D47E` | Profit, undervalued |
| CRIMSON_RED | `#FF4757` | Risk, overvalued |
| TEXT_PRIMARY | `#FFFFFF` | Headers |
| TEXT_SECONDARY | `#A0A4AB` | Body text |

---

## 📁 Data Schema

### `sales` Table (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique identifier |
| lat | REAL | Latitude |
| lon | REAL | Longitude |
| price | INTEGER | List price ($) |
| sqft | INTEGER | Square footage |
| bedrooms | INTEGER | Number of bedrooms |
| year_built | INTEGER | Construction year |
| zip_code | TEXT | Zip code |
| condition | INTEGER | 1-5 rating |
| days_on_market | INTEGER | Days listed |

---

## 🔒 Security

- API keys loaded from `.env` (never committed)
- `.gitignore` blocks secrets, data, and model files
- Graceful fallback when API unavailable

---

## 🧪 Testing

Run the ETL pipeline:
```bash
python -m src.etl
```

Test the model:
```bash
python -m src.model
```

Test the oracle:
```bash
python -m src.oracle
```

---

## 📈 Performance

- **Target**: Dashboard loads in < 3 seconds
- **Caching**: `@st.cache_data` for data, `@st.cache_resource` for model
- **Data Format**: Parquet for fast columnar access

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linting: `ruff check .`
5. Submit a pull request

---

Built with ❤️ for Real Estate Analysts
