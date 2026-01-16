# S.P.E.C. Valuation Engine

**Spatial · Predictive · Explainable · Conversational**

A production-grade Automated Valuation Model (AVM) for residential real estate. Uses machine learning to predict property values, SHAP for explainability, and GPT-4 for AI-powered investment analysis.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat&logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)

---

## Key Results

| Metric | Value | Industry Standard |
|--------|-------|-------------------|
| **PPE10** (within ±10%) | **69.2%** | 50-60% |
| **MdAPE** (median error) | **8.1%** | 10-15% |
| **R²** (variance explained) | **0.78** | 0.65-0.75 |

> Model outperforms typical AVMs by 10-15% on accuracy metrics.

---

## Features

### 🎯 ML Valuation Engine
- XGBoost regression with Optuna hyperparameter tuning
- 17 engineered features including H3 spatial lags
- Isolation Forest outlier detection for training data

### 📊 Explainability (SHAP)
- Per-property feature attribution
- Interactive waterfall charts showing price drivers
- "Why is this property worth $X?" answered visually

### 🤖 AI Investment Memos
- GPT-4o-mini powered analysis
- Structured recommendations (BUY / HOLD / AVOID)
- Risk factors, market outlook, and comparable analysis

### 🗺️ Interactive Dashboard
- Property map with undervalued/overvalued markers
- Comparable properties finder (5 most similar)
- Real-time renovation simulator

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| ML/AI | XGBoost, SHAP, Optuna, OpenAI GPT-4 |
| Data | Pandas, SQLite, Parquet, Pandera |
| Geospatial | H3 Hexagonal Indexing, Walk Score API |
| Frontend | Streamlit, Plotly, Custom CSS |
| Infrastructure | Docker, MLflow, ChromaDB |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ColbyRReichenbach/S.P.E.C-Valuation.git
cd S.P.E.C-Valuation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys (optional for AI features)
cp .env.example .env
# Edit .env to add OPENAI_API_KEY

# Run the dashboard
streamlit run app.py
```

Open http://localhost:8501

---

## Project Structure

```
├── app.py              # Streamlit dashboard
├── config/settings.py  # Configuration & feature lists
├── src/
│   ├── model.py        # XGBoost + SHAP + Optuna
│   ├── etl.py          # Data pipeline + validation
│   ├── spatial.py      # H3 indexing + Comps finder
│   ├── oracle.py       # AI memos + RAG
│   └── ai_security.py  # Token limits, cost controls
├── data/               # Raw, processed, vector DB
└── docs/               # Model changelog, roadmap
```

---

## How It Works

1. **Data Ingestion** → Validate with Pandera, impute missing values
2. **Feature Engineering** → H3 spatial lags, distance to downtown, Walk Score
3. **Model Training** → XGBoost with Optuna (50 trials), MLflow tracking
4. **Prediction** → Real-time inference with SHAP explanations
5. **AI Analysis** → GPT-4 generates investment recommendations

---

## Sample Output

**Property: 954 sqft, 2 bed, 1917 built, 94114**

| Component | Value |
|-----------|-------|
| Market Baseline | $1,379,388 |
| Square Footage Impact | -$457,118 |
| H3 Neighborhood Premium | +$97,000 |
| **Model Prediction** | **$979,800** |
| AI Recommendation | HOLD |

---

## Roadmap

- [x] V1.0: Basic XGBoost model
- [x] V2.0: SHAP + MLflow + Docker
- [x] V3.0: AI Memos + Comps + H3 Spatial Features ← *Current*
- [ ] V4.0: Multi-persona AI, data expansion to 10K records

---

## Author

**Colby Reichenbach**  
[GitHub](https://github.com/ColbyRReichenbach) · [LinkedIn](https://linkedin.com/in/colbyreichenbach)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
