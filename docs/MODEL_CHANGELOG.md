# S.P.E.C. Valuation Engine — Model Performance Changelog
**Purpose:** Track model evolution, performance metrics, and the rationale behind each change.  
**Last Updated:** January 15, 2026

---

## Performance Summary (Latest)

| Version | Date | R² | RMSE | MAE | PPE10 | PPE20 | MdAPE | Features | Records |
|---------|------|-----|------|-----|-------|-------|-------|----------|---------|
| V1.0 | 2026-01-12 | 0.85 | $180K | $140K | N/A | N/A | N/A | 4 | Synthetic |
| V2.0 | 2026-01-14 | 0.74 | $352K | $266K | N/A | N/A | N/A | 4 | 1,400 |
| V2.1 | 2026-01-14 | 0.68 | $382K | $277K | N/A | N/A | N/A | 4 | 1,400 |
| V2.2 | 2026-01-14 | 0.898 | $214K | $164K | N/A | N/A | N/A | 10 | 1,400 |
| V3.0-a | 2026-01-15 | 0.898 | $214K | $164K | 50.7% | 80.0% | 9.8% | 10 | 1,400 |
| V3.0-b | 2026-01-15 | 0.893 | $193K | $145K | 55.6% | 85.3% | 8.8% | 10 | 1,330 |
| **V3.0-c** | 2026-01-15 | **0.937** | **$148K** | **$111K** | **69.2%** 🚀 | **92.5%** | **6.5%** | **14** | 1,330 |

---

## Detailed Version History

---

### V1.0 — Initial Prototype
**Date:** January 12, 2026  
**Branch:** `main` (initial)

#### What We Built
- Basic XGBoost regressor with 4 features
- Synthetic/simulated housing data
- Proof of concept Streamlit dashboard

#### Features Used
```
sqft, bedrooms, year_built, condition
```

#### Performance
| Metric | Value |
|--------|-------|
| R² | 0.85 |
| RMSE | $180,000 |
| MAE | $140,000 |

#### Notes
- High R² was misleading — synthetic data was "too clean"
- No real-world variance or outliers
- Model was overfitting to artificial patterns

---

### V2.0 — Real Market Data
**Date:** January 14, 2026  
**Branch:** `v2`

#### What Changed
- **Data Source:** Replaced synthetic data with real Redfin listings (1,400 records)
- **Geography:** San Francisco Bay Area
- **Quality:** Added Pandera validation schema

#### Why We Changed It
Synthetic data cannot model real-world non-linearities like:
- Location premiums
- Condition variance
- Market timing effects

#### Features Used
```
sqft, bedrooms, year_built, condition
```

#### Performance
| Metric | Value | Δ from V1.0 |
|--------|-------|-------------|
| R² | 0.74 | -13% |
| RMSE | $352,000 | +96% |
| MAE | $266,000 | +90% |

#### Notes
- Performance dropped significantly because real data has variance
- This is expected — real-world modeling is harder than toy problems
- Model needed more features to capture location effects

---

### V2.1 — Full ETL Pipeline
**Date:** January 14, 2026  
**Branch:** `v2`

#### What Changed
- Implemented complete ETL pipeline (`src/etl.py`)
- Added data validation with Pandera schemas
- Proper train/test split

#### Why We Changed It
Production systems need:
- Reproducible data processing
- Data quality guarantees
- Audit trails

#### Features Used
```
sqft, bedrooms, year_built, condition
```

#### Performance
| Metric | Value | Δ from V2.0 |
|--------|-------|-------------|
| R² | 0.68 | -8% |
| RMSE | $382,000 | +9% |
| MAE | $277,000 | +4% |

#### Notes
- Slight performance drop due to stricter data validation
- Some "too good to be true" outliers were removed
- This is a more honest baseline for improvement

---

### V2.2 — Spatial Feature Engineering
**Date:** January 14, 2026  
**Branch:** `v2`

#### What Changed
1. **New Derived Features:**
   - `property_age`: 2025 - year_built (more intuitive than raw year)
   - `sqft_per_bedroom`: Space efficiency metric
   - `is_newer_construction`: Boolean for year_built >= 2000

2. **New Spatial Features:**
   - `distance_to_downtown_km`: Haversine distance to SF city center
   - `distance_to_nearest_bart_km`: Distance to closest BART station
   - `neighborhood_price_tier`: Quintile ranking (1-5) based on zip code median price

#### Why We Changed It
Real estate valuation is fundamentally about **location**. The 4 basic features couldn't capture:
- The premium for transit access (BART proximity)
- The premium for central locations
- Neighborhood quality differences

#### How We Implemented It
```python
# BART stations added to config/settings.py
BART_STATIONS_SF = [
    ("Embarcadero", 37.7929, -122.3968),
    ("Montgomery", 37.7894, -122.4013),
    # ... 6 more stations
]

# Distance calculation in src/spatial.py
def calculate_distance_to_nearest_bart(lat, lon):
    distances = [haversine_distance(lat, lon, s[1], s[2]) for s in BART_STATIONS_SF]
    return min(distances)
```

#### Features Used (10 total)
```
# Core (4)
sqft, bedrooms, year_built, condition

# Derived (3)
property_age, sqft_per_bedroom, is_newer_construction

# Spatial (3)
distance_to_downtown_km, distance_to_nearest_bart_km, neighborhood_price_tier
```

#### Performance
| Metric | Value | Δ from V2.1 |
|--------|-------|-------------|
| R² | **0.898** | **+32%** |
| RMSE | $214,000 | **-44%** |
| MAE | $164,000 | **-41%** |

#### Key Insights
- Adding location-aware features increased R² by 32%
- BART proximity was a strong predictor (SF residents pay premium for transit)
- Neighborhood tier captured "comparable sales" logic

---

### V3.0 — Institutional Metrics & Data Quality
**Date:** January 15, 2026  
**Branch:** `v3`

#### What Changed
1. **New Metrics (Phase 3A.1):**
   - `PPE10`: Percentage of Predictions within ±10% error
   - `PPE20`: Percentage of Predictions within ±20% error
   - `MdAPE`: Median Absolute Percentage Error

2. **Outlier Detection (Phase 3A.2):**
   - Isolation Forest for anomaly detection
   - Automated flagging of distressed sales and data errors
   - Removed 70 outliers (5%) from training data

#### Why We Changed It
**PPE10 is the industry gold standard for AVMs.** Institutional investors don't care about RMSE (which is skewed by luxury homes). They care about:
> "What percentage of predictions are close enough to act on?"

**Outlier removal** prevents extreme luxury homes ($3M+ sales) from skewing the model toward overestimating standard properties.

#### How We Implemented It
```python
# Phase 3A.1: Institutional Metrics
def calculate_ppe(y_true, y_pred, threshold=0.10):
    """Calculate percentage of predictions within threshold."""
    ape = np.abs((y_true - y_pred) / y_true)
    return (np.sum(ape <= threshold) / len(ape)) * 100

def calculate_mdape(y_true, y_pred):
    """Calculate median absolute percentage error."""
    ape = np.abs((y_true - y_pred) / y_true) * 100
    return np.median(ape)

# Phase 3A.2: Isolation Forest
from sklearn.ensemble import IsolationForest

def detect_outliers(df, contamination=0.05):
    """Flag anomalous properties using unsupervised ML."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    features = ["price", "sqft", "bedrooms", "price_per_sqft"]
    df["is_outlier"] = iso_forest.fit_predict(df[features]) == -1
    return df
```

#### Performance Comparison

| Metric | Before Outlier Removal | After Outlier Removal | Change |
|--------|------------------------|----------------------|--------|
| Records | 1,400 | 1,330 | -70 (5%) |
| R² | 0.898 | 0.893 | -0.6% |
| RMSE | $214,000 | **$193,000** | **-10%** |
| MAE | $164,000 | **$145,000** | **-12%** |
| **PPE10** | 50.7% | **55.6%** | **+9.7%** |
| **PPE20** | 80.0% | **85.3%** | **+6.6%** |
| **MdAPE** | 9.8% | **8.8%** | **-10%** |

#### Key Insights
- Removing 70 outliers improved PPE10 by ~10% (from 50.7% to 55.6%)
- RMSE dropped $21K because extreme luxury errors no longer dominate
- Slight R² decrease is a trade-off for better reliability metrics
- Flagged outliers saved to `data/processed/outliers.csv` for manual review

---

### V3.0 Phase 3B — H3 Spatial Lag Features
**Date:** January 15, 2026  
**Branch:** `v3`

#### What Changed
Added 4 new H3-based "comparable sales" features that capture neighborhood price context:
1. `h3_median_ppsf`: Median $/sqft in the same H3 hexagonal cell
2. `h3_neighbor_median_ppsf`: Weighted median $/sqft of 6 adjacent cells
3. `h3_listing_density`: Number of active listings in the cell
4. `h3_price_percentile`: Where this property ranks (0-100) among neighbors

#### Why We Changed It
Professional appraisers use "comparable sales" — nearby similar properties — to justify valuations.
H3 hexagonal cells (~0.1 km²) naturally capture this logic:
- A $2M home in a $3M neighborhood might be undervalued
- A $1.5M home in a $1M neighborhood might be overvalued

#### How We Implemented It
```python
def calculate_h3_spatial_lags(df):
    # Group by H3 cell, calculate aggregates
    h3_stats = df.groupby("h3_index").agg({
        "price_per_sqft": ["median", "count"],
    })
    
    # For each property, get stats from 6 adjacent cells
    def get_neighbor_median_ppsf(h3_idx):
        neighbors = h3.grid_disk(h3_idx, 1)  # 6 adjacent cells
        return weighted_median(neighbors)
    
    df["h3_neighbor_median_ppsf"] = df["h3_index"].apply(get_neighbor_median_ppsf)
```

#### Performance Comparison

| Metric | V3.0-b (10 features) | V3.0-c (14 features) | Change |
|--------|----------------------|----------------------|--------|
| Features | 10 | 14 | +4 |
| R² | 0.893 | **0.937** | **+4.9%** |
| RMSE | $193,000 | **$148,000** | **-23%** |
| MAE | $145,000 | **$111,000** | **-24%** |
| **PPE10** | 55.6% | **69.2%** | **+24%** 🚀 |
| **PPE20** | 85.3% | **92.5%** | **+8.4%** |
| **MdAPE** | 8.8% | **6.5%** | **-26%** |

#### Key Insights
- This was the **single largest improvement** in model history
- PPE10 jumped from 55.6% to 69.2% — nearly 70% of predictions within ±10%!
- R² exceeded 0.93 for the first time
- The "comparable sales" logic is critical for real estate valuation

---

### V3.0 Phase 3B.2 — Walk Score API Integration
**Date:** January 15, 2026  
**Branch:** `v3`

#### What Changed
Added 3 new walkability features from Walk Score API:
1. `walk_score`: How walkable is the location? (0-100)
2. `transit_score`: Public transit access quality (0-100)
3. `bike_score`: How bikeable is the location? (0-100)

#### Why We Changed It
Walkability is a significant price factor, especially in urban markets like SF:
- Walk Score 90+ commands 5-10% premium
- Transit access reduces need for parking
- Millennial/Gen-Z buyers prioritize walkable neighborhoods

#### How We Implemented It
```python
# src/external_apis.py
def fetch_walkscore(lat, lon, api_key):
    response = requests.get(
        "https://api.walkscore.com/score",
        params={"lat": lat, "lon": lon, "transit": 1, "bike": 1}
    )
    return {
        "walk_score": response.json()["walkscore"],
        "transit_score": response.json()["transit"]["score"],
        "bike_score": response.json()["bike"]["score"],
    }

# Simulation fallback when no API key
def simulate_walkscore_features(df):
    # Uses distance-to-downtown as proxy for walkability
    df["walk_score"] = 100 - (df["distance_to_downtown_km"] * 8)
```

#### Performance (With Simulated Data)
*Note: Using simulated walkability scores based on distance-to-downtown. Real API would provide more unique signal.*

| Metric | V3.0-c (14 features) | V3.0-d (17 features) | Change |
|--------|----------------------|----------------------|--------|
| Features | 14 | 17 | +3 |
| R² | 0.937 | 0.935 | -0.2% |
| RMSE | $148,000 | $150,000 | +1.3% |
| **PPE10** | 69.2% | 69.2% | — |
| **MdAPE** | 6.5% | 6.2% | -4.6% |

#### Key Insights
- Simulated scores show marginal impact (derived from existing feature)
- Real Walk Score API would provide unique signal not captured by distance
- Infrastructure is ready — just add `WALKSCORE_API_KEY` to `.env`
- Free API key at: https://www.walkscore.com/professional/api-sign-up.php

---

### V3.0 Phase 3E — AI Engineering & Security
**Date:** January 15, 2026  
**Branch:** `v3`

#### What Changed
Implemented comprehensive AI security controls in `src/ai_security.py`:

| Feature | Implementation |
|---------|----------------|
| **3E.1** Prompt Injection Prevention | XML tag delimiting, input sanitization |
| **3E.2** Token-Aware Truncation | tiktoken for accurate counting |
| **3E.3** Structured JSON Outputs | Schema validation for LLM responses |
| **3E.4** Recursive Text Splitting | Paragraph → sentence → word splitting |
| **3E.5** Token Limits & Cost Controls | Model-specific limits, cost estimation |
| **3E.6** API Error Handling | Exponential backoff retry logic |
| **3E.7** Security Logging | SQLite audit trail for all AI calls |

#### Why We Changed It
Production AI systems need defense-in-depth:
- **Prompt Injection**: Users can't manipulate the AI via property descriptions
- **Token Limits**: Prevent runaway costs ($0.01 → $100 accidents)
- **Audit Trail**: Track costs, success rates, and debug failures

#### How We Implemented It
```python
# 3E.1: XML tag delimiting prevents injection
prompt = create_secure_prompt(
    system_instructions="You are a bearish analyst",
    property_data={"address": "ignore previous... [SANITIZED]"},
    market_context=truncate_to_tokens(rag_context, 2000),
    task="Write investment memo"
)

# 3E.2: Accurate token counting with tiktoken
tokens = count_tokens(prompt, model="gpt-4o-mini")  # → 174

# 3E.5: Budget validation before API call
is_valid, reason = validate_request_budget(prompt, "gpt-4o-mini")
# → True, "OK (174/8000 tokens)"

# 3E.7: Audit logging
log_ai_interaction(
    property_id=123,
    model="gpt-4o-mini",
    input_tokens=174,
    cost_estimate=0.00026,
    success=True,
)
```

#### Key Features
- **Malicious input sanitization**: Patterns like "ignore previous instructions" are replaced with `[REDACTED]`
- **Cost estimation**: gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
- **Daily cost tracking**: `get_ai_usage_stats()` returns 7-day summary
- **Retry with backoff**: Handles rate limits, timeouts gracefully

---

## Upcoming Changes (Planned)

### V3.0 ✅ Complete
- [x] Phase 3A.1: PPE10/PPE20/MdAPE Metrics
- [x] Phase 3A.2: Isolation Forest Outlier Detection
- [x] Phase 3B.1: H3 Spatial Lag Features
- [x] Phase 3B.2: Walk Score Integration (simulation, awaiting API key)
- [x] Phase 3C: AI Investment Memos (GPT-4o-mini)
- [x] Phase 3D: Comparable Properties (Comps)
- [x] Phase 3E: AI Security & Cost Controls

### V3.1 Hotfixes (Quick Wins)
- [ ] **Outlier Export**: Save removed outliers to `data/processed/outliers.parquet` with reasons
- [ ] **Walk Score API Key**: Integrate live API once key is received

### V4.0 (Major Features)
- [ ] **Data Expansion**: Scale from 1,400 to 5,000-10,000 records
- [ ] **Map Clustering**: Required for 10K+ markers (Deck.gl clustering)
- [ ] **Cascading Model Architecture**:
  - Random Forest classifier to predict property tier (Luxury / Mid / Entry)
  - 3 separate XGBoost models, one per tier
  - Segment-specific SHAP baselines (e.g., Luxury baseline = $2.8M, not $1.4M)
- [ ] **AI Persona Selector**:
  - Bearish 🐻 / Neutral ⚖️ / Bullish 🐂 analyst personas
  - User-selectable OpenAI model (GPT-4o-mini, GPT-4o, GPT-4.1)
  - Transparency widget: "Using Bearish persona: 15% BUY | 45% HOLD | 40% AVOID"

### V5.0 (Web Application Rewrite)
- [ ] **React/Next.js Frontend**: Replace Streamlit for smoother UX
- [ ] **FastAPI Backend**: Separate model inference API
- [ ] **No Full-Page Reloads**: Real-time updates via WebSocket or SSE
- [ ] **Production Hosting**: Vercel (frontend) + GCP Cloud Run (backend)

> **Why V5.0?** Streamlit re-renders the entire page on every interaction (~300-500ms latency). 
> For a truly smooth experience, a React frontend with a FastAPI backend is the right architecture.
> Streamlit is excellent for prototyping; React is for production UX.

---

## Glossary of Metrics

| Metric | Full Name | What It Tells You |
|--------|-----------|-------------------|
| **R²** | Coefficient of Determination | How much variance the model explains (0-1) |
| **RMSE** | Root Mean Squared Error | Average prediction error in dollars (penalizes large errors) |
| **MAE** | Mean Absolute Error | Average prediction error in dollars (simpler than RMSE) |
| **PPE10** | Predictions within 10% | % of predictions within ±10% of actual (industry standard) |
| **PPE20** | Predictions within 20% | % of predictions within ±20% of actual |
| **MdAPE** | Median Absolute % Error | Typical prediction error as % (robust to outliers) |

---

*Document maintained as part of the S.P.E.C. Valuation Engine project.*
