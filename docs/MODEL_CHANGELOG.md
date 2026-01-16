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
| **V3.0-b** | 2026-01-15 | **0.893** | **$193K** | **$145K** | **55.6%** | **85.3%** | **8.8%** | 10 | **1,330** |

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

## Upcoming Changes (Planned)

### V3.0 Completion (In Progress)
- [x] Phase 3A.1: PPE10/PPE20/MdAPE Metrics ✅
- [x] Phase 3A.2: Isolation Forest Outlier Detection ✅
- [ ] Phase 3B: H3 Spatial Lag Features
- [ ] Phase 3B: Walk Score API Integration
- [ ] Phase 3C: OpenAI Activation
- [ ] Phase 3E: AI Security (Prompt Injection Prevention)

### V4.0 (Future)
- [ ] Data expansion to 5,000-10,000 records
- [ ] Cascading Model Architecture (segmented by property tier)
- [ ] FRED Interest Rate Integration
- [ ] Multi-provider LLM Routing

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
