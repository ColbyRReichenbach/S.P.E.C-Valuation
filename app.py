"""
S.P.E.C. Valuation Engine - Streamlit Dashboard
================================================
Spatial, Predictive, Explainable, Conversational
Real Estate Automated Valuation Model Dashboard

A professional tool for Real Estate Analysts to validate AVM predictions
with full transparency between "Black Box" predictions and "White Box" explainability.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pydeck as pdk
import folium
from streamlit_folium import st_folium

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    COLORS,
    DATABASE_PATH,
    CACHE_TTL_DATA,
    CACHE_TTL_MODEL,
    CONDITION_SCALE,
    ZIP_CODES,
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
)
from src.etl import load_processed_data, run_etl_pipeline, execute_sql_query
from src.model import ValuationModel, get_model_instance
from src.spatial import add_valuation_status, prepare_map_data
from src.oracle import generate_investment_memo


# ====================================
# PAGE CONFIGURATION
# ====================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# ====================================
# CUSTOM CSS (Finance Dark Theme)
# ====================================
def inject_custom_css():
    """Inject custom CSS for the finance-grade dark theme."""
    st.markdown(f"""
    <style>
        /* Import professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global font */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Main background */
        .stApp {{
            background-color: {COLORS.BG_PRIMARY};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS.BG_SECONDARY};
            border-right: 1px solid {COLORS.BG_TERTIARY};
        }}
        
        /* Cards and containers */
        .stMetric {{
            background-color: {COLORS.BG_SECONDARY};
            padding: 1.25rem;
            border-radius: 8px;
            border: 1px solid {COLORS.BG_TERTIARY};
        }}
        
        /* Headers */
        h1 {{
            color: {COLORS.TEXT_PRIMARY};
            font-weight: 600;
            letter-spacing: -0.025em;
        }}
        
        h2, h3 {{
            color: {COLORS.TEXT_PRIMARY};
            font-weight: 500;
            letter-spacing: -0.01em;
        }}
        
        h4 {{
            color: {COLORS.TEXT_SECONDARY};
            font-weight: 500;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1.5rem;
        }}
        
        /* Text */
        p, span, label {{
            color: {COLORS.TEXT_SECONDARY};
        }}
        
        /* Metric values */
        [data-testid="stMetricValue"] {{
            color: {COLORS.TEXT_PRIMARY};
            font-weight: 600;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {COLORS.EMERALD_GREEN};
        }}
        
        /* Custom metric card */
        .metric-card {{
            background: {COLORS.BG_SECONDARY};
            border: 1px solid {COLORS.BG_TERTIARY};
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .metric-card h3 {{
            color: {COLORS.TEXT_MUTED};
            font-size: 0.75rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        
        .metric-card .value {{
            color: {COLORS.TEXT_PRIMARY};
            font-size: 1.75rem;
            font-weight: 600;
        }}
        
        .metric-card .delta {{
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}
        
        .delta-positive {{
            color: {COLORS.EMERALD_GREEN};
        }}
        
        .delta-negative {{
            color: {COLORS.CRIMSON_RED};
        }}
        
        /* Undervalued/Overvalued badges */
        .badge-undervalued {{
            background-color: {COLORS.EMERALD_GREEN}15;
            color: {COLORS.EMERALD_GREEN};
            padding: 0.35rem 0.85rem;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }}
        
        .badge-overvalued {{
            background-color: {COLORS.CRIMSON_RED}15;
            color: {COLORS.CRIMSON_RED};
            padding: 0.35rem 0.85rem;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {COLORS.BG_SECONDARY};
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        
        /* Code blocks */
        code {{
            background-color: {COLORS.BG_TERTIARY};
            color: {COLORS.CHART_CYAN};
            font-size: 0.85rem;
        }}
        
        /* Dividers */
        hr {{
            border-color: {COLORS.BG_TERTIARY};
            opacity: 0.5;
        }}
        
        /* Selectbox and inputs */
        .stSelectbox > div > div {{
            background-color: {COLORS.BG_SECONDARY};
            border-color: {COLORS.BG_TERTIARY};
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            gap: 0.5rem;
        }}
    </style>
    """, unsafe_allow_html=True)


# ====================================
# CACHED DATA LOADING (OPTIMIZED)
# ====================================
@st.cache_data(ttl=CACHE_TTL_DATA, show_spinner=False)
def get_housing_data() -> pd.DataFrame:
    """
    Load and cache housing data.
    
    Optimization: Load directly from Parquet if exists,
    bypassing the full ETL pipeline for faster startup.
    Also ensures V2.2 features are present.
    """
    parquet_path = Path("data/processed/housing.parquet")
    
    # Fast path: load from Parquet directly
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        
        # Ensure V3.0 features are present (H3 Spatial Lags + Walk Score)
        # This fixes KeyError when model expects V3 features but data is stale
        required_features = [
            'h3_median_ppsf', 'h3_neighbor_median_ppsf',  # V3.0 H3 Lags
            'walk_score', 'transit_score'                 # V3.0 Walk Score
        ]
        
        missing = [f for f in required_features if f not in df.columns]
        
        if missing:
            # Add V2.2 features first (dependencies)
            from src.spatial import add_v2_2_features
            df = add_v2_2_features(df)
            
            # Add V3.0 H3 spatial lags
            from src.spatial import calculate_h3_spatial_lags
            df = calculate_h3_spatial_lags(df)
            
            # Add V3.0 Walk Score (simulated if no API key)
            from src.external_apis import ensure_walkscore_features
            df = ensure_walkscore_features(df, use_simulation_fallback=True)
        
        return df
    
    # Slow path: run ETL pipeline
    return load_processed_data()


@st.cache_resource(ttl=CACHE_TTL_MODEL, show_spinner=False)
def get_model() -> ValuationModel:
    """
    Load and cache the valuation model.
    
    Optimization: Use show_spinner=False to avoid blocking UI.
    Model is cached for 24 hours.
    """
    return get_model_instance()


@st.cache_data(ttl=CACHE_TTL_DATA, show_spinner=False)
def get_predictions(df_hash: str, df: pd.DataFrame, _model: ValuationModel) -> pd.DataFrame:
    """
    Cache model predictions on the dataset.
    
    This is a major optimization - predictions are computed once
    and cached, rather than on every page interaction.
    
    Args:
        df_hash: Hash of the dataframe for cache key
        df: Housing data
        _model: Trained model (underscore prefix = not hashed)
    
    Returns:
        DataFrame with predictions and valuation status added
    """
    result = df.copy()
    result["model_price"] = _model.predict_batch(result)
    result = add_valuation_status(result)
    return result


def get_df_hash(df: pd.DataFrame) -> str:
    """Generate a stable hash for a dataframe for caching."""
    return str(hash(tuple(df.shape) + tuple(df.columns)))


@st.cache_data(ttl=CACHE_TTL_DATA, show_spinner=False)
def get_market_metrics() -> Dict[str, Any]:
    """
    Compute market metrics using raw SQL queries.
    
    OPTIMIZATION: Combined multiple queries into a single CTE-based query
    for faster execution. Only the top zip codes need a separate query.
    """
    metrics = {}
    
    # Single combined query for core metrics (faster than 2 separate queries)
    combined_query = """
    SELECT 
        ROUND(AVG(days_on_market), 1) as avg_dom,
        MIN(days_on_market) as min_dom,
        MAX(days_on_market) as max_dom,
        COUNT(*) as total_listings,
        SUM(price) as total_volume,
        ROUND(AVG(price), 0) as avg_price
    FROM sales
    """
    
    result = execute_sql_query(combined_query)
    metrics["avg_days_on_market"] = result["avg_dom"].iloc[0] or 0
    metrics["total_listings"] = int(result["total_listings"].iloc[0] or 0)
    metrics["total_volume"] = result["total_volume"].iloc[0] or 0
    metrics["avg_price"] = result["avg_price"].iloc[0] or 0
    
    # Store individual query strings for display (educational purpose)
    metrics["query_dom"] = """SELECT ROUND(AVG(days_on_market), 1) as avg_dom
FROM sales"""
    
    metrics["query_volume"] = """SELECT COUNT(*) as total_listings,
       SUM(price) as total_volume
FROM sales"""
    
    # Query for top zip codes (still separate for proper aggregation)
    zip_query = """
    SELECT 
        zip_code,
        ROUND(AVG(price / sqft), 0) as price_per_sqft,
        COUNT(*) as count
    FROM sales
    GROUP BY zip_code
    ORDER BY price_per_sqft DESC
    LIMIT 5
    """
    metrics["top_zip_codes"] = execute_sql_query(zip_query)
    metrics["query_zip"] = zip_query.strip()
    
    return metrics


# ====================================
# VISUALIZATION FUNCTIONS
# ====================================
def create_shap_waterfall(explanation: Dict[str, Any]) -> go.Figure:
    """
    Create a SHAP waterfall chart using Plotly.
    
    Shows how each feature contributes to the final prediction.
    """
    base_value = explanation["base_value"]
    shap_values = explanation["shap_values"]
    predicted_price = explanation["predicted_price"]
    
    # Filter out features with near-zero contribution
    active_items = {k: v for k, v in shap_values.items() if abs(v) > 1}  # $1 threshold
    
    # Sort by absolute value for visual clarity
    sorted_items = sorted(active_items.items(), key=lambda x: abs(x[1]), reverse=True)
    
    features = [item[0].replace("_", " ").title() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Build waterfall data
    y_labels = ["Market Baseline"] + features + ["Final Price"]
    
    # Calculate cumulative positions
    measures = ["absolute"] + ["relative"] * len(features) + ["total"]
    waterfall_values = [base_value] + values + [0]
    
    # Colors based on positive/negative
    colors = [COLORS.TEXT_MUTED]  # Baseline
    for v in values:
        colors.append(SHAP_POSITIVE_COLOR if v > 0 else SHAP_NEGATIVE_COLOR)
    colors.append(COLORS.CHART_BLUE)  # Total
    
    
    # Helper to format shortened currency (e.g. $150k)
    def fmt_k(val):
        is_neg = val < 0
        abs_val = abs(val)
        if abs_val >= 1_000_000:
            s = f"${abs_val/1_000_000:.1f}M"
        elif abs_val >= 1000:
            s = f"${abs_val/1000:.0f}k"
        else:
            s = f"${abs_val:.0f}"
        
        if is_neg:
            return f"-{s}"
        return f"+{s}" if val != base_value and val != 0 else s  # No + for baseline/total

    text_labels = []
    current_total = 0
    max_val = base_value
    min_val = base_value
    
    # Calculate running totals to determine y-axis range
    running_total = base_value
    for v in values:
        running_total += v
        max_val = max(max_val, running_total)
        min_val = min(min_val, running_total)
    
    # Generate labels
    text_labels = [fmt_k(base_value).replace("+", "")] # Baseline (no +)
    for v in values:
        text_labels.append(fmt_k(v))
    text_labels.append(fmt_k(predicted_price).replace("+", "")) # Final (no +)

    fig = go.Figure(go.Waterfall(
        name="Price Breakdown",
        orientation="v",
        measure=measures,
        x=y_labels,
        y=waterfall_values,
        textposition="auto",  # Automatically place inside or outside
        text=text_labels,
        connector={"line": {"color": COLORS.TEXT_MUTED}},
        increasing={"marker": {"color": SHAP_POSITIVE_COLOR}},
        decreasing={"marker": {"color": SHAP_NEGATIVE_COLOR}},
        totals={"marker": {"color": COLORS.CHART_BLUE}},
        textfont=dict(size=12, color="white"),  # White text works best for 'auto' inside
        cliponaxis=False,  # Allow text to overflow axis area if needed
    ))
    
    # Add headroom
    y_range_padding = (max_val - min_val) * 0.2
    
    fig.update_layout(
        title="Property Valuation Breakdown (SHAP)",
        showlegend=False,
        plot_bgcolor=COLORS.BG_PRIMARY,
        paper_bgcolor=COLORS.BG_PRIMARY,
        font=dict(color=COLORS.TEXT_SECONDARY),
        height=450,
        margin=dict(t=80, b=100, l=80, r=80),
        uniformtext=dict(minsize=10, mode='hide'),  # Ensure readable text
    )
    
    fig.update_yaxes(
        title="Price ($)",
        tickformat="$,.0f",
        gridcolor=COLORS.BG_TERTIARY,
        range=[min_val - y_range_padding, max_val + y_range_padding], # Explicit range with padding
        automargin=True,
    )
    
    fig.update_xaxes(
        tickangle=45,
        automargin=True,  # Auto-adjust for labels
    )
    
    return fig


def create_price_distribution_chart(df: pd.DataFrame, selected_price: float = None) -> go.Figure:
    """Create a price distribution histogram with optional marker."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df["price"],
        nbinsx=30,
        marker_color=COLORS.CHART_BLUE,
        opacity=0.7,
        name="Price Distribution",
    ))
    
    if selected_price:
        fig.add_vline(
            x=selected_price,
            line_dash="dash",
            line_color=COLORS.EMERALD_GREEN,
            annotation_text=f"Selected: ${selected_price:,.0f}",
            annotation_position="top",
        )
    
    fig.update_layout(
        title="Market Price Distribution",
        xaxis_title="Price ($)",
        yaxis_title="Count",
        plot_bgcolor=COLORS.BG_PRIMARY,
        paper_bgcolor=COLORS.BG_PRIMARY,
        font=dict(color=COLORS.TEXT_SECONDARY),
        height=300,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    fig.update_xaxes(tickformat="$,.0f", gridcolor=COLORS.BG_TERTIARY)
    fig.update_yaxes(gridcolor=COLORS.BG_TERTIARY)
    
    return fig


# ====================================
# MAIN APPLICATION
# ====================================
def main():
    """Main application entry point."""
    
    # One-time cache clear for V2.2 upgrade (remove after first run)
    if 'v2_2_cache_cleared' not in st.session_state:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state['v2_2_cache_cleared'] = True
    
    # Inject custom CSS
    inject_custom_css()
    
    # ================================
    # HEADER
    # ================================
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 2.5rem 0;">
        <h1 style="font-size: 2.25rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: -0.025em;">
            S.P.E.C. Valuation Engine
        </h1>
        <p style="font-size: 0.95rem; color: #6B7280; letter-spacing: 0.1em; text-transform: uppercase;">
            Spatial · Predictive · Explainable · Conversational
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================
    # LOAD DATA & MODEL (OPTIMIZED)
    # ================================
    # Load data and model (cached)
    df_raw = get_housing_data()
    model = get_model()
    
    # Get cached predictions (major performance boost)
    df_hash = get_df_hash(df_raw)
    df = get_predictions(df_hash, df_raw, model)
    
    # ================================
    # MARKET PULSE (Header Metrics)
    # ================================
    st.markdown("### Market Pulse")
    
    metrics = get_market_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Days on Market",
            value=f"{metrics['avg_days_on_market']:.0f} days",
            delta=None,  # Removed fake delta - would need historical data
        )
        with st.expander("View SQL Query"):
            st.code(metrics["query_dom"], language="sql")
    
    with col2:
        st.metric(
            label="Total Market Volume",
            value=f"${metrics['total_volume'] / 1_000_000:.1f}M",
            delta=None,  # Removed - was showing listing count incorrectly as delta
            help=f"{metrics['total_listings']} total listings",
        )
        with st.expander("View SQL Query"):
            st.code(metrics["query_volume"], language="sql")
    
    with col3:
        st.metric(
            label="Average Price",
            value=f"${metrics['avg_price']:,.0f}",
            delta=None,  # Removed fake delta - would need historical data
        )
        with st.expander("View SQL Query"):
            st.code(metrics["query_zip"], language="sql")
    
    st.markdown("---")
    
    # ================================
    # MAIN CONTENT (Two Columns)
    # ================================
    left_col, right_col = st.columns([1, 1.5])
    
    # ================================
    # LEFT COLUMN: THE SCREENER
    # ================================
    with left_col:
        st.markdown("### Property Screener")
        
        # Filters
        st.markdown("#### Filters")
        
        # Price Range
        min_price = int(df["price"].min())
        max_price = int(df["price"].max())
        price_range = st.slider(
            "Price Range ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            format="$%d",
            key="price_filter",
        )
        
        # Zip Code Filter
        all_zips = ["All"] + sorted(df["zip_code"].unique().tolist())
        selected_zip = st.selectbox("Zip Code", all_zips, key="zip_filter")
        
        # Valuation Status Filter
        valuation_filter = st.radio(
            "Valuation Status",
            ["All", "Undervalued", "Overvalued"],
            horizontal=True,
            key="valuation_filter",
        )
        
        # Apply filters
        filtered_df = df.copy()
        filtered_df = filtered_df[
            (filtered_df["price"] >= price_range[0]) &
            (filtered_df["price"] <= price_range[1])
        ]
        
        if selected_zip != "All":
            filtered_df = filtered_df[filtered_df["zip_code"] == selected_zip]
        
        if valuation_filter != "All":
            filtered_df = filtered_df[filtered_df["valuation_status"] == valuation_filter]
        
        st.markdown(f"**{len(filtered_df)}** properties found")
        
        # Initialize session state for selected property
        if "selected_property_id" not in st.session_state:
            st.session_state.selected_property_id = None
        
        # Property Map with Click-to-Select
        st.markdown("#### Property Map")
        st.caption("Click a marker to select a property")
        
        if len(filtered_df) > 0:
            # Center map on data
            center_lat = filtered_df["lat"].mean()
            center_lon = filtered_df["lon"].mean()
            
            # Create Folium map with dark theme
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="CartoDB dark_matter",
            )
            
            # Add markers for each property
            for _, row in filtered_df.iterrows():
                property_id = row["id"]
                is_selected = (st.session_state.selected_property_id == property_id)
                is_undervalued = row["valuation_status"] == "Undervalued"
                
                # Determine marker color and size
                if is_selected:
                    # Selected property: bright blue, larger
                    color = "#3B82F6"  # Blue
                    radius = 12
                    fill_opacity = 1.0
                elif is_undervalued:
                    color = "#00D47E"  # Green
                    radius = 8
                    fill_opacity = 0.7
                else:
                    color = "#FF4757"  # Red
                    radius = 8
                    fill_opacity = 0.7
                
                # Create rich tooltip (SHOWS ON HOVER - faster UX)
                # Fixed-width layout for consistent appearance
                status_color = '#00D47E' if is_undervalued else '#FF4757'
                tooltip_html = f"""
                <div style="font-family: Inter, sans-serif; width: 180px; font-size: 12px; line-height: 1.4;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <b style="font-size: 13px;">Property #{property_id}</b>
                        <span style="color: {status_color}; font-weight: 600; font-size: 11px;">
                            {row['valuation_status'].upper()}
                        </span>
                    </div>
                    <hr style="margin: 0 0 4px 0; border: none; border-top: 1px solid #444;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>Price:</b></span>
                        <span>${row['price']:,.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>Model:</b></span>
                        <span>${row['model_price']:,.0f} <span style="color: {status_color};">({row['price_delta_pct']:+.1f}%)</span></span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>Sqft:</b></span>
                        <span>{row['sqft']:,}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>Beds:</b></span>
                        <span>{row['bedrooms']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>Zip:</b></span>
                        <span>{row['zip_code']}</span>
                    </div>
                </div>
                """
                
                # Add circle marker (no popup for speed, rich tooltip instead)
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    weight=2 if is_selected else 1,
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                ).add_to(m)
                
                # Add highlight ring for selected marker
                if is_selected:
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=16,
                        color="#3B82F6",
                        fill=False,
                        weight=2,
                        opacity=0.5,
                    ).add_to(m)
            
            # Render Folium map and capture clicks
            # use_container_width for responsiveness, center_on_click=False to reduce lag
            map_data = st_folium(
                m,
                use_container_width=True,
                height=400,
                returned_objects=["last_object_clicked"],
                key="property_map",
            )
            
            # Handle map click events
            if map_data and map_data.get("last_object_clicked"):
                clicked_lat = map_data["last_object_clicked"].get("lat")
                clicked_lng = map_data["last_object_clicked"].get("lng")
                
                if clicked_lat and clicked_lng:
                    # Find the closest property to the click
                    filtered_df["_dist"] = (
                        (filtered_df["lat"] - clicked_lat) ** 2 +
                        (filtered_df["lon"] - clicked_lng) ** 2
                    )
                    closest_property = filtered_df.loc[filtered_df["_dist"].idxmin()]
                    
                    # Update selected property if different
                    if st.session_state.selected_property_id != closest_property["id"]:
                        st.session_state.selected_property_id = closest_property["id"]
                        st.rerun()
            
            # Legend
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap;">
                <span class="badge-undervalued">● Undervalued</span>
                <span class="badge-overvalued">● Overvalued</span>
                <span style="background-color: #3B82F615; color: #3B82F6; padding: 0.35rem 0.85rem; 
                             border-radius: 4px; font-weight: 500; font-size: 0.8rem;">
                    ● Selected
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No properties match the current filters.")
        
        # Property Selection (synced with map clicks)
        st.markdown("#### Select Property")
        
        if len(filtered_df) > 0:
            property_options = filtered_df.apply(
                lambda row: f"ID {row['id']}: ${row['price']:,.0f} | {row['sqft']} sqft | {row['zip_code']}",
                axis=1
            ).tolist()
            
            # Find the index of currently selected property (from map click)
            default_index = 0
            if st.session_state.selected_property_id is not None:
                matching_options = [i for i, opt in enumerate(property_options) 
                                   if f"ID {st.session_state.selected_property_id}:" in opt]
                if matching_options:
                    default_index = matching_options[0]
            
            # Callback to update session state without extra rerun
            def on_property_select():
                selected_str = st.session_state.property_select_widget
                new_id = int(selected_str.split(":")[0].replace("ID ", ""))
                st.session_state.selected_property_id = new_id
            
            selected_property_str = st.selectbox(
                "Choose a property to analyze",
                property_options,
                index=default_index,
                key="property_select_widget",
                on_change=on_property_select,
            )
            
            # Extract selected property ID from dropdown (without rerun)
            selected_id = int(selected_property_str.split(":")[0].replace("ID ", ""))
            selected_property = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        else:
            selected_property = None
    
    # ================================
    # RIGHT COLUMN: THE INSPECTOR
    # ================================
    with right_col:
        st.markdown("### Property Analysis")
        
        if selected_property is not None:
            # Property Details Card
            status = selected_property["valuation_status"]
            badge_class = "badge-undervalued" if status == "Undervalued" else "badge-overvalued"
            delta_color = COLORS.EMERALD_GREEN if status == "Undervalued" else COLORS.CRIMSON_RED
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Selected Property</h3>
                <div class="value">${selected_property['price']:,.0f}</div>
                <div class="delta" style="color: {delta_color};">
                    Model: ${selected_property['model_price']:,.0f} 
                    ({selected_property['price_delta_pct']:+.1f}%)
                </div>
                <div style="margin-top: 0.5rem;">
                    <span class="{badge_class}">{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Property specs
            spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
            with spec_col1:
                st.metric("Sqft", f"{selected_property['sqft']:,}")
            with spec_col2:
                st.metric("Bedrooms", selected_property['bedrooms'])
            with spec_col3:
                st.metric("Year Built", selected_property['year_built'])
            with spec_col4:
                condition_label = CONDITION_SCALE.get(selected_property['condition'], "N/A")
                st.metric("Condition", condition_label)
            
            # ================================
            # WATERFALL CHART (SHAP)
            # ================================
            st.markdown("#### Valuation Breakdown")
            
            explanation = model.explain(property_data=selected_property.to_dict())
            
            waterfall_chart = create_shap_waterfall(explanation)
            st.plotly_chart(waterfall_chart, use_container_width=True)
            
            # ================================
            # RENOVATION SIMULATOR
            # ================================
            st.markdown("#### Renovation Simulator")
            st.caption("Adjust property features to see how value changes in real-time.")
            
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                sim_sqft = st.slider(
                    "Simulated Sqft",
                    min_value=500,
                    max_value=5000,
                    value=int(selected_property["sqft"]),
                    step=100,
                    key="sim_sqft",
                )
            
            with sim_col2:
                sim_condition = st.slider(
                    "Simulated Condition",
                    min_value=1,
                    max_value=5,
                    value=int(selected_property["condition"]),
                    step=1,
                    key="sim_condition",
                    help="1=Poor, 5=Excellent",
                )
            
            # Prepare simulation data: Start with original property features, update with new inputs
            sim_data = selected_property.to_dict().copy()
            sim_data["sqft"] = sim_sqft
            sim_data["condition"] = sim_condition
            
            # Calculate new prediction
            new_price = model.predict(property_data=sim_data)
            
            original_model_price = selected_property["model_price"]
            price_change = new_price - original_model_price
            
            sim_result_col1, sim_result_col2 = st.columns(2)
            
            with sim_result_col1:
                st.metric(
                    "New Estimated Value",
                    f"${new_price:,.0f}",
                    delta=f"${price_change:+,.0f}",
                )
            
            with sim_result_col2:
                roi = (price_change / original_model_price) * 100 if original_model_price > 0 else 0
                st.metric(
                    "Value Change (%)",
                    f"{roi:+.2f}%",
                )
            
            # ================================
            # AI INVESTMENT MEMO
            # ================================
            st.markdown("#### Investment Analysis")
            
            # Initialize session state for AI memo
            if "ai_memo_property_id" not in st.session_state:
                st.session_state.ai_memo_property_id = None
                st.session_state.ai_memo_content = None
            
            # Use Index as ID if 'id' column missing
            current_property_id = selected_property.get("id")
            if pd.isna(current_property_id):
                current_property_id = selected_property.name  # Use index
            
            # Check if we already have a memo for this property
            if st.session_state.ai_memo_property_id == current_property_id and st.session_state.ai_memo_content:
                st.markdown(st.session_state.ai_memo_content, unsafe_allow_html=True)
                st.success("✅ AI Analysis Complete")
                
                # Option to regenerate
                if st.button("🔄 Regenerate"):
                    st.session_state.ai_memo_content = None
                    st.rerun()
                    
            else:
                # Show template memo by default (no API call)
                template_memo = generate_investment_memo(
                    price=selected_property["model_price"],
                    shap_data=explanation,
                    zip_code=selected_property["zip_code"],
                    use_ai=False,  # Use template only, no API call
                )
                st.markdown(template_memo, unsafe_allow_html=True)
                
                # Button to run AI analysis
                st.markdown("---")
                
                # Check for API key
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    st.warning("⚠️ OPENAI_API_KEY not found in .env file. AI analysis disabled.")
                else:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        run_ai = st.button(
                            "🤖 Run AI Analysis",
                            help="Generate AI-powered investment memo using OpenAI",
                            type="primary",
                            use_container_width=True,
                        )
                    
                    with col2:
                        estimated_cost = 0.0003
                        st.caption(f"Est. cost: ${estimated_cost:.4f}")
                    
                    if run_ai:
                        with st.spinner("🤖 AI is analyzing property..."):
                            try:
                                ai_memo = generate_investment_memo(
                                    price=selected_property["model_price"],
                                    shap_data=explanation,
                                    zip_code=selected_property["zip_code"],
                                    use_ai=True,
                                )
                                
                                # Cache the result
                                st.session_state.ai_memo_property_id = current_property_id
                                st.session_state.ai_memo_content = ai_memo
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"AI Generation Failed: {str(e)}")
        
        else:
            st.info("Select a property from the Screener to view detailed analysis.")
    
    # ================================
    # FOOTER
    # ================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #4B5563; font-size: 0.8rem; padding: 1rem 0;">
        <p style="margin-bottom: 0.25rem;">S.P.E.C. Valuation Engine v1.0</p>
        <p style="color: #6B7280;">Model predictions are for demonstration purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
