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
        /* Main background */
        .stApp {{
            background-color: {COLORS.BG_PRIMARY};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS.BG_SECONDARY};
        }}
        
        /* Cards and containers */
        .stMetric {{
            background-color: {COLORS.BG_TERTIARY};
            padding: 1rem;
            border-radius: 0.5rem;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {COLORS.TEXT_PRIMARY};
        }}
        
        /* Text */
        p, span, label {{
            color: {COLORS.TEXT_SECONDARY};
        }}
        
        /* Metric values */
        [data-testid="stMetricValue"] {{
            color: {COLORS.EMERALD_GREEN};
        }}
        
        /* Custom metric card */
        .metric-card {{
            background: linear-gradient(135deg, {COLORS.BG_SECONDARY} 0%, {COLORS.BG_TERTIARY} 100%);
            border: 1px solid {COLORS.BG_TERTIARY};
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .metric-card h3 {{
            color: {COLORS.TEXT_MUTED};
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metric-card .value {{
            color: {COLORS.TEXT_PRIMARY};
            font-size: 2rem;
            font-weight: 700;
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
            background-color: {COLORS.EMERALD_GREEN}20;
            color: {COLORS.EMERALD_GREEN};
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
        }}
        
        .badge-overvalued {{
            background-color: {COLORS.CRIMSON_RED}20;
            color: {COLORS.CRIMSON_RED};
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {COLORS.BG_TERTIARY};
            border-radius: 0.5rem;
        }}
        
        /* Code blocks */
        code {{
            background-color: {COLORS.BG_TERTIARY};
            color: {COLORS.CHART_CYAN};
        }}
    </style>
    """, unsafe_allow_html=True)


# ====================================
# CACHED DATA LOADING
# ====================================
@st.cache_data(ttl=CACHE_TTL_DATA)
def get_housing_data() -> pd.DataFrame:
    """Load and cache housing data."""
    return load_processed_data()


@st.cache_resource(ttl=CACHE_TTL_MODEL)
def get_model() -> ValuationModel:
    """Load and cache the valuation model."""
    return get_model_instance()


@st.cache_data(ttl=CACHE_TTL_DATA)
def get_market_metrics() -> Dict[str, Any]:
    """
    Compute market metrics using raw SQL queries.
    Demonstrates SQL skills for the dashboard.
    """
    metrics = {}
    
    # Query 1: Average Days on Market
    query1 = """
    SELECT 
        ROUND(AVG(days_on_market), 1) as avg_dom,
        MIN(days_on_market) as min_dom,
        MAX(days_on_market) as max_dom
    FROM sales
    """
    result1 = execute_sql_query(query1)
    metrics["avg_days_on_market"] = result1["avg_dom"].iloc[0]
    metrics["query_dom"] = query1.strip()
    
    # Query 2: Total Volume (Sum of all prices)
    query2 = """
    SELECT 
        COUNT(*) as total_listings,
        SUM(price) as total_volume,
        ROUND(AVG(price), 0) as avg_price
    FROM sales
    """
    result2 = execute_sql_query(query2)
    metrics["total_listings"] = int(result2["total_listings"].iloc[0])
    metrics["total_volume"] = result2["total_volume"].iloc[0]
    metrics["avg_price"] = result2["avg_price"].iloc[0]
    metrics["query_volume"] = query2.strip()
    
    # Query 3: Price per SqFt by Zip Code (Top 5)
    query3 = """
    SELECT 
        zip_code,
        ROUND(AVG(price / sqft), 0) as price_per_sqft,
        COUNT(*) as count
    FROM sales
    GROUP BY zip_code
    ORDER BY price_per_sqft DESC
    LIMIT 5
    """
    result3 = execute_sql_query(query3)
    metrics["top_zip_codes"] = result3
    metrics["query_zip"] = query3.strip()
    
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
    
    # Sort by absolute value for visual clarity
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    features = [item[0].replace("_", " ").title() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Build waterfall data
    y_labels = ["Baseline"] + features + ["Final Price"]
    
    # Calculate cumulative positions
    measures = ["absolute"] + ["relative"] * len(features) + ["total"]
    waterfall_values = [base_value] + values + [0]
    
    # Colors based on positive/negative
    colors = [COLORS.TEXT_MUTED]  # Baseline
    for v in values:
        colors.append(SHAP_POSITIVE_COLOR if v > 0 else SHAP_NEGATIVE_COLOR)
    colors.append(COLORS.CHART_BLUE)  # Total
    
    fig = go.Figure(go.Waterfall(
        name="Price Breakdown",
        orientation="v",
        measure=measures,
        x=y_labels,
        y=waterfall_values,
        textposition="outside",
        text=[f"${v:,.0f}" if i == 0 else f"${v:+,.0f}" if i < len(waterfall_values) - 1 else f"${predicted_price:,.0f}" 
              for i, v in enumerate(waterfall_values)],
        connector={"line": {"color": COLORS.TEXT_MUTED}},
        increasing={"marker": {"color": SHAP_POSITIVE_COLOR}},
        decreasing={"marker": {"color": SHAP_NEGATIVE_COLOR}},
        totals={"marker": {"color": COLORS.CHART_BLUE}},
    ))
    
    fig.update_layout(
        title="Property Valuation Breakdown (SHAP)",
        showlegend=False,
        plot_bgcolor=COLORS.BG_PRIMARY,
        paper_bgcolor=COLORS.BG_PRIMARY,
        font=dict(color=COLORS.TEXT_SECONDARY),
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    fig.update_yaxes(
        title="Price ($)",
        tickformat="$,.0f",
        gridcolor=COLORS.BG_TERTIARY,
    )
    
    fig.update_xaxes(
        tickangle=45,
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
    
    # Inject custom CSS
    inject_custom_css()
    
    # ================================
    # HEADER
    # ================================
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            🏠 S.P.E.C. Valuation Engine
        </h1>
        <p style="font-size: 1.1rem; color: #A0A4AB;">
            <strong>S</strong>patial • <strong>P</strong>redictive • <strong>E</strong>xplainable • <strong>C</strong>onversational
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================
    # LOAD DATA & MODEL
    # ================================
    with st.spinner("Loading market data..."):
        df = get_housing_data()
        model = get_model()
        
        # Add model predictions to data
        df["model_price"] = model.predict_batch(df)
        df = add_valuation_status(df)
    
    # ================================
    # MARKET PULSE (Header Metrics)
    # ================================
    st.markdown("### 📊 The Market Pulse")
    
    metrics = get_market_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Days on Market",
            value=f"{metrics['avg_days_on_market']:.0f} days",
            delta="-5 vs last month",
        )
        with st.expander("📄 View SQL Query"):
            st.code(metrics["query_dom"], language="sql")
    
    with col2:
        st.metric(
            label="Total Market Volume",
            value=f"${metrics['total_volume'] / 1_000_000:.1f}M",
            delta=f"{metrics['total_listings']} listings",
        )
        with st.expander("📄 View SQL Query"):
            st.code(metrics["query_volume"], language="sql")
    
    with col3:
        st.metric(
            label="Average Price",
            value=f"${metrics['avg_price']:,.0f}",
            delta="+3.2% YoY",
        )
        with st.expander("📄 View SQL Query"):
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
        st.markdown("### 🔍 The Screener")
        
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
        
        # Property Map
        st.markdown("#### Property Map")
        
        if len(filtered_df) > 0:
            # Prepare map data with colors
            map_df = prepare_map_data(filtered_df)
            
            # Add color column for visualization context
            # Note: st.map doesn't support colors, but we show the data
            st.map(map_df[["lat", "lon"]], use_container_width=True)
            
            # Legend
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                <span class="badge-undervalued">● Undervalued</span>
                <span class="badge-overvalued">● Overvalued</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No properties match the current filters.")
        
        # Property Selection
        st.markdown("#### Select Property")
        
        if len(filtered_df) > 0:
            property_options = filtered_df.apply(
                lambda row: f"ID {row['id']}: ${row['price']:,.0f} | {row['sqft']} sqft | {row['zip_code']}",
                axis=1
            ).tolist()
            
            selected_property_str = st.selectbox(
                "Choose a property to analyze",
                property_options,
                key="property_select",
            )
            
            # Extract selected property ID
            selected_id = int(selected_property_str.split(":")[0].replace("ID ", ""))
            selected_property = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        else:
            selected_property = None
    
    # ================================
    # RIGHT COLUMN: THE INSPECTOR
    # ================================
    with right_col:
        st.markdown("### 🔬 The Inspector")
        
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
            st.markdown("#### 📊 Valuation Breakdown (White Box)")
            
            explanation = model.explain(
                sqft=selected_property["sqft"],
                bedrooms=selected_property["bedrooms"],
                year_built=selected_property["year_built"],
                condition=selected_property["condition"],
            )
            
            waterfall_chart = create_shap_waterfall(explanation)
            st.plotly_chart(waterfall_chart, use_container_width=True)
            
            # ================================
            # RENOVATION SIMULATOR
            # ================================
            st.markdown("#### 🔧 Renovation Simulator")
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
            
            # Calculate new prediction
            new_price = model.predict(
                sqft=sim_sqft,
                bedrooms=selected_property["bedrooms"],
                year_built=selected_property["year_built"],
                condition=sim_condition,
            )
            
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
            st.markdown("#### 🤖 AI Investment Memo")
            
            with st.spinner("Generating analysis..."):
                memo = generate_investment_memo(
                    price=selected_property["model_price"],
                    shap_data=explanation,
                    zip_code=selected_property["zip_code"],
                )
            
            st.markdown(memo)
        
        else:
            st.info("👈 Select a property from the Screener to analyze.")
    
    # ================================
    # FOOTER
    # ================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>S.P.E.C. Valuation Engine v1.0 | Built for Real Estate Analysts</p>
        <p>⚠️ Model predictions are for demonstration purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
