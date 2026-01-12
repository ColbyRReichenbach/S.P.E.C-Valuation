"""
S.P.E.C. Valuation Engine - AI Oracle
======================================
LLM-powered investment memo generation with market context.
Implements the "Conversational" aspect of S.P.E.C.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# ====================================
# MOCK NEWS DATABASE
# ====================================
# Simulated local market news tied to zip codes
MARKET_NEWS_DATABASE: Dict[str, Dict[str, Any]] = {
    "94102": {
        "neighborhood": "Tenderloin/Civic Center",
        "sentiment": "Bearish",
        "news": [
            "City proposes new affordable housing mandates affecting investor returns",
            "Homeless services expansion planned for 2024",
            "Tech company announces office closure nearby"
        ],
        "risk_factors": ["High crime rate", "Rent control pressure", "Commercial vacancy"],
    },
    "94103": {
        "neighborhood": "SoMa",
        "sentiment": "Neutral",
        "news": [
            "Mixed-use development approved on Folsom Street",
            "Transit improvements planned for 2025",
            "Some tech companies returning to office"
        ],
        "risk_factors": ["Market volatility", "Zoning changes pending"],
    },
    "94104": {
        "neighborhood": "Financial District",
        "sentiment": "Bearish",
        "news": [
            "Office-to-residential conversions gaining momentum",
            "Major bank reducing downtown footprint",
            "Daytime foot traffic down 40% from 2019 levels"
        ],
        "risk_factors": ["Commercial exodus", "Structural market shift"],
    },
    "94105": {
        "neighborhood": "Rincon Hill/South Beach",
        "sentiment": "Bullish",
        "news": [
            "New waterfront park opening increases desirability",
            "Luxury condo sales up 15% YoY",
            "Oracle Park events driving local economy"
        ],
        "risk_factors": ["High HOA fees", "Earthquake liquefaction zone"],
    },
    "94107": {
        "neighborhood": "Potrero Hill/Dogpatch",
        "sentiment": "Bullish",
        "news": [
            "Biotech hub expansion creating jobs",
            "Historic warehouse conversions in demand",
            "New MUNI line improves connectivity"
        ],
        "risk_factors": ["Limited parking", "Industrial adjacency"],
    },
    "94109": {
        "neighborhood": "Nob Hill/Russian Hill",
        "sentiment": "Stable",
        "news": [
            "Historic preservation status limits new supply",
            "Cable car renovation completed",
            "Luxury rental market stabilizing"
        ],
        "risk_factors": ["Limited appreciation potential", "Aging building stock"],
    },
    "94110": {
        "neighborhood": "Mission District",
        "sentiment": "Neutral",
        "news": [
            "Small business revival gaining momentum",
            "Cultural district designation approved",
            "Gentrification concerns continue"
        ],
        "risk_factors": ["Political intervention risk", "Rent control"],
    },
    "94114": {
        "neighborhood": "Castro",
        "sentiment": "Stable",
        "news": [
            "Neighborhood retail occupancy improving",
            "Historic building renovation incentives available",
            "Strong community investment continues"
        ],
        "risk_factors": ["Limited inventory", "High entry price"],
    },
    "94115": {
        "neighborhood": "Pacific Heights",
        "sentiment": "Bullish",
        "news": [
            "Billionaires Row sees continued demand",
            "Trophy property sold for $30M+",
            "Private school enrollment driving family demand"
        ],
        "risk_factors": ["Ultra-high price point", "Limited buyer pool"],
    },
    "94117": {
        "neighborhood": "Haight-Ashbury",
        "sentiment": "Neutral",
        "news": [
            "Tourism recovery boosting local businesses",
            "Victorian restoration projects active",
            "Parking challenges persist"
        ],
        "risk_factors": ["Tourist impact", "Older housing stock"],
    },
}

# Default news for unknown zip codes
DEFAULT_MARKET_NEWS = {
    "neighborhood": "Greater San Francisco",
    "sentiment": "Neutral",
    "news": [
        "Regional housing market showing mixed signals",
        "Interest rates impacting buyer behavior",
        "Inventory levels below historical average"
    ],
    "risk_factors": ["Market uncertainty", "Rate sensitivity"],
}


def get_market_context(zip_code: str) -> Dict[str, Any]:
    """
    Retrieve market context for a given zip code.
    
    Args:
        zip_code: Property zip code.
    
    Returns:
        Dictionary with neighborhood info, news, and risk factors.
    """
    return MARKET_NEWS_DATABASE.get(zip_code, DEFAULT_MARKET_NEWS)


def _get_openai_client():
    """
    Get OpenAI client if API key is available.
    
    Returns:
        OpenAI client or None if key not found.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not configured. Running in simulation mode.")
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.error("OpenAI package not installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def generate_investment_memo(
    price: float,
    shap_data: Dict[str, Any],
    zip_code: str,
    use_ai: bool = True,
) -> str:
    """
    Generate an investment memo using AI or fallback to template.
    
    The AI acts as a "Bearish Investment Analyst" who critically evaluates
    the property valuation and market conditions.
    
    Args:
        price: Predicted property price.
        shap_data: SHAP explanation data from the model.
        zip_code: Property zip code for market context.
        use_ai: Whether to attempt AI generation.
    
    Returns:
        Investment memo as a formatted string.
    """
    market_context = get_market_context(zip_code)
    
    # Build the context for the memo
    shap_values = shap_data.get("shap_values", {})
    feature_values = shap_data.get("feature_values", {})
    base_value = shap_data.get("base_value", 0)
    
    # Sort SHAP values by absolute impact
    sorted_shap = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Attempt AI generation
    if use_ai:
        client = _get_openai_client()
        
        if client:
            try:
                memo = _generate_ai_memo(
                    client=client,
                    price=price,
                    base_value=base_value,
                    sorted_shap=sorted_shap,
                    feature_values=feature_values,
                    market_context=market_context,
                )
                return memo
            except Exception as e:
                logger.error(f"AI memo generation failed: {e}")
    
    # Fallback to template-based memo
    return _generate_template_memo(
        price=price,
        base_value=base_value,
        sorted_shap=sorted_shap,
        feature_values=feature_values,
        market_context=market_context,
    )


def _generate_ai_memo(
    client,
    price: float,
    base_value: float,
    sorted_shap: list,
    feature_values: Dict,
    market_context: Dict,
) -> str:
    """Generate memo using OpenAI API."""
    
    # Build the prompt
    shap_explanation = "\n".join([
        f"  - {feat}: ${val:+,.0f} impact"
        for feat, val in sorted_shap
    ])
    
    news_items = "\n".join([f"  - {n}" for n in market_context["news"]])
    risk_items = "\n".join([f"  - {r}" for r in market_context["risk_factors"]])
    
    prompt = f"""You are a Bearish Investment Analyst reviewing a residential property valuation. 
Your job is to critically evaluate the property and highlight potential risks while remaining data-driven.

PROPERTY DATA:
- Model Valuation: ${price:,.0f}
- Market Baseline: ${base_value:,.0f}
- Square Footage: {feature_values.get('sqft', 'N/A')} sqft
- Bedrooms: {feature_values.get('bedrooms', 'N/A')}
- Year Built: {feature_values.get('year_built', 'N/A')}
- Condition Rating: {feature_values.get('condition', 'N/A')}/5

VALUATION BREAKDOWN (SHAP Analysis):
{shap_explanation}

MARKET CONTEXT - {market_context['neighborhood']}:
Sentiment: {market_context['sentiment']}

Recent News:
{news_items}

Risk Factors:
{risk_items}

Write a concise investment memo (3-4 paragraphs) that:
1. Summarizes the valuation and key price drivers
2. Critically analyzes risks and bearish considerations  
3. Provides a recommendation with caveats

Use a professional, analytical tone. Include specific numbers. Be skeptical but fair."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior real estate investment analyst with a bearish disposition."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )
    
    return response.choices[0].message.content


def _generate_template_memo(
    price: float,
    base_value: float,
    sorted_shap: list,
    feature_values: Dict,
    market_context: Dict,
) -> str:
    """Generate fallback template-based memo."""
    
    # Determine overall assessment
    sentiment = market_context["sentiment"]
    neighborhood = market_context["neighborhood"]
    
    # Get top positive and negative drivers
    positive_drivers = [(f, v) for f, v in sorted_shap if v > 0]
    negative_drivers = [(f, v) for f, v in sorted_shap if v < 0]
    
    # Calculate price premium/discount
    price_diff = price - base_value
    price_diff_pct = (price_diff / base_value) * 100 if base_value > 0 else 0
    
    # Determine recommendation based on sentiment
    if sentiment == "Bullish":
        recommendation = "CAUTIOUS BUY"
        rec_color = "#00D47E"
    elif sentiment == "Bearish":
        recommendation = "HOLD / AVOID"
        rec_color = "#FF4757"
    else:
        recommendation = "NEUTRAL"
        rec_color = "#FFB946"
    
    # Build the memo
    memo = f"""
<div style="background: #1a1d24; border: 1px solid #2d3139; border-radius: 8px; padding: 1.5rem; margin-top: 0.5rem;">

<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #2d3139;">
    <div>
        <p style="color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">Investment Recommendation</p>
        <p style="color: {rec_color}; font-size: 1.25rem; font-weight: 600; margin: 0.25rem 0 0 0;">{recommendation}</p>
    </div>
    <div style="text-align: right;">
        <p style="color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">Model Confidence</p>
        <p style="color: #E5E7EB; font-size: 1.25rem; font-weight: 600; margin: 0.25rem 0 0 0;">HIGH</p>
    </div>
</div>

<div style="margin-bottom: 1.25rem;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Valuation Summary</p>
    <p style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.6; margin: 0;">
        The model values this property at <strong style="color: #fff;">${price:,.0f}</strong>, representing a 
        <span style="color: {'#00D47E' if price_diff >= 0 else '#FF4757'};">{price_diff_pct:+.1f}%</span> 
        ({'+' if price_diff >= 0 else ''}{price_diff:,.0f}) deviation from the market baseline of ${base_value:,.0f}. 
        The valuation is derived from {len(sorted_shap)} key property attributes analyzed via SHAP explainability.
    </p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem;">
    <div>
        <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Value Drivers (+)</p>
"""
    
    if positive_drivers:
        for feat, val in positive_drivers[:3]:
            memo += f"""        <p style="color: #00D47E; font-size: 0.85rem; margin: 0.25rem 0;">{feat.replace('_', ' ').title()}: +${val:,.0f}</p>\n"""
    else:
        memo += """        <p style="color: #6B7280; font-size: 0.85rem; margin: 0.25rem 0;">No positive drivers identified</p>\n"""
    
    memo += """    </div>
    <div>
        <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Value Detractors (-)</p>
"""
    
    if negative_drivers:
        for feat, val in negative_drivers[:3]:
            memo += f"""        <p style="color: #FF4757; font-size: 0.85rem; margin: 0.25rem 0;">{feat.replace('_', ' ').title()}: ${val:,.0f}</p>\n"""
    else:
        memo += """        <p style="color: #6B7280; font-size: 0.85rem; margin: 0.25rem 0;">No negative drivers identified</p>\n"""
    
    memo += f"""    </div>
</div>

<div style="margin-bottom: 1.25rem; padding: 1rem; background: #12141a; border-radius: 6px;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Market Context: {neighborhood}</p>
    <div style="display: flex; gap: 2rem; margin-bottom: 0.75rem;">
        <div>
            <span style="color: #6B7280; font-size: 0.75rem;">Sentiment</span>
            <p style="color: {'#00D47E' if sentiment == 'Bullish' else '#FF4757' if sentiment == 'Bearish' else '#FFB946'}; font-size: 0.9rem; font-weight: 500; margin: 0.125rem 0 0 0;">{sentiment}</p>
        </div>
    </div>
    <p style="color: #6B7280; font-size: 0.75rem; margin: 0.5rem 0 0.25rem 0;">Recent Developments:</p>
"""
    for news in market_context["news"][:2]:
        memo += f"""    <p style="color: #9CA3AF; font-size: 0.8rem; margin: 0.2rem 0; padding-left: 0.75rem; border-left: 2px solid #2d3139;">{news}</p>\n"""
    
    memo += f"""</div>

<div style="margin-bottom: 1rem;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Risk Factors</p>
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
"""
    for risk in market_context["risk_factors"]:
        memo += f"""        <span style="background: #FF475715; color: #FF4757; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">{risk}</span>\n"""
    
    memo += """    </div>
</div>

<div style="padding-top: 1rem; border-top: 1px solid #2d3139;">
    <p style="color: #4B5563; font-size: 0.7rem; font-style: italic; margin: 0;">
        Analysis generated via template engine. Configure OPENAI_API_KEY for AI-powered insights.
    </p>
</div>

</div>
"""
    
    return memo


if __name__ == "__main__":
    # Test the oracle
    test_shap_data = {
        "predicted_price": 950000,
        "base_value": 850000,
        "shap_values": {
            "sqft": 75000,
            "bedrooms": 25000,
            "year_built": -10000,
            "condition": 10000,
        },
        "feature_values": {
            "sqft": 1800,
            "bedrooms": 3,
            "year_built": 1965,
            "condition": 4,
        }
    }
    
    memo = generate_investment_memo(
        price=950000,
        shap_data=test_shap_data,
        zip_code="94107",
    )
    
    print(memo)
