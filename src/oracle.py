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
    
    # Build the memo
    memo = f"""## 📊 Investment Memo - Simulation Mode

**⚠️ AI Oracle Offline** - Using template analysis. Configure OPENAI_API_KEY for full AI insights.

---

### Valuation Summary
The model values this property at **${price:,.0f}**, which is **${price - base_value:+,.0f}** relative to the market baseline of ${base_value:,.0f}.

### Key Value Drivers
"""
    
    if positive_drivers:
        memo += "\n**Positive Factors:**\n"
        for feat, val in positive_drivers[:2]:
            memo += f"- {feat.replace('_', ' ').title()}: +${val:,.0f}\n"
    
    if negative_drivers:
        memo += "\n**Negative Factors:**\n"
        for feat, val in negative_drivers[:2]:
            memo += f"- {feat.replace('_', ' ').title()}: ${val:,.0f}\n"
    
    memo += f"""
### Market Context - {neighborhood}
**Current Sentiment:** {sentiment}

**Local News:**
"""
    for news in market_context["news"][:2]:
        memo += f"- {news}\n"
    
    memo += f"""
### Risk Assessment
"""
    for risk in market_context["risk_factors"]:
        memo += f"- ⚠️ {risk}\n"
    
    memo += """
---
*This is a simulated analysis. Enable AI for comprehensive investment insights.*
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
