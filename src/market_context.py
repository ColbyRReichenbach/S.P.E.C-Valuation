"""
S.P.E.C. Valuation Engine - Market Context Module
==================================================
Provides real-time market context including interest rates,
economic indicators, and market conditions for investment analysis.

Currently uses static data that can be upgraded to live APIs (FRED, etc.)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ====================================
# CURRENT MARKET DATA (Updated: Jan 2025)
# ====================================
# Source: Freddie Mac Primary Mortgage Market Survey
# Update these periodically or integrate with FRED API
CURRENT_RATES = {
    "30_year_fixed": 6.89,
    "15_year_fixed": 6.12,
    "5_1_arm": 6.25,
    "last_updated": "2025-01-09",
    "source": "Freddie Mac PMMS",
}

# Historical context for comparison
RATE_HISTORY = {
    "one_year_ago": 6.66,  # Jan 2024
    "two_years_ago": 6.33,  # Jan 2023
    "five_years_ago": 3.65,  # Jan 2020
    "ten_year_avg": 4.25,
    "historical_low": 2.65,  # Jan 2021
    "historical_high": 18.63,  # Oct 1981
}

# Market sentiment based on rate environment
RATE_IMPACT = {
    "high": {  # Above 7%
        "sentiment": "Challenging",
        "buyer_impact": "Significantly reduced purchasing power",
        "seller_impact": "Longer days on market expected",
        "investment_note": "Focus on cash flow, not appreciation",
    },
    "elevated": {  # 6-7%
        "sentiment": "Cautious",
        "buyer_impact": "Reduced affordability vs 2021-2022",
        "seller_impact": "Price sensitivity increasing",
        "investment_note": "Negotiate aggressively, expect slower appreciation",
    },
    "moderate": {  # 5-6%
        "sentiment": "Balanced",
        "buyer_impact": "Moderate affordability pressure",
        "seller_impact": "Normal market dynamics",
        "investment_note": "Standard underwriting applies",
    },
    "favorable": {  # Below 5%
        "sentiment": "Bullish",
        "buyer_impact": "Strong purchasing power",
        "seller_impact": "Multiple offers likely",
        "investment_note": "Act quickly, expect competition",
    },
}


def get_current_rates() -> Dict[str, Any]:
    """
    Get current mortgage rates and context.
    
    Returns:
        Dictionary with current rates and metadata.
    """
    return CURRENT_RATES.copy()


def get_rate_context() -> Dict[str, Any]:
    """
    Get full rate context including historical comparison.
    
    Returns:
        Dictionary with rates, history, and analysis.
    """
    current_30yr = CURRENT_RATES["30_year_fixed"]
    
    # Determine rate environment
    if current_30yr >= 7.0:
        environment = "high"
    elif current_30yr >= 6.0:
        environment = "elevated"
    elif current_30yr >= 5.0:
        environment = "moderate"
    else:
        environment = "favorable"
    
    impact = RATE_IMPACT[environment]
    
    # Calculate changes
    yoy_change = current_30yr - RATE_HISTORY["one_year_ago"]
    vs_10yr_avg = current_30yr - RATE_HISTORY["ten_year_avg"]
    
    return {
        "current_rates": CURRENT_RATES,
        "history": RATE_HISTORY,
        "environment": environment,
        "impact": impact,
        "analysis": {
            "yoy_change": yoy_change,
            "yoy_direction": "up" if yoy_change > 0 else "down",
            "vs_10yr_avg": vs_10yr_avg,
            "vs_10yr_direction": "above" if vs_10yr_avg > 0 else "below",
        }
    }


def calculate_monthly_payment(
    price: float,
    down_payment_pct: float = 0.20,
    rate: Optional[float] = None,
    term_years: int = 30,
) -> Dict[str, float]:
    """
    Calculate monthly mortgage payment.
    
    Args:
        price: Property price.
        down_payment_pct: Down payment as decimal (0.20 = 20%).
        rate: Annual interest rate (uses current 30yr if None).
        term_years: Loan term in years.
    
    Returns:
        Dictionary with payment breakdown.
    """
    if rate is None:
        rate = CURRENT_RATES["30_year_fixed"]
    
    loan_amount = price * (1 - down_payment_pct)
    monthly_rate = rate / 100 / 12
    num_payments = term_years * 12
    
    # Calculate monthly payment (principal + interest)
    if monthly_rate > 0:
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments
    
    # Estimate taxes and insurance (rough SF estimates)
    monthly_tax = (price * 0.0115) / 12  # ~1.15% property tax in SF
    monthly_insurance = (price * 0.003) / 12  # ~0.3% homeowners insurance
    
    total_monthly = monthly_payment + monthly_tax + monthly_insurance
    
    return {
        "loan_amount": loan_amount,
        "down_payment": price * down_payment_pct,
        "monthly_pi": monthly_payment,
        "monthly_tax": monthly_tax,
        "monthly_insurance": monthly_insurance,
        "total_monthly": total_monthly,
        "rate_used": rate,
    }


def get_affordability_context(price: float) -> str:
    """
    Generate affordability context for investment memo.
    
    Args:
        price: Property price.
    
    Returns:
        Formatted string for investment memo.
    """
    rate_ctx = get_rate_context()
    payment = calculate_monthly_payment(price)
    
    current_rate = rate_ctx["current_rates"]["30_year_fixed"]
    environment = rate_ctx["environment"]
    impact = rate_ctx["impact"]
    analysis = rate_ctx["analysis"]
    
    context = f"""
**Interest Rate Environment: {impact['sentiment'].upper()}**

Current 30-Year Fixed: {current_rate:.2f}% (as of {rate_ctx['current_rates']['last_updated']})
- Year-over-Year: {analysis['yoy_change']:+.2f}% ({analysis['yoy_direction']})
- vs 10-Year Average: {analysis['vs_10yr_avg']:+.2f}% ({analysis['vs_10yr_direction']})

**Monthly Payment Estimate** (20% down):
- Principal & Interest: ${payment['monthly_pi']:,.0f}
- Est. Taxes: ${payment['monthly_tax']:,.0f}
- Est. Insurance: ${payment['monthly_insurance']:,.0f}
- **Total PITI: ${payment['total_monthly']:,.0f}/month**

**Rate Impact Assessment:**
- Buyer Impact: {impact['buyer_impact']}
- Investment Note: {impact['investment_note']}
"""
    return context.strip()


if __name__ == "__main__":
    # Test the module
    print("=" * 50)
    print("Market Context Module Test")
    print("=" * 50)
    
    print("\nCurrent Rates:")
    rates = get_current_rates()
    for key, value in rates.items():
        print(f"  {key}: {value}")
    
    print("\nRate Context:")
    ctx = get_rate_context()
    print(f"  Environment: {ctx['environment']}")
    print(f"  Sentiment: {ctx['impact']['sentiment']}")
    
    print("\nAffordability for $1.5M property:")
    print(get_affordability_context(1_500_000))
