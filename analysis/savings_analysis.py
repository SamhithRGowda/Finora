"""
savings_analysis.py — Savings Behavior Analysis (FIXED for multi-month data).

KEY FIX: All calculations now use monthly_spend (normalized),
NOT total_spend across all months.
"""

import pandas as pd
from utils.normalizer import normalize_to_monthly

RECOMMENDED_SAVINGS_RATE = 30.0


def analyze_savings_behavior(df: pd.DataFrame, monthly_income: float) -> dict:
    """
    Calculate savings rate using MONTHLY normalized spend.

    Args:
        df:             Categorized transactions DataFrame.
        monthly_income: User's monthly income in ₹.

    Returns dict with savings metrics (all monthly values).
    """
    # ── Normalize to monthly (THE KEY FIX) ───────────────────────────────────
    norm = normalize_to_monthly(df, monthly_income)

    monthly_spend   = norm["monthly_spend"]
    monthly_savings = norm["monthly_savings"]
    savings_rate    = norm["savings_rate"]
    num_months      = norm["num_months"]
    total_spend     = norm["total_spend"]

    insights = []

    # ── Status & insights ─────────────────────────────────────────────────────
    if savings_rate >= RECOMMENDED_SAVINGS_RATE:
        status = "good"
        insights.append(
            f"✅ Excellent! Your monthly savings rate is **{savings_rate:.1f}%** — "
            f"above the recommended {RECOMMENDED_SAVINGS_RATE}%."
        )
    elif savings_rate >= 15:
        status = "okay"
        insights.append(
            f"📈 Your savings rate is **{savings_rate:.1f}%**. "
            f"Aim for {RECOMMENDED_SAVINGS_RATE}% for financial security."
        )
    elif savings_rate >= 0:
        status = "poor"
        shortfall = (RECOMMENDED_SAVINGS_RATE / 100 * monthly_income) - monthly_savings
        insights.append(
            f"⚠️ Your savings rate is **{savings_rate:.1f}%** — "
            f"below the recommended {RECOMMENDED_SAVINGS_RATE}%. "
            f"Cut expenses by ₹{shortfall:,.0f}/month to hit 30%."
        )
    else:
        status = "poor"
        insights.append(
            f"🚨 You're spending **₹{abs(monthly_savings):,.0f} more** than you earn monthly! "
            f"Immediate action needed."
        )

    # ── Emergency fund ────────────────────────────────────────────────────────
    ef_target = monthly_income * 6
    months_to_ef = ef_target / max(monthly_savings, 1)
    insights.append(
        f"🏦 Emergency fund target (6× income): ₹{ef_target:,.0f}. "
        f"At ₹{max(monthly_savings,0):,.0f}/month savings, "
        f"you'd reach it in {months_to_ef:.0f} months."
    )

    # ── 50/30/20 rule ─────────────────────────────────────────────────────────
    needs_cats  = ["Utilities & Bills", "Healthcare", "Credit Card & Banking"]
    wants_cats  = ["Food & Dining", "Shopping", "Transport", "Entertainment"]

    cat_monthly = norm["category_monthly"]
    needs_spend = sum(cat_monthly.get(c, 0) for c in needs_cats)
    wants_spend = sum(cat_monthly.get(c, 0) for c in wants_cats)

    insights.append(
        f"📐 **50/30/20 Rule (monthly):**\n"
        f"  • Needs: ₹{needs_spend:,.0f} (budget ₹{monthly_income*0.5:,.0f}) "
        f"{'✅' if needs_spend <= monthly_income*0.5 else '❌'}\n"
        f"  • Wants: ₹{wants_spend:,.0f} (budget ₹{monthly_income*0.3:,.0f}) "
        f"{'✅' if wants_spend <= monthly_income*0.3 else '❌'}\n"
        f"  • Savings target: ₹{monthly_income*0.2:,.0f}/month"
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    if savings_rate >= RECOMMENDED_SAVINGS_RATE:
        recommendation = "Invest your surplus in index funds or increase your SIP amount."
    elif savings_rate >= 15:
        shortfall = (RECOMMENDED_SAVINGS_RATE / 100 * monthly_income) - monthly_savings
        recommendation = f"Reduce spending by ₹{shortfall:,.0f}/month to reach 30% savings rate."
    else:
        recommendation = "Cut your top spending category first. ₹2,000/month saved = ₹24,000/year."

    return {
        # Monthly values (use these everywhere)
        "total_spent":    monthly_spend,      # ← monthly avg (renamed for compatibility)
        "savings":        monthly_savings,
        "savings_rate":   savings_rate,
        "status":         status,
        "insights":       insights,
        "recommendation": recommendation,
        # Extra context
        "monthly_spend":  monthly_spend,
        "total_raw_spend": total_spend,
        "num_months":     num_months,
        "data_period":    norm["data_period_label"],
    }