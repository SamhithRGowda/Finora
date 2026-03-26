"""
savings_analysis.py — Analyze savings rate and savings behavior.
"""

import pandas as pd


RECOMMENDED_SAVINGS_RATE = 30.0   # % — standard financial advice


def analyze_savings_behavior(df: pd.DataFrame, monthly_income: float) -> dict:
    """
    Calculate savings rate and generate savings behavior insights.

    Args:
        df:             Categorized transactions DataFrame.
        monthly_income: User's monthly income in ₹.

    Returns:
        {
          "total_spent":    float,
          "savings":        float,
          "savings_rate":   float,
          "status":         "good" | "okay" | "poor",
          "insights":       list of strings,
          "recommendation": string
        }
    """
    total_spent = df["Amount"].sum() if not df.empty else 0
    savings = monthly_income - total_spent
    savings_rate = (savings / monthly_income * 100) if monthly_income > 0 else 0

    insights = []

    # ── Status ────────────────────────────────────────────────────────────────
    if savings_rate >= RECOMMENDED_SAVINGS_RATE:
        status = "good"
        insights.append(
            f"✅ Excellent! Your savings rate is **{savings_rate:.1f}%** — "
            f"above the recommended {RECOMMENDED_SAVINGS_RATE}%."
        )
    elif savings_rate >= 15:
        status = "okay"
        insights.append(
            f"📈 Your savings rate is **{savings_rate:.1f}%**. "
            f"You're on track but aim for {RECOMMENDED_SAVINGS_RATE}% for financial security."
        )
    elif savings_rate >= 0:
        status = "poor"
        insights.append(
            f"⚠️ Your savings rate is only **{savings_rate:.1f}%** — "
            f"below the recommended {RECOMMENDED_SAVINGS_RATE}%. "
            f"You need to cut expenses by ₹{(RECOMMENDED_SAVINGS_RATE/100 * monthly_income - savings):,.0f} more."
        )
    else:
        status = "poor"
        insights.append(
            f"🚨 You are **overspending** by ₹{abs(savings):,.0f}! "
            f"Your expenses exceed your income. Immediate action needed."
        )

    # ── Emergency fund ────────────────────────────────────────────────────────
    emergency_fund_target = monthly_income * 6
    insights.append(
        f"🏦 Your emergency fund target (6 months expenses) should be "
        f"₹{emergency_fund_target:,.0f}. "
        f"At your current savings of ₹{max(savings,0):,.0f}/month, "
        f"you'd reach it in {emergency_fund_target / max(savings, 1):.0f} months."
    )

    # ── 50/30/20 rule check ───────────────────────────────────────────────────
    needs_budget  = monthly_income * 0.50
    wants_budget  = monthly_income * 0.30
    savings_target = monthly_income * 0.20

    needs_cats  = ["Utilities & Bills", "Healthcare", "Credit Card & Banking"]
    wants_cats  = ["Food & Dining", "Shopping", "Transport", "Entertainment"]

    needs_spent = df[df["Category"].isin(needs_cats)]["Amount"].sum() if not df.empty else 0
    wants_spent = df[df["Category"].isin(wants_cats)]["Amount"].sum() if not df.empty else 0

    insights.append(
        f"📐 **50/30/20 Rule Check:**\n"
        f"  • Needs: ₹{needs_spent:,.0f} (budget ₹{needs_budget:,.0f})"
        f"{'✅' if needs_spent <= needs_budget else '❌'}\n"
        f"  • Wants: ₹{wants_spent:,.0f} (budget ₹{wants_budget:,.0f})"
        f"{'✅' if wants_spent <= wants_budget else '❌'}\n"
        f"  • Savings target: ₹{savings_target:,.0f}"
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    if savings_rate >= RECOMMENDED_SAVINGS_RATE:
        recommendation = "Start investing your surplus in index funds or increase your SIP amount."
    elif savings_rate >= 15:
        recommendation = f"Reduce discretionary spending by ₹{(RECOMMENDED_SAVINGS_RATE/100 * monthly_income - savings):,.0f} to hit 30% savings rate."
    else:
        recommendation = "Focus on cutting your top spending category first. Even ₹2000/month saved = ₹24,000/year."

    return {
        "total_spent":    total_spent,
        "savings":        savings,
        "savings_rate":   savings_rate,
        "status":         status,
        "insights":       insights,
        "recommendation": recommendation,
    }
