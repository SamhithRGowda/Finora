"""
financial_score.py — Calculate a Financial Health Score (0–100).

FIXED: Uses monthly normalized spend instead of raw total spend.
This ensures correct scoring when data spans multiple months.

Factors:
  1. Savings Rate          (max 30 pts)
  2. Spending Stability    (max 20 pts)
  3. Subscription Burden   (max 20 pts)
  4. Emergency Fund Est.   (max 15 pts)
  5. Anomaly Frequency     (max 15 pts)
"""

import pandas as pd
from analysis.anomaly_detection import detect_anomalies
from analysis.subscriptions import detect_subscriptions
from utils.normalizer import normalize_to_monthly


def calculate_financial_score(df: pd.DataFrame, monthly_income: float) -> dict:
    """
    Calculate a Financial Health Score from 0–100.
    Uses monthly normalized spend — correct for multi-month data.
    """
    # ── Normalize spend to monthly (KEY FIX) ─────────────────────────────────
    norm         = normalize_to_monthly(df, monthly_income)
    monthly_spend = norm["monthly_spend"]
    savings       = norm["monthly_savings"]
    savings_rate  = norm["savings_rate"]

    breakdown  = {}
    max_points = {}
    tips       = []

    # ── Factor 1: Savings Rate (30 pts) ──────────────────────────────────────
    max_points["Savings Rate"] = 30

    if savings_rate >= 30:
        s1 = 30
    elif savings_rate >= 20:
        s1 = 20
    elif savings_rate >= 10:
        s1 = 10
    elif savings_rate >= 0:
        s1 = 5
    else:
        s1 = 0  # overspending

    breakdown["Savings Rate"] = s1
    if s1 < 20:
        tips.append(f"💰 Boost your savings rate (currently {savings_rate:.1f}%). Target 30% by cutting discretionary spend.")

    # ── Factor 2: Spending Stability (20 pts) ────────────────────────────────
    max_points["Spending Stability"] = 20
    if not df.empty and len(df) > 1:
        std_dev = df["Amount"].std()
        mean    = df["Amount"].mean()
        cv      = (std_dev / mean) if mean > 0 else 0   # coefficient of variation

        if cv < 0.5:
            s2 = 20    # very stable
        elif cv < 1.0:
            s2 = 15
        elif cv < 1.5:
            s2 = 10
        elif cv < 2.0:
            s2 = 5
        else:
            s2 = 0     # highly erratic spending
    else:
        s2 = 10   # not enough data — neutral

    breakdown["Spending Stability"] = s2
    if s2 < 10:
        tips.append("📉 Your spending is highly variable. Try budgeting fixed amounts per category each month.")

    # ── Factor 3: Subscription Burden (20 pts) ───────────────────────────────
    max_points["Subscription Burden"] = 20
    sub_result   = detect_subscriptions(df)
    sub_cost     = sub_result["total_monthly"]
    sub_pct      = (sub_cost / monthly_income * 100) if monthly_income > 0 else 0

    if sub_pct < 5:
        s3 = 20
    elif sub_pct < 10:
        s3 = 15
    elif sub_pct < 15:
        s3 = 10
    elif sub_pct < 20:
        s3 = 5
    else:
        s3 = 0    # subscriptions eating >20% of income

    breakdown["Subscription Burden"] = s3
    if s3 < 15:
        tips.append(f"🔄 Subscriptions cost ₹{sub_cost:,.0f}/month ({sub_pct:.1f}% of income). Audit and cancel unused ones.")

    # ── Factor 4: Emergency Fund Estimate (15 pts) ───────────────────────────
    max_points["Emergency Fund"] = 15
    # Estimate: if savings > 0 and savings rate is reasonable, they likely have some buffer
    if savings_rate >= 20:
        s4 = 15   # likely building a fund
    elif savings_rate >= 10:
        s4 = 10
    elif savings_rate >= 0:
        s4 = 5
    else:
        s4 = 0    # overspending = no fund possible

    breakdown["Emergency Fund"] = s4
    if s4 < 10:
        target = monthly_income * 6
        tips.append(f"🏦 Build an emergency fund of ₹{target:,.0f} (6× income). Start with ₹500/month.")

    # ── Factor 5: Anomaly Frequency (15 pts) ─────────────────────────────────
    max_points["Anomaly Frequency"] = 15
    anomaly_result = detect_anomalies(df)
    anomaly_count  = anomaly_result["count"]
    total_txns     = len(df) if not df.empty else 1
    anomaly_pct    = (anomaly_count / total_txns * 100)

    if anomaly_count == 0:
        s5 = 15
    elif anomaly_pct < 5:
        s5 = 10
    elif anomaly_pct < 10:
        s5 = 5
    else:
        s5 = 0

    breakdown["Anomaly Frequency"] = s5
    if s5 < 10 and anomaly_count > 0:
        tips.append(f"🚨 {anomaly_count} unusual transaction(s) detected. Review them in the Anomaly tab.")

    # ── Final Score ───────────────────────────────────────────────────────────
    score = s1 + s2 + s3 + s4 + s5

    # ── Grade ─────────────────────────────────────────────────────────────────
    if score >= 80:
        grade = "Excellent"
        grade_emoji = "🏆"
        grade_color = "#34d399"
    elif score >= 60:
        grade = "Good"
        grade_emoji = "✅"
        grade_color = "#60a5fa"
    elif score >= 40:
        grade = "Needs Improvement"
        grade_emoji = "⚠️"
        grade_color = "#fbbf24"
    else:
        grade = "Risky"
        grade_emoji = "🚨"
        grade_color = "#f87171"

    # ── Explanation ───────────────────────────────────────────────────────────
    strengths = [k for k, v in breakdown.items() if v >= max_points[k] * 0.75]
    weaknesses = [k for k, v in breakdown.items() if v < max_points[k] * 0.5]

    strength_str  = ", ".join(strengths) if strengths else "none"
    weakness_str  = ", ".join(weaknesses) if weaknesses else "none"

    explanation = (
        f"Your financial health score is **{score}/100** — {grade} {grade_emoji}. "
        f"Your savings rate is {savings_rate:.1f}%"
        f"{'✅' if savings_rate >= 20 else '⚠️'}. "
    )
    if weaknesses:
        explanation += f"Areas dragging your score: **{weakness_str}**. "
    if strengths:
        explanation += f"You're doing well in: **{strength_str}**."

    return {
        "score":           score,
        "grade":           grade,
        "grade_emoji":     grade_emoji,
        "grade_color":     grade_color,
        "score_breakdown": breakdown,
        "max_breakdown":   max_points,
        "savings_rate":    savings_rate,
        "sub_cost":        sub_cost,
        "anomaly_count":   anomaly_count,
        "explanation":     explanation,
        "tips":            tips if tips else ["🎉 Great job! Keep up your healthy financial habits."],
    }