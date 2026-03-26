"""
action_plan_generator.py — Generate a personalized AI financial action plan.
Uses the same multi-provider LLM setup (NVIDIA / Groq / Gemini).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_financial_action_plan(data_summary: dict) -> str:
    """
    Generate a step-by-step personalized financial action plan using the LLM.

    Args:
        data_summary: Dict containing aggregated findings from all analysis modules:
          - monthly_income
          - total_spent
          - savings_rate
          - top_category
          - anomaly_count
          - subscriptions_cost
          - patterns         (list of strings)
          - anomalies        (list of dicts)
          - subscriptions    (list of dicts)

    Returns:
        Markdown-formatted action plan string.
    """
    from ai.gemini_ai import _call_llm

    income          = data_summary.get("monthly_income", 0)
    spent           = data_summary.get("total_spent", 0)
    savings_rate    = data_summary.get("savings_rate", 0)
    top_cat         = data_summary.get("top_category", "Unknown")
    anomaly_count   = data_summary.get("anomaly_count", 0)
    sub_cost        = data_summary.get("subscriptions_cost", 0)
    patterns        = "\n".join(data_summary.get("patterns", []))
    anomaly_descs   = ", ".join([a.get("description","") for a in data_summary.get("anomalies", [])])
    sub_names       = ", ".join([s.get("name","") for s in data_summary.get("subscriptions", [])])

    prompt = f"""You are Finora, an expert AI financial advisor for Indian users.

Based on this user's complete financial analysis, generate a **personalized step-by-step financial action plan**.

--- USER FINANCIAL SUMMARY ---
Monthly Income      : ₹{income:,.0f}
Total Spent         : ₹{spent:,.0f}
Savings Rate        : {savings_rate:.1f}%
Top Spending Categ. : {top_cat}
Anomalies Detected  : {anomaly_count}
Subscriptions Cost  : ₹{sub_cost:,.0f}/month
Subscriptions Found : {sub_names if sub_names else 'None'}
Unusual Transactions: {anomaly_descs if anomaly_descs else 'None'}

Spending Patterns:
{patterns if patterns else 'No patterns detected'}
------------------------------

Generate a financial action plan in this exact markdown format:

## 🎯 Your Personalized Financial Action Plan

### 📍 Current Financial Status
2-3 sentences summarizing their financial health honestly.

### 🚀 Immediate Actions (This Week)
Numbered list of 2-3 things they should do RIGHT NOW.
Be very specific — mention actual ₹ amounts and merchant names from their data.

### 📅 Short-Term Goals (Next 30 Days)
Numbered list of 2-3 goals for this month with specific targets.

### 📈 Long-Term Strategy (Next 12 Months)
Numbered list of 2-3 bigger financial goals with realistic milestones.

### 💡 One Key Insight
One powerful, memorable insight about their financial behavior.
Make it specific to their data — not generic advice.

Keep total under 350 words. Use ₹ symbols. Reference actual numbers from their data.
Be direct, specific, and motivating — like a real financial advisor."""

    try:
        return _call_llm(prompt, max_tokens=1200)
    except Exception as e:
        # Fallback: rule-based action plan if LLM unavailable
        return _fallback_action_plan(data_summary)


def _fallback_action_plan(data_summary: dict) -> str:
    """Rule-based action plan when LLM is not available."""
    income       = data_summary.get("monthly_income", 0)
    savings_rate = data_summary.get("savings_rate", 0)
    top_cat      = data_summary.get("top_category", "N/A")
    sub_cost     = data_summary.get("subscriptions_cost", 0)
    anomalies    = data_summary.get("anomalies", [])

    steps = []
    step = 1

    if savings_rate < 20:
        steps.append(f"**Step {step} — Boost your savings rate**\nYour current savings rate is {savings_rate:.1f}%. Aim for at least 20% by reducing {top_cat} spending.")
        step += 1

    if sub_cost > 500:
        steps.append(f"**Step {step} — Audit subscriptions**\nYou're spending ₹{sub_cost:,.0f}/month on subscriptions. Review and cancel at least one.")
        step += 1

    if anomalies:
        steps.append(f"**Step {step} — Review unusual transactions**\n{len(anomalies)} unusual transaction(s) detected. Verify these are legitimate.")
        step += 1

    steps.append(f"**Step {step} — Start an emergency fund**\nTarget 6 months of expenses = ₹{income * 6:,.0f}.")
    step += 1

    steps.append(f"**Step {step} — Start a SIP**\nInvest at least ₹1,000/month in an index fund. Even small amounts compound significantly over 10 years.")

    plan = "## 🎯 Your Financial Action Plan\n\n"
    plan += "\n\n".join(steps)
    plan += "\n\n---\n*Add your API key in the sidebar for a fully personalized AI-generated plan.*"
    return plan
