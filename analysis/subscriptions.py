"""
subscriptions.py — Detect recurring subscription transactions.
Logic: same merchant keyword + similar amount + ~monthly interval
"""

import pandas as pd


# Known subscription keywords to look for
SUBSCRIPTION_KEYWORDS = [
    "netflix", "spotify", "hotstar", "amazon prime", "prime",
    "youtube", "zee5", "sonyliv", "apple", "google one",
    "adobe", "github", "notion", "dropbox", "canva",
    "swiggy one", "zomato gold", "linkedin", "chatgpt",
    "openai", "microsoft", "office", "icloud",
]


def detect_subscriptions(df: pd.DataFrame) -> dict:
    """
    Detect recurring subscription-like transactions.

    Returns:
        {
          "subscriptions": list of dicts with name/amount/frequency,
          "total_monthly": float,
          "insights": list of strings
        }
    """
    if df.empty:
        return {"subscriptions": [], "total_monthly": 0, "insights": []}

    found = []
    seen = set()

    for _, row in df.iterrows():
        desc_lower = str(row["Description"]).lower()
        for kw in SUBSCRIPTION_KEYWORDS:
            if kw in desc_lower and kw not in seen:
                found.append({
                    "name": row["Description"],
                    "amount": row["Amount"],
                    "frequency": "Monthly",
                    "keyword": kw,
                })
                seen.add(kw)
                break

    # Also detect by amount repetition — same amount appearing 2+ times
    amount_counts = df["Amount"].value_counts()
    repeated_amounts = amount_counts[amount_counts >= 2].index.tolist()

    # Keywords that are NOT subscriptions even if they repeat
    NOT_SUBSCRIPTION = [
        "uber", "ola", "swiggy", "zomato", "rapido",
        "amazon purchase", "flipkart", "myntra", "bigbasket",
        "credit card", "hdfc", "icici", "electricity",
    ]

    for amt in repeated_amounts:
        repeated_rows = df[df["Amount"] == amt]
        desc = repeated_rows.iloc[0]["Description"]
        desc_lower = desc.lower()
        already = any(s["amount"] == amt for s in found)
        is_not_sub = any(kw in desc_lower for kw in NOT_SUBSCRIPTION)
        if not already and not is_not_sub and amt < 2000:
            found.append({
                "name": desc,
                "amount": amt,
                "frequency": f"Recurring ({len(repeated_rows)}x found)",
                "keyword": "auto-detected",
            })

    total_monthly = sum(s["amount"] for s in found)

    insights = []
    if found:
        insights.append(
            f"🔄 Detected **{len(found)} recurring subscription(s)** "
            f"costing ₹{total_monthly:,.0f}/month."
        )
        if total_monthly > 1500:
            insights.append(
                f"⚠️ Your subscriptions cost ₹{total_monthly:,.0f}/month "
                f"(₹{total_monthly * 12:,.0f}/year). "
                f"Consider auditing and cancelling unused ones."
            )
        for s in found:
            insights.append(f"  • {s['name']} — ₹{s['amount']:,.0f} ({s['frequency']})")
    else:
        insights.append("✅ No recurring subscriptions detected in this period.")

    return {
        "subscriptions": found,
        "total_monthly": total_monthly,
        "insights": insights,
    }