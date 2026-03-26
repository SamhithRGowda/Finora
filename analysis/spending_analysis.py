"""
spending_analysis.py — Week 2: Spending Analysis Engine
Categorizes transactions, summarizes spending, and generates insights.
"""

import pandas as pd


# ─────────────────────────────────────────────
#  Category keyword rules (case-insensitive)
# ─────────────────────────────────────────────
CATEGORY_RULES = {
    "Food & Dining": [
        "swiggy", "zomato", "uber eats", "dominos", "pizza", "kfc", "mcdonalds",
        "restaurant", "cafe", "food", "biryani", "burger", "instamart", "bigbasket",
        "grofers", "blinkit", "dunzo", "eat", "dine",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "snapdeal",
        "shopsy", "tatacliq", "reliance", "retail", "mall", "shop", "store",
        "purchase", "sale",
    ],
    "Transport": [
        "uber", "ola", "rapido", "metro", "irctc", "makemytrip", "goibibo",
        "redbus", "petrol", "fuel", "cab", "taxi", "ride", "bus", "train",
        "flight", "indigo", "spicejet", "airindia",
    ],
    "Entertainment": [
        "netflix", "hotstar", "prime", "spotify", "youtube", "zee5", "sony",
        "bookmyshow", "pvr", "inox", "gaming", "steam", "playstation",
        "subscription", "gold", "membership",
    ],
    "Utilities & Bills": [
        "electricity", "water", "gas", "internet", "broadband", "airtel", "jio",
        "vodafone", "bsnl", "recharge", "bill", "emi", "loan", "insurance",
        "paytm", "phonepe", "gpay",
    ],
    "Healthcare": [
        "apollo", "medplus", "pharmeasy", "1mg", "netmeds", "hospital",
        "clinic", "doctor", "pharmacy", "medicine", "health",
    ],
    "Credit Card & Banking": [
        "credit card", "hdfc", "icici", "sbi", "axis", "kotak", "bank",
        "transfer", "neft", "imps", "rtgs",
    ],
}

OTHER_CATEGORY = "Other"


# ─────────────────────────────────────────────
#  Public Functions
# ─────────────────────────────────────────────

def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Category' column to the transactions DataFrame.

    Args:
        df: DataFrame with at least a 'Description' column.

    Returns:
        DataFrame with an added 'Category' column.
    """
    df = df.copy()
    df["Category"] = df["Description"].apply(_assign_category)
    return df


def generate_spending_summary(df: pd.DataFrame) -> pd.Series:
    """
    Compute total spending per category.

    Args:
        df: Categorized DataFrame (must have 'Category' and 'Amount' columns).

    Returns:
        A pandas Series: category → total amount, sorted descending.
    """
    summary = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    return summary


def generate_spending_insights(df: pd.DataFrame, monthly_income: float = 50000) -> list:
    """
    Generate human-readable financial insights.

    Args:
        df:             Categorized transactions DataFrame.
        monthly_income: User's approximate monthly income (default ₹50,000).

    Returns:
        A list of insight strings.
    """
    insights = []
    summary = generate_spending_summary(df)
    total_spending = summary.sum()

    if total_spending == 0:
        return ["No spending data available to generate insights."]

    # ── Insight 1: Total spending vs income ──────────────────────────────────
    savings = monthly_income - total_spending
    savings_pct = (savings / monthly_income) * 100 if monthly_income > 0 else 0
    spend_pct = (total_spending / monthly_income) * 100 if monthly_income > 0 else 0

    if savings >= 0:
        insights.append(
            f"💰 You spent ₹{total_spending:,.0f} this period "
            f"({spend_pct:.1f}% of ₹{monthly_income:,.0f} income), "
            f"saving ₹{savings:,.0f} ({savings_pct:.1f}%)."
        )
    else:
        insights.append(
            f"⚠️ You overspent by ₹{abs(savings):,.0f}! "
            f"Total spending (₹{total_spending:,.0f}) exceeded income (₹{monthly_income:,.0f})."
        )

    # ── Insight 2: Biggest spending category ─────────────────────────────────
    top_category = summary.index[0]
    top_amount = summary.iloc[0]
    top_pct = (top_amount / total_spending) * 100
    insights.append(
        f"📊 Your biggest spending category is '{top_category}' "
        f"at ₹{top_amount:,.0f} ({top_pct:.1f}% of total spending)."
    )

    # ── Insight 3: Food spending warning ─────────────────────────────────────
    food_spend = summary.get("Food & Dining", 0)
    food_pct_income = (food_spend / monthly_income) * 100 if monthly_income > 0 else 0
    if food_spend > 0:
        if food_pct_income > 30:
            insights.append(
                f"🍔 You're spending {food_pct_income:.1f}% of your income on food. "
                f"Consider cooking at home more often to cut costs."
            )
        else:
            insights.append(
                f"🍽️ Food spending is ₹{food_spend:,.0f} ({food_pct_income:.1f}% of income) — within a healthy range."
            )

    # ── Insight 4: Savings rate advice ───────────────────────────────────────
    if savings_pct >= 20:
        insights.append("✅ Great job! You're saving more than 20% of your income — keep it up!")
    elif 10 <= savings_pct < 20:
        insights.append(
            "📈 You're saving between 10–20% of income. "
            "Aim for 20%+ by trimming discretionary spending."
        )
    elif 0 <= savings_pct < 10:
        insights.append(
            "🔔 Your savings rate is below 10%. "
            "Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings."
        )

    # ── Insight 5: Entertainment warning ─────────────────────────────────────
    entertainment_spend = summary.get("Entertainment", 0)
    if entertainment_spend > 0:
        ent_pct = (entertainment_spend / total_spending) * 100
        if ent_pct > 15:
            insights.append(
                f"🎬 Entertainment costs ₹{entertainment_spend:,.0f} ({ent_pct:.1f}% of spending). "
                f"Review your subscriptions — cancel ones you rarely use."
            )

    # ── Insight 6: Number of transactions ────────────────────────────────────
    num_txns = len(df)
    insights.append(f"📋 You made {num_txns} transactions in this period.")

    return insights


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────

def _assign_category(description: str) -> str:
    """Match a description string against keyword rules and return a category."""
    desc_lower = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return OTHER_CATEGORY
