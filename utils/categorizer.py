"""
categorizer.py — Merchant-based category correction for Finora.

Problem this solves:
  LLMs (Groq/NVIDIA) sometimes miscategorize merchants:
  e.g. Flipkart → "Food & Dining", BigBasket → "Transport"

Solution:
  After LLM categorization, override Category using merchant name
  in the Description column. Merchant names are ground truth.

Used by:
  - app.py (applied once after LLM categorization, fixes all tabs)
  - rag_engine.py (applied in filter_df before category filtering)
"""

import pandas as pd


# Merchant → correct Category column value
# These override whatever the LLM assigned
MERCHANT_CATEGORY_OVERRIDE = {
    # Food & Dining
    "swiggy":          "Food & Dining",
    "zomato":          "Food & Dining",
    "bigbasket":       "Food & Dining",
    "blinkit":         "Food & Dining",
    "instamart":       "Food & Dining",
    "dominos":         "Food & Dining",
    "kfc":             "Food & Dining",
    # Transport
    "uber":            "Transport",
    "ola":             "Transport",
    "rapido":          "Transport",
    # Shopping
    "amazon":          "Shopping",
    "flipkart":        "Shopping",
    "myntra":          "Shopping",
    "ajio":            "Shopping",
    "makemytrip":      "Shopping",
    # Entertainment
    "netflix":         "Entertainment",
    "spotify":         "Entertainment",
    "hotstar":         "Entertainment",
    # Utilities & Bills
    "airtel":          "Utilities & Bills",
    "jio":             "Utilities & Bills",
    # Healthcare
    "apollo":          "Healthcare",
    "medplus":         "Healthcare",
    "1mg":             "Healthcare",
    # Credit Card & Banking
    "emi credit card": "Credit Card & Banking",
    "neft/hdfc bank":  "Credit Card & Banking",
    "hdfc bank/emi":   "Credit Card & Banking",
    # P2P transfers — purpose unknown
    "9845012345":      "Other",
    "8800123456":      "Other",
}


def correct_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Override LLM-assigned Category values using merchant name in Description.

    Args:
        df: DataFrame with 'Description' and 'Category' columns.

    Returns:
        DataFrame with corrected Category values.

    Example:
        Before: Flipkart | Food & Dining   (LLM mistake)
        After:  Flipkart | Shopping        (merchant override)
    """
    if "Description" not in df.columns or "Category" not in df.columns:
        return df

    df = df.copy()

    def _override(row) -> str:
        # Skip rows the user manually labelled — never overwrite user input
        if row.get("_manual", False):
            return row["Category"]
        desc_lower = str(row["Description"]).lower()
        for merchant, correct_cat in MERCHANT_CATEGORY_OVERRIDE.items():
            if merchant in desc_lower:
                return correct_cat
        return row["Category"]   # keep LLM category if no match

    df["Category"] = df.apply(_override, axis=1)
    return df