"""
normalizer.py — Monthly spend normalization for Finora.

The core problem this solves:
  - User uploads 6 months of data
  - Total spend = ₹1,51,859 across 6 months
  - Monthly income = ₹50,000
  - WRONG: savings = 50000 - 151859 = -101859 (shows as overspending!)
  - RIGHT:  monthly_spend = 151859 / 6 = 25310
            savings = 50000 - 25310 = 24690 ✅

This module detects how many months are in the dataframe,
then normalizes all spending to monthly averages.
"""

import pandas as pd


def get_num_months(df: pd.DataFrame) -> int:
    """
    Detect the number of unique months in the transaction data.

    Strategy:
      1. Parse the Date column
      2. Count unique year-month periods
      3. Default to 1 if dates are invalid or missing

    Returns: int (minimum 1 to avoid division errors)
    """
    if df is None or df.empty or "Date" not in df.columns:
        return 1

    try:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if dates.empty:
            return 1
        num_months = dates.dt.to_period("M").nunique()
        return max(int(num_months), 1)
    except Exception:
        return 1


def normalize_to_monthly(df: pd.DataFrame, monthly_income: float) -> dict:
    """
    Compute all monthly-normalized financial metrics from a transaction df.

    Args:
        df:             Categorized transactions DataFrame.
        monthly_income: User's monthly income (₹).

    Returns dict with:
        num_months        — number of months detected in data
        total_spend       — raw total across all months
        monthly_spend     — total_spend / num_months  ← KEY FIX
        monthly_income    — same as input
        monthly_savings   — monthly_income - monthly_spend
        savings_rate      — (monthly_savings / monthly_income) * 100
        category_monthly  — dict of category → monthly average spend
        data_period_label — human-readable string e.g. "6 months (Oct 2024 – Mar 2025)"
    """
    if df is None or df.empty:
        return {
            "num_months":       1,
            "total_spend":      0,
            "monthly_spend":    0,
            "monthly_income":   monthly_income,
            "monthly_savings":  monthly_income,
            "savings_rate":     100.0,
            "category_monthly": {},
            "data_period_label": "No data",
        }

    num_months  = get_num_months(df)
    total_spend = float(df["Amount"].sum())

    # ── Core fix: normalize to monthly ───────────────────────────────────────
    monthly_spend   = total_spend / num_months
    monthly_savings = monthly_income - monthly_spend
    savings_rate    = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0

    # ── Per-category monthly averages ─────────────────────────────────────────
    category_monthly = {}
    if "Category" in df.columns:
        cat_totals = df.groupby("Category")["Amount"].sum()
        category_monthly = {cat: amt / num_months for cat, amt in cat_totals.items()}

    # ── Human-readable period label ───────────────────────────────────────────
    try:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if not dates.empty:
            start = dates.min().strftime("%b %Y")
            end   = dates.max().strftime("%b %Y")
            data_period_label = f"{num_months} month{'s' if num_months > 1 else ''} ({start} – {end})"
        else:
            data_period_label = f"{num_months} month{'s' if num_months > 1 else ''}"
    except Exception:
        data_period_label = f"{num_months} month{'s' if num_months > 1 else ''}"

    return {
        "num_months":        num_months,
        "total_spend":       total_spend,
        "monthly_spend":     monthly_spend,
        "monthly_income":    monthly_income,
        "monthly_savings":   monthly_savings,
        "savings_rate":      savings_rate,
        "category_monthly":  category_monthly,
        "data_period_label": data_period_label,
    }