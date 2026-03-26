"""
helpers.py — Shared utility functions for Finora
"""

import pandas as pd


def format_inr(amount: float) -> str:
    """Format a number as Indian Rupees."""
    return f"₹{amount:,.0f}"


def dataframe_is_valid(df) -> bool:
    """Check if a DataFrame is non-empty and has the required columns."""
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    required = {"Date", "Description", "Amount"}
    return required.issubset(df.columns)
