"""
spending_patterns.py — Detect spending trends and patterns across categories.
"""

import pandas as pd


def detect_spending_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze spending trends across categories and time.

    Returns a dict with:
      - category_totals     : total per category
      - top_category        : highest spending category
      - patterns            : list of human-readable insight strings
    """
    if df.empty:
        return {"category_totals": {}, "top_category": "N/A", "patterns": []}

    patterns = []
    category_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    total_spend = df["Amount"].sum()

    # ── Top category ──────────────────────────────────────────────────────────
    top_cat = category_totals.index[0]
    top_amt = category_totals.iloc[0]
    top_pct = (top_amt / total_spend * 100) if total_spend > 0 else 0
    patterns.append(
        f"📊 Your biggest spending category is **{top_cat}** — "
        f"₹{top_amt:,.0f} ({top_pct:.1f}% of total spending)."
    )

    # ── Category share insights ───────────────────────────────────────────────
    for cat, amt in category_totals.items():
        pct = (amt / total_spend * 100) if total_spend > 0 else 0
        if cat == "Food & Dining" and pct > 30:
            patterns.append(
                f"🍔 Food & Dining takes up {pct:.1f}% of your spending (₹{amt:,.0f}). "
                f"Consider meal prepping to cut this down."
            )
        if cat == "Entertainment" and pct > 15:
            patterns.append(
                f"🎬 Entertainment is {pct:.1f}% of your budget (₹{amt:,.0f}). "
                f"Review your subscriptions — cancel ones you rarely use."
            )
        if cat == "Shopping" and pct > 25:
            patterns.append(
                f"🛍️ Shopping accounts for {pct:.1f}% of spending (₹{amt:,.0f}). "
                f"Try a 48-hour rule before impulse purchases."
            )

    # ── Transaction frequency ─────────────────────────────────────────────────
    num_txns = len(df)
    avg_txn = df["Amount"].mean()
    patterns.append(
        f"🔢 You made **{num_txns} transactions** with an average of ₹{avg_txn:,.0f} each."
    )

    # ── High value transactions ───────────────────────────────────────────────
    high_thresh = df["Amount"].mean() + 2 * df["Amount"].std()
    high_txns = df[df["Amount"] > high_thresh]
    if not high_txns.empty:
        patterns.append(
            f"💸 {len(high_txns)} transaction(s) were significantly above your average — "
            f"worth reviewing: {', '.join(high_txns['Description'].head(3).tolist())}."
        )

    return {
        "category_totals": category_totals.to_dict(),
        "top_category": top_cat,
        "patterns": patterns,
    }
