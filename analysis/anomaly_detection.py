"""
anomaly_detection.py — Detect unusual transactions using statistical rules.
Rule: transaction > 3x the category average = anomaly
"""

import pandas as pd


def detect_anomalies(df: pd.DataFrame, threshold_multiplier: float = 3.0) -> dict:
    """
    Identify unusual transactions within each spending category.

    Args:
        df:                   Categorized transactions DataFrame.
        threshold_multiplier: Flag transaction if amount > N × category average.
                              Default is 3.0 (3x the average).

    Returns:
        {
          "anomalies": list of anomaly dicts,
          "alerts":    list of human-readable alert strings,
          "count":     int
        }
    """
    if df.empty:
        return {"anomalies": [], "alerts": [], "count": 0}

    anomalies = []
    alerts = []

    # Compute per-category average
    cat_avg = df.groupby("Category")["Amount"].mean()
    cat_std = df.groupby("Category")["Amount"].std().fillna(0)

    for _, row in df.iterrows():
        cat = row["Category"]
        amt = row["Amount"]
        avg = cat_avg.get(cat, 0)

        if avg > 0 and amt > threshold_multiplier * avg:
            multiplier = amt / avg
            anomalies.append({
                "date":        row["Date"],
                "description": row["Description"],
                "category":    cat,
                "amount":      amt,
                "category_avg": avg,
                "multiplier":  multiplier,
            })
            alerts.append(
                f"🚨 **{row['Description']}** — ₹{amt:,.0f} "
                f"({multiplier:.1f}× your avg ₹{avg:,.0f} in {cat})"
            )

    if not anomalies:
        alerts.append("✅ No unusual transactions detected in this period.")

    return {
        "anomalies": anomalies,
        "alerts":    alerts,
        "count":     len(anomalies),
    }
