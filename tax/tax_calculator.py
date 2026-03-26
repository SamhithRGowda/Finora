"""
tax_calculator.py — Week 3: Tax Optimization Module
Compares Old vs New tax regime for Indian taxpayers (FY 2024-25).
Pure Python logic — no LLM needed.
"""


# ─────────────────────────────────────────────
#  Tax Slabs (FY 2024-25 / AY 2025-26)
# ─────────────────────────────────────────────

OLD_REGIME_SLABS = [
    (250_000, 0.00),   # Up to ₹2.5L → 0%
    (500_000, 0.05),   # ₹2.5L–₹5L → 5%
    (1_000_000, 0.20), # ₹5L–₹10L → 20%
    (float("inf"), 0.30),  # Above ₹10L → 30%
]

NEW_REGIME_SLABS = [
    (300_000, 0.00),   # Up to ₹3L → 0%
    (600_000, 0.05),   # ₹3L–₹6L → 5%
    (900_000, 0.10),   # ₹6L–₹9L → 10%
    (1_200_000, 0.15), # ₹9L–₹12L → 15%
    (1_500_000, 0.20), # ₹12L–₹15L → 20%
    (float("inf"), 0.30),  # Above ₹15L → 30%
]

# Standard deduction available in both regimes
STANDARD_DEDUCTION_OLD = 50_000
STANDARD_DEDUCTION_NEW = 75_000  # Increased in Budget 2024

# Rebate u/s 87A
REBATE_87A_OLD_LIMIT = 500_000   # Tax-free if taxable income ≤ ₹5L (old)
REBATE_87A_NEW_LIMIT = 700_000   # Tax-free if taxable income ≤ ₹7L (new)
REBATE_87A_OLD_MAX = 12_500
REBATE_87A_NEW_MAX = 25_000

# Surcharge and Cess
CESS_RATE = 0.04  # Health & Education Cess


# ─────────────────────────────────────────────
#  Public Function
# ─────────────────────────────────────────────

def compare_tax_regimes(
    income: float,
    hra: float = 0,
    deductions_80c: float = 0,
    deductions_80d: float = 0,
) -> dict:
    """
    Compare Old vs New tax regime and recommend the better one.

    Args:
        income:           Gross annual income (₹).
        hra:              HRA exemption claimed (₹) — only in old regime.
        deductions_80c:   Section 80C deductions (max ₹1.5L) — only in old regime.
        deductions_80d:   Section 80D deductions (medical insurance) — only in old regime.

    Returns:
        A dictionary with full breakdown and recommendation.
    """
    # ── Old Regime ────────────────────────────────────────────────────────────
    hra_capped = min(hra, income * 0.40)  # HRA exemption capped at 40% of income
    deductions_80c_capped = min(deductions_80c, 150_000)
    deductions_80d_capped = min(deductions_80d, 25_000)

    old_taxable = (
        income
        - STANDARD_DEDUCTION_OLD
        - hra_capped
        - deductions_80c_capped
        - deductions_80d_capped
    )
    old_taxable = max(old_taxable, 0)
    old_tax_before_cess = _compute_tax(old_taxable, OLD_REGIME_SLABS)
    old_tax_before_cess = _apply_rebate_87a(old_taxable, old_tax_before_cess, REBATE_87A_OLD_LIMIT, REBATE_87A_OLD_MAX)
    old_cess = old_tax_before_cess * CESS_RATE
    old_tax_total = old_tax_before_cess + old_cess

    # ── New Regime ────────────────────────────────────────────────────────────
    new_taxable = max(income - STANDARD_DEDUCTION_NEW, 0)
    new_tax_before_cess = _compute_tax(new_taxable, NEW_REGIME_SLABS)
    new_tax_before_cess = _apply_rebate_87a(new_taxable, new_tax_before_cess, REBATE_87A_NEW_LIMIT, REBATE_87A_NEW_MAX)
    new_cess = new_tax_before_cess * CESS_RATE
    new_tax_total = new_tax_before_cess + new_cess

    # ── Recommendation ───────────────────────────────────────────────────────
    savings = abs(old_tax_total - new_tax_total)
    if new_tax_total < old_tax_total:
        recommended = "New Regime"
        recommendation_reason = (
            f"The New Regime saves you ₹{savings:,.0f} in taxes. "
            "It works best when you have fewer deductions or claim less HRA."
        )
    elif old_tax_total < new_tax_total:
        recommended = "Old Regime"
        recommendation_reason = (
            f"The Old Regime saves you ₹{savings:,.0f} in taxes. "
            "Your deductions (HRA, 80C, 80D) significantly reduce your taxable income."
        )
    else:
        recommended = "Either (Equal)"
        recommendation_reason = "Both regimes result in the same tax liability."

    return {
        "old_regime": {
            "gross_income": income,
            "standard_deduction": STANDARD_DEDUCTION_OLD,
            "hra_exemption": hra_capped,
            "deductions_80c": deductions_80c_capped,
            "deductions_80d": deductions_80d_capped,
            "taxable_income": old_taxable,
            "tax_before_cess": old_tax_before_cess,
            "cess": old_cess,
            "total_tax": old_tax_total,
        },
        "new_regime": {
            "gross_income": income,
            "standard_deduction": STANDARD_DEDUCTION_NEW,
            "hra_exemption": 0,          # Not applicable
            "deductions_80c": 0,          # Not applicable
            "deductions_80d": 0,          # Not applicable
            "taxable_income": new_taxable,
            "tax_before_cess": new_tax_before_cess,
            "cess": new_cess,
            "total_tax": new_tax_total,
        },
        "recommended": recommended,
        "savings": savings,
        "reason": recommendation_reason,
    }


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _compute_tax(taxable_income: float, slabs: list) -> float:
    """Compute tax using progressive slab rates."""
    tax = 0.0
    prev_limit = 0
    for limit, rate in slabs:
        if taxable_income <= prev_limit:
            break
        slab_income = min(taxable_income, limit) - prev_limit
        tax += slab_income * rate
        prev_limit = limit
    return tax


def _apply_rebate_87a(taxable_income: float, tax: float, limit: float, max_rebate: float) -> float:
    """Apply Section 87A rebate if taxable income is within the limit."""
    if taxable_income <= limit:
        rebate = min(tax, max_rebate)
        return max(tax - rebate, 0)
    return tax
