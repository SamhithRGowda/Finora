"""
financial_simulator.py — Simulate financial decisions like SIP investments,
loan repayments, and expense reduction scenarios.
"""


def simulate_sip(monthly_amount: float, years: int, annual_return: float = 12.0) -> dict:
    """
    Simulate a SIP (Systematic Investment Plan) investment.

    Args:
        monthly_amount: Amount invested every month (₹).
        years:          Investment duration in years.
        annual_return:  Expected annual return % (default 12% for equity mutual funds).

    Returns:
        Dict with invested amount, estimated value, and wealth gained.
    """
    months = years * 12
    monthly_rate = annual_return / 100 / 12

    # Future Value of SIP formula: FV = P × [((1 + r)^n - 1) / r] × (1 + r)
    if monthly_rate > 0:
        future_value = monthly_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        future_value = monthly_amount * months

    invested = monthly_amount * months
    wealth_gained = future_value - invested

    return {
        "monthly_amount": monthly_amount,
        "years":          years,
        "annual_return":  annual_return,
        "total_invested": invested,
        "future_value":   future_value,
        "wealth_gained":  wealth_gained,
    }


def simulate_expense_cut(
    monthly_cut: float,
    years: int,
    invest_savings: bool = True,
    annual_return: float = 12.0,
) -> dict:
    """
    Simulate what happens if you reduce a monthly expense by a fixed amount.

    Args:
        monthly_cut:     Amount saved per month by cutting expense (₹).
        years:           Duration to simulate.
        invest_savings:  If True, assume saved money is invested in SIP.
        annual_return:   Annual return % if invested.

    Returns:
        Dict with simple savings vs invested value.
    """
    simple_savings = monthly_cut * years * 12

    if invest_savings:
        sip = simulate_sip(monthly_cut, years, annual_return)
        invested_value = sip["future_value"]
    else:
        invested_value = simple_savings

    return {
        "monthly_cut":    monthly_cut,
        "years":          years,
        "simple_savings": simple_savings,
        "invested_value": invested_value,
        "extra_gained":   invested_value - simple_savings,
    }


def simulate_emergency_fund(
    monthly_income: float,
    monthly_savings: float,
    months_target: int = 6,
) -> dict:
    """
    Calculate how long to build an emergency fund.

    Args:
        monthly_income:  Monthly income (₹).
        monthly_savings: Current monthly savings (₹).
        months_target:   Target months of expenses to save (default 6).

    Returns:
        Dict with target amount and months to reach it.
    """
    target = monthly_income * months_target
    months_to_reach = target / monthly_savings if monthly_savings > 0 else float("inf")

    return {
        "target_amount":   target,
        "monthly_savings": monthly_savings,
        "months_to_reach": months_to_reach,
        "years_to_reach":  months_to_reach / 12,
    }


def simulate_financial_scenarios(
    monthly_income: float,
    monthly_expenses: float,
) -> list:
    """
    Run multiple pre-defined what-if scenarios and return results.

    Args:
        monthly_income:   User's monthly income.
        monthly_expenses: User's current monthly expenses.

    Returns:
        List of scenario result dicts.
    """
    monthly_savings = max(monthly_income - monthly_expenses, 0)
    scenarios = []

    # Scenario 1: Start a ₹2000/month SIP
    sip_2k = simulate_sip(2000, 10)
    scenarios.append({
        "title":       "💹 Invest ₹2,000/month in SIP",
        "description": "Start a ₹2,000/month SIP in an equity mutual fund at 12% annual return.",
        "results": [
            f"After 5 years  → ₹{simulate_sip(2000, 5)['future_value']:,.0f}",
            f"After 10 years → ₹{sip_2k['future_value']:,.0f}",
            f"Total invested → ₹{sip_2k['total_invested']:,.0f}",
            f"Wealth gained  → ₹{sip_2k['wealth_gained']:,.0f}",
        ]
    })

    # Scenario 2: Invest current savings
    if monthly_savings > 0:
        sip_savings = simulate_sip(monthly_savings, 10)
        scenarios.append({
            "title":       f"📈 Invest your current savings (₹{monthly_savings:,.0f}/month)",
            "description": f"Put all your monthly surplus into an index fund.",
            "results": [
                f"After 5 years  → ₹{simulate_sip(monthly_savings, 5)['future_value']:,.0f}",
                f"After 10 years → ₹{sip_savings['future_value']:,.0f}",
                f"Wealth gained  → ₹{sip_savings['wealth_gained']:,.0f}",
            ]
        })

    # Scenario 3: Cut food spending by ₹2000
    cut = simulate_expense_cut(2000, 10, invest_savings=True)
    scenarios.append({
        "title":       "🍽️ Cut food spending by ₹2,000/month",
        "description": "Reduce restaurant/delivery orders and invest the savings.",
        "results": [
            f"Simple savings over 10 years → ₹{cut['simple_savings']:,.0f}",
            f"If invested in SIP           → ₹{cut['invested_value']:,.0f}",
            f"Extra wealth from investing  → ₹{cut['extra_gained']:,.0f}",
        ]
    })

    # Scenario 4: Emergency fund
    ef = simulate_emergency_fund(monthly_income, max(monthly_savings, 1000))
    scenarios.append({
        "title":       "🏦 Build Emergency Fund (6 months)",
        "description": f"Target: ₹{ef['target_amount']:,.0f} (6× monthly income)",
        "results": [
            f"Target amount    → ₹{ef['target_amount']:,.0f}",
            f"Saving per month → ₹{ef['monthly_savings']:,.0f}",
            f"Time to reach it → {ef['months_to_reach']:.0f} months ({ef['years_to_reach']:.1f} years)",
        ]
    })

    return scenarios
