"""
rag_engine.py — Deterministic, hallucination-free chat for Finora.

Architecture (strict):
  Step 1: detect_intent(query)     → category, month, metric
  Step 2: filter_df(df, intent)    → filtered DataFrame
  Step 3: compute_result(df)       → Python-computed facts dict
  Step 4: format_response(facts)   → LLM narrates ONLY given numbers

LLM rule: receives only verified Python-computed numbers.
          CANNOT calculate, estimate, or invent anything.
"""

import pandas as pd


# ─────────────────────────────────────────────
#  App.py compatibility stubs
# ─────────────────────────────────────────────
def _faiss_available() -> bool:         return True
_chromadb_available = _faiss_available
def index_transactions(df) -> bool:     return True
def retrieve_relevant_context(*a):      return ""


# ─────────────────────────────────────────────
#  STEP 1 — Intent detection
# ─────────────────────────────────────────────

CATEGORY_MAP = {
    "food":          ["food", "dining", "swiggy", "zomato", "restaurant",
                      "bigbasket", "instamart", "dominos", "kfc", "blinkit", "uber eats"],
    "shopping":      ["shopping", "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa"],
    "transport":     ["transport", "uber", "ola", "rapido", "metro", "petrol", "fuel", "cab", "irctc"],
    "entertainment": ["entertainment", "netflix", "spotify", "hotstar", "prime video",
                      "bookmyshow", "pvr", "gaming"],
    "utilities":     ["utilities", "electricity", "airtel", "jio", "vodafone",
                      "broadband", "recharge", "internet", "bill"],
    "healthcare":    ["healthcare", "apollo", "medplus", "1mg", "pharmeasy",
                      "pharmacy", "medicine", "hospital", "doctor"],
    "subscriptions": ["subscription", "netflix", "spotify", "hotstar", "prime",
                      "recurring", "monthly plan"],
}

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

METRIC_KEYWORDS = {
    "total_spend":    ["total", "how much", "spend", "spent", "cost", "amount"],
    "monthly_spend":  ["monthly", "per month", "average", "avg", "month"],
    "biggest":        ["biggest", "largest", "most expensive", "highest", "top", "maximum", "max"],
    "smallest":       ["smallest", "cheapest", "lowest", "minimum", "min", "least"],
    "count":          ["how many", "count", "number of", "times", "frequency"],
    "list":           ["list", "show", "all transactions", "what are"],
}


def detect_intent(query: str) -> dict:
    """
    Parse natural language query into structured intent.
    Returns: {category, month, metric, raw_query}
    """
    q = query.lower().strip()

    # Detect category
    category = None
    for cat, keywords in CATEGORY_MAP.items():
        if any(kw in q for kw in keywords):
            category = cat
            break

    # Detect month
    month = None
    for word, num in MONTH_MAP.items():
        if word in q:
            month = num
            break

    # Detect metric (default: total_spend)
    metric = "total_spend"
    for m, keywords in METRIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            metric = m
            break

    return {
        "category":  category,
        "month":     month,
        "metric":    metric,
        "raw_query": query,
    }


# ─────────────────────────────────────────────
#  STEP 2 — Filter DataFrame (no FAISS, no LLM)
# ─────────────────────────────────────────────

def filter_df(df: pd.DataFrame, intent: dict) -> tuple:
    """
    Apply category and month filters to the full DataFrame.
    Returns (filtered_df, scope_label).
    Falls back to full df if filter returns empty.
    """
    filtered   = df.copy()
    scope_parts = []

    # Parse dates once
    filtered["_date"] = pd.to_datetime(
        filtered["Date"], dayfirst=True, errors="coerce"
    )

    # Category filter
    if intent["category"]:
        keywords = CATEGORY_MAP.get(intent["category"], [])
        cat_mask = filtered["Category"].str.lower().str.contains(
            intent["category"], na=False
        )
        desc_mask = filtered["Description"].str.lower().apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        result = filtered[cat_mask | desc_mask]
        if not result.empty:
            filtered = result
            scope_parts.append(intent["category"].title())

    # Month filter
    if intent["month"]:
        result = filtered[filtered["_date"].dt.month == intent["month"]]
        if not result.empty:
            filtered = result
            month_name = filtered["_date"].dt.month_name().iloc[0]
            scope_parts.append(month_name)

    scope_label = " in ".join(scope_parts) if scope_parts else "All transactions"
    return filtered, scope_label


# ─────────────────────────────────────────────
#  STEP 3 — Compute result in Python (MANDATORY)
# ─────────────────────────────────────────────

def compute_result(
    filtered: pd.DataFrame,
    intent: dict,
    monthly_income: float,
    scope_label: str,
) -> dict:
    """
    All financial calculations happen HERE in Python.
    LLM receives only the output of this function.
    """
    if filtered.empty:
        return {
            "found":   False,
            "message": f"No transactions found for: {scope_label}.",
        }

    metric = intent["metric"]

    # Core numbers (always computed)
    total_spend   = float(filtered["Amount"].sum())
    count         = len(filtered)
    avg_per_txn   = total_spend / count if count > 0 else 0

    # Monthly normalization
    unique_months = max(int(
        filtered["_date"].dt.to_period("M").nunique()
    ), 1) if "_date" in filtered.columns else 1
    monthly_avg   = total_spend / unique_months

    pct_of_income = (
        (monthly_avg / monthly_income * 100) if monthly_income > 0 else 0
    )

    # Top transactions
    top_df   = filtered.nlargest(5, "Amount")[["Date", "Description", "Amount", "Category"]]
    top_list = [
        f"₹{r['Amount']:,.0f} — {r['Description']} ({r['Date']})"
        for _, r in top_df.iterrows()
    ]

    # Bottom transactions
    bot_df   = filtered.nsmallest(3, "Amount")[["Date", "Description", "Amount"]]
    bot_list = [
        f"₹{r['Amount']:,.0f} — {r['Description']} ({r['Date']})"
        for _, r in bot_df.iterrows()
    ]

    # Category breakdown (when not filtered to single category)
    cat_breakdown = {}
    if "Category" in filtered.columns and not intent["category"]:
        cat_breakdown = (
            filtered.groupby("Category")["Amount"].sum()
            .sort_values(ascending=False).head(5).to_dict()
        )

    # Listing (top 10 only)
    list_preview = []
    if metric == "list":
        preview_df = filtered.head(10)[["Date", "Description", "Amount", "Category"]]
        list_preview = [
            f"{r['Date']} | {r['Description']} | ₹{r['Amount']:,.0f} | {r['Category']}"
            for _, r in preview_df.iterrows()
        ]

    return {
        "found":          True,
        "metric":         metric,
        "scope":          scope_label,
        "count":          count,
        "total_spend":    total_spend,
        "monthly_avg":    monthly_avg,
        "avg_per_txn":    avg_per_txn,
        "unique_months":  unique_months,
        "pct_of_income":  pct_of_income,
        "biggest":        top_list[0] if top_list else "N/A",
        "top5":           top_list,
        "smallest":       bot_list[0] if bot_list else "N/A",
        "cat_breakdown":  cat_breakdown,
        "list_preview":   list_preview,
        "monthly_income": monthly_income,
    }


# ─────────────────────────────────────────────
#  STEP 4 — Build verified context for LLM
# ─────────────────────────────────────────────

def build_verified_context(result: dict) -> str:
    """
    Serialize computed result into a structured string.
    LLM reads only this — cannot access raw data.
    """
    if not result["found"]:
        return result["message"]

    lines = [
        "=== PYTHON-COMPUTED FACTS — USE ONLY THESE NUMBERS ===",
        f"Scope              : {result['scope']}",
        f"Transactions       : {result['count']}",
        f"Total spend        : ₹{result['total_spend']:,.0f}",
        f"Monthly average    : ₹{result['monthly_avg']:,.0f}  (over {result['unique_months']} month(s))",
        f"Avg per transaction: ₹{result['avg_per_txn']:,.0f}",
        f"% of monthly income: {result['pct_of_income']:.1f}%",
        f"Biggest transaction: {result['biggest']}",
        f"Smallest transaction: {result['smallest']}",
    ]

    if result["top5"]:
        lines.append("Top 5 by amount:")
        lines.extend(f"  {t}" for t in result["top5"])

    if result["cat_breakdown"]:
        lines.append("Breakdown by category:")
        for cat, amt in result["cat_breakdown"].items():
            lines.append(f"  {cat}: ₹{amt:,.0f}")

    if result["list_preview"]:
        lines.append(f"Transactions (showing top 10 of {result['count']}):")
        lines.extend(f"  {t}" for t in result["list_preview"])

    lines.append("=== END — DO NOT INVENT ANY OTHER NUMBERS ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  Main: rag_chat
# ─────────────────────────────────────────────

def rag_chat(
    user_message: str,
    chat_history: list,
    df,
    monthly_income: float,
    financial_context: str,
) -> tuple:
    """
    Deterministic, hallucination-free chat.

    Pipeline:
      detect_intent → filter_df → compute_result → LLM formats only
    """
    from ai.gemini_ai import chat_with_finances, _call_llm_chat

    if df is None or df.empty:
        reply, updated = chat_with_finances(
            user_message, chat_history, financial_context
        )
        return reply, updated, False

    try:
        # Steps 1-3: fully in Python
        intent          = detect_intent(user_message)
        filtered, scope = filter_df(df, intent)
        result          = compute_result(filtered, intent, monthly_income, scope)
        context         = build_verified_context(result)

        # Empty result — direct answer, no LLM needed
        if not result["found"]:
            chat_history.append({"role": "user",  "parts": [user_message]})
            chat_history.append({"role": "model", "parts": [result["message"]]})
            return result["message"], chat_history, True

        # Step 4: LLM formats only — strict prompt
        system_prompt = f"""You are Finora, a concise AI financial advisor.

HARD RULES — violations are not allowed:
1. Use ONLY numbers from the VERIFIED FACTS section
2. NEVER calculate, estimate, or change any number
3. ALWAYS start with: "Based on [count] [scope] transactions..."
4. Keep response under 80 words
5. Use ₹ symbol

{context}

Narrate the answer using only the facts above."""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            text = msg["parts"][0] if msg["parts"] else ""
            if "USER'S FINANCIAL DATA" in text and role == "user":
                text = text.split("User says:")[-1].strip()
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        reply = _call_llm_chat(messages, max_tokens=200)

        chat_history.append({"role": "user",  "parts": [user_message]})
        chat_history.append({"role": "model", "parts": [reply]})
        return reply, chat_history, True

    except Exception as e:
        print(f"[RAG] Error: {e}")
        reply, updated = chat_with_finances(
            user_message, chat_history, financial_context
        )
        return reply, updated, False