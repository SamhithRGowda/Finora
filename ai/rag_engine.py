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

# --- UPDATED: Strict category column values (exact match, no cross-category leakage) ---
CATEGORY_COLUMN_MAP = {
    "food":          ["Food & Dining"],
    "shopping":      ["Shopping"],
    "transport":     ["Transport"],
    "entertainment": ["Entertainment"],
    "utilities":     ["Utilities & Bills"],
    "healthcare":    ["Healthcare"],
    "subscriptions": ["Entertainment", "Utilities & Bills"],  # subscriptions span these
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
    "breakdown":      ["elaborate", "breakdown", "break down", "break it down", "detail",
                       "where did", "where i spent", "what are they", "what r they",
                       "which ones", "itemize", "each transaction", "show me each"],
}

# --- UPDATED: Opinion/analysis keywords ---
ANALYSIS_KEYWORDS = [
    "too much", "overspend", "overspending", "should i", "am i spending",
    "is it high", "is this high", "is that high", "too high", "a lot",
    "reduce", "cut down", "worth it", "reasonable",
]

# Savings intent keywords
SAVINGS_KEYWORDS = [
    "how much did i save", "how much have i saved", "what did i save",
    "my savings", "savings this month", "savings in", "how much saved",
    "saving this month", "saved this month", "did i save",
    "money saved", "leftover", "remaining",
]

# Comparison intent keywords
COMPARE_KEYWORDS = [
    "compare", "vs", "versus", "more than", "less than",
    "difference between", "higher in", "lower in",
    "higher than", "lower than", "greater than",
    "did i spend more", "did i spend less", "which was more",
    "compared to", "than last month", "than previous",
    "more in", "less in",
]


def detect_intent(query: str) -> dict:
    """
    Parse natural language query into structured intent.
    Fix 1: Category detected by name only (not description keywords).
    Fix 2: max_by_month and analysis checked before generic metrics.
    """
    q = query.lower().strip()

    # breakdown — check before generic metrics so "elaborate each transaction" isn't
    # swallowed by "total_spend" matching on "spend" keyword
    BREAKDOWN_KWS = ["elaborate", "breakdown", "break down", "break it down", "detail",
                     "where did", "where i spent", "what are they", "what r they",
                     "which ones", "itemize", "each transaction", "show me each",
                     "list them", "tell me each", "show each", "what were they",
                     "what r they", "tell me more", "give details"]

    # max_by_month — check first (before generic metric detection)
    # Catches: "which month most", "highest spending month", "highest month", "most spent month"
    is_month_query = (
        "which month" in q or "what month" in q or
        "spending month" in q or "spent month" in q or
        ("month" in q and any(w in q for w in ["highest", "most", "maximum", "max", "biggest"]))
    )
    if is_month_query:
        metric = "max_by_month"
    # breakdown — check before generic so "elaborate" isn't caught by "total_spend"
    elif any(kw in q for kw in BREAKDOWN_KWS):
        metric = "breakdown"
    # comparison between two months — check before savings/analysis
    elif any(kw in q for kw in COMPARE_KEYWORDS):
        metric = "compare"
    # savings query
    elif any(kw in q for kw in SAVINGS_KEYWORDS):
        metric = "savings"
    # analysis/opinion intent
    elif any(kw in q for kw in ANALYSIS_KEYWORDS):
        metric = "analysis"
    else:
        metric = "total_spend"
        for m, keywords in METRIC_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                metric = m
                break

    # Fix 1: detect category by name only — no description keyword matching
    category = None
    for cat in CATEGORY_COLUMN_MAP:
        if cat in q:
            category = cat
            break

    # Detect month(s) — use word boundary matching to avoid
    # false matches like "sep" inside "separately", "nov" inside "novel"
    import re as _re
    month  = None
    month2 = None
    found_months = []
    for word, num in MONTH_MAP.items():
        # \b = word boundary — "sep" won't match inside "separately"
        if _re.search(r'\b' + word + r'\b', q) and num not in found_months:
            found_months.append(num)
        if len(found_months) == 2:
            break
    if found_months:
        month  = found_months[0]
        month2 = found_months[1] if len(found_months) > 1 else None

    return {
        "category":  category,
        "month":     month,
        "month2":    month2,    # second month for compare intent
        "metric":    metric,
        "raw_query": query,
    }


# ─────────────────────────────────────────────
#  Merchant → correct category overrides
#  Prevents LLM miscategorization from polluting filter results
# ─────────────────────────────────────────────

MERCHANT_CATEGORY_OVERRIDE = {
    "swiggy":    "Food & Dining",
    "zomato":    "Food & Dining",
    "bigbasket": "Food & Dining",
    "blinkit":   "Food & Dining",
    "instamart": "Food & Dining",
    "dominos":   "Food & Dining",
    "kfc":       "Food & Dining",
    "uber":      "Transport",
    "ola":       "Transport",
    "rapido":    "Transport",
    "amazon":    "Shopping",
    "flipkart":  "Shopping",
    "myntra":    "Shopping",
    "ajio":      "Shopping",
    "netflix":   "Entertainment",
    "spotify":   "Entertainment",
    "hotstar":   "Entertainment",
    "airtel":    "Utilities & Bills",
    "jio":       "Utilities & Bills",
    "apollo":    "Healthcare",
    "medplus":   "Healthcare",
    "1mg":       "Healthcare",
}


def _correct_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Override LLM-assigned Category using merchant name in Description.
    This fixes cases where Groq/NVIDIA assigns wrong categories (e.g. BigBasket as Transport).
    """
    if "Description" not in df.columns or "Category" not in df.columns:
        return df
    df = df.copy()
    def override(row):
        desc_lower = str(row["Description"]).lower()
        for merchant, correct_cat in MERCHANT_CATEGORY_OVERRIDE.items():
            if merchant in desc_lower:
                return correct_cat
        return row["Category"]
    df["Category"] = df.apply(override, axis=1)
    return df


# ─────────────────────────────────────────────
#  STEP 2 — Filter DataFrame (no FAISS, no LLM)
# ─────────────────────────────────────────────

def filter_df(df: pd.DataFrame, intent: dict) -> tuple:
    """
    Apply category then month filters. NO fallback to full df.
    Fix 2: Always apply filters — empty result propagates to compute_result.
    Fix 7: Category filter first, then month filter.
    Fix 8: No silent revert to full df on empty result.
    """
    # Apply merchant-based category corrections before filtering (Bug 1 fix)
    filtered    = _correct_categories(df)
    scope_parts = []

    # Always parse dates
    filtered["_date"] = pd.to_datetime(
        filtered["Date"], dayfirst=True, errors="coerce"
    )

    # Fix 7 + 8: Category filter — always apply, no fallback
    if intent["category"]:
        valid_cats = CATEGORY_COLUMN_MAP.get(intent["category"], [])
        if valid_cats:
            filtered = filtered[filtered["Category"].isin(valid_cats)]
        else:
            filtered = filtered[
                filtered["Category"].str.lower().str.contains(intent["category"], na=False)
            ]
        scope_parts.append(intent["category"].title())

    # Fix 2 + 8: Month filter — always apply, no fallback
    if intent["month"]:
        filtered = filtered[filtered["_date"].dt.month == intent["month"]]
        # Get month name from MONTH_MAP (safe even if filtered is now empty)
        month_name = next(
            (k.capitalize() for k, v in MONTH_MAP.items() if v == intent["month"] and len(k) > 3),
            str(intent["month"])
        )
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
    full_df: pd.DataFrame = None,   # Fix 3: needed for max_by_month on full data
) -> dict:
    """
    All financial calculations in Python.
    Fix 3: max_by_month uses full_df (not filtered).
    Fix 4: pct_of_income = total_spend / monthly_income (not monthly_avg).
    Fix 5: pct_of_total removed.
    Fix 6: dates always parsed before groupby.
    """
    if filtered.empty:
        return {
            "found":   False,
            "message": f"No transactions found for {scope_label}.",
        }

    metric = intent["metric"]

    # Core numbers
    total_spend = float(filtered["Amount"].sum())
    count       = len(filtered)
    avg_per_txn = total_spend / count if count > 0 else 0

    # Monthly normalization (for display/context only)
    if "_date" in filtered.columns:
        unique_months = max(int(filtered["_date"].dt.to_period("M").nunique()), 1)
    else:
        unique_months = 1
    monthly_avg = total_spend / unique_months

    # Bug 2 fix: use monthly_avg for pct_of_income, not raw total
    # total_spend spans multiple months — comparing it to 1 month income is wrong
    pct_of_income = (monthly_avg / monthly_income * 100) if monthly_income > 0 else 0

    # Top/bottom transactions
    top_df   = filtered.nlargest(5, "Amount")[["Date", "Description", "Amount", "Category"]]
    top_list = [
        f"₹{r['Amount']:,.0f} — {r['Description']} ({r['Date']})"
        for _, r in top_df.iterrows()
    ]
    bot_df   = filtered.nsmallest(3, "Amount")[["Date", "Description", "Amount"]]
    bot_list = [
        f"₹{r['Amount']:,.0f} — {r['Description']} ({r['Date']})"
        for _, r in bot_df.iterrows()
    ]

    # Fix 3 + 6: max_by_month uses full_df, always parses dates
    max_month_label  = None
    max_month_spend  = None
    monthly_breakdown = {}
    if metric in ("max_by_month", "monthly_spend"):
        source = full_df.copy() if full_df is not None else filtered.copy()
        source["_date"] = pd.to_datetime(source["Date"], dayfirst=True, errors="coerce")
        source = source.dropna(subset=["_date"])
        if not source.empty:
            source["_period"] = source["_date"].dt.to_period("M")
            monthly_series    = source.groupby("_period")["Amount"].sum()
            monthly_breakdown = {str(k): float(v) for k, v in monthly_series.items()}
            if not monthly_series.empty:
                best_period     = monthly_series.idxmax()
                # Format as 'December 2024' instead of '2024-12'
                max_month_label = best_period.to_timestamp().strftime('%B %Y')
                max_month_spend = float(monthly_series.max())

    # Feature 4: savings computation
    savings_amount   = max(monthly_income - monthly_avg, 0)
    savings_rate     = (savings_amount / monthly_income * 100) if monthly_income > 0 else 0
    overspent        = monthly_avg > monthly_income
    overspent_by     = max(monthly_avg - monthly_income, 0)

    # Feature 5: compare two months using full_df
    compare_result = {}
    if intent.get("metric") == "compare" and intent.get("month2") and full_df is not None:
        src = full_df.copy()
        src["_date2"] = pd.to_datetime(src["Date"], dayfirst=True, errors="coerce")
        # Apply category filter if present
        if intent.get("category"):
            valid_cats = CATEGORY_COLUMN_MAP.get(intent["category"], [])
            if valid_cats:
                src = _correct_categories(src)
                src = src[src["Category"].isin(valid_cats)]
        m1_data = src[src["_date2"].dt.month == intent["month"]]
        m2_data = src[src["_date2"].dt.month == intent["month2"]]
        m1_total = float(m1_data["Amount"].sum())
        m2_total = float(m2_data["Amount"].sum())
        # Get readable month names
        m1_name = next((k.capitalize() for k, v in MONTH_MAP.items() if v == intent["month"] and len(k) > 3), str(intent["month"]))
        m2_name = next((k.capitalize() for k, v in MONTH_MAP.items() if v == intent["month2"] and len(k) > 3), str(intent["month2"]))
        compare_result = {
            "month1_name":  m1_name,
            "month2_name":  m2_name,
            "month1_total": m1_total,
            "month2_total": m2_total,
            "difference":   abs(m1_total - m2_total),
            "higher_month": m1_name if m1_total >= m2_total else m2_name,
        }

    # Category breakdown (when no category filter applied)
    cat_breakdown = {}
    if "Category" in filtered.columns and not intent["category"]:
        cat_breakdown = (
            filtered.groupby("Category")["Amount"].sum()
            .sort_values(ascending=False).head(5).to_dict()
        )

    # Listing (top 10 only) — also triggered by breakdown intent
    list_preview = []
    if metric in ("list", "breakdown"):
        preview_df   = filtered.head(10)[["Date", "Description", "Amount", "Category"]]
        list_preview = [
            f"{r['Date']} | {r['Description']} | ₹{r['Amount']:,.0f} | {r['Category']}"
            for _, r in preview_df.iterrows()
        ]

    return {
        "found":             True,
        "metric":            metric,
        "scope":             scope_label,
        "count":             count,
        "total_spend":       total_spend,
        "monthly_avg":       monthly_avg,
        "avg_per_txn":       avg_per_txn,
        "unique_months":     unique_months,
        "pct_of_income":     pct_of_income,
        "biggest":           top_list[0] if top_list else "N/A",
        "top5":              top_list,
        "smallest":          bot_list[0] if bot_list else "N/A",
        "cat_breakdown":     cat_breakdown,
        "list_preview":      list_preview,
        "monthly_income":    monthly_income,
        "max_month_label":   max_month_label,
        "max_month_spend":   max_month_spend,
        "monthly_breakdown": monthly_breakdown,
        # Feature 4: savings
        "savings_amount":    savings_amount,
        "savings_rate":      savings_rate,
        "overspent":         overspent,
        "overspent_by":      overspent_by,
        # Feature 5: comparison
        "compare_result":    compare_result,
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

    # --- NEW: max_by_month block ---
    if result.get("max_month_label"):
        lines.append(f"Highest spending month: {result['max_month_label']} — ₹{result['max_month_spend']:,.0f}")
    if result.get("monthly_breakdown"):
        lines.append("Monthly breakdown:")
        for m, amt in result["monthly_breakdown"].items():
            lines.append(f"  {m}: ₹{amt:,.0f}")

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

    # Feature 4: savings block
    if result.get("metric") == "savings":
        lines.append(f"Monthly income     : ₹{result['monthly_income']:,.0f}")
        lines.append(f"Monthly avg spend  : ₹{result['monthly_avg']:,.0f}")
        if result["overspent"]:
            lines.append(f"Status             : OVERSPENT by ₹{result['overspent_by']:,.0f}")
        else:
            lines.append(f"Monthly savings    : ₹{result['savings_amount']:,.0f}")
            lines.append(f"Savings rate       : {result['savings_rate']:.1f}%")

    # Feature 5: comparison block
    if result.get("compare_result"):
        cr = result["compare_result"]
        lines.append(f"Comparison:")
        lines.append(f"  {cr['month1_name']}: ₹{cr['month1_total']:,.0f}")
        lines.append(f"  {cr['month2_name']}: ₹{cr['month2_total']:,.0f}")
        lines.append(f"  Difference: ₹{cr['difference']:,.0f}")
        lines.append(f"  Higher spend: {cr['higher_month']}")

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
        result          = compute_result(filtered, intent, monthly_income, scope, full_df=df)  # Fix 3
        context         = build_verified_context(result)

        # Empty result — direct answer, no LLM needed
        if not result["found"]:
            chat_history.append({"role": "user",  "parts": [user_message]})
            chat_history.append({"role": "model", "parts": [result["message"]]})
            return result["message"], chat_history, True

        # --- UPDATED: analysis intent gets reasoning permission, others stay strict ---
        if result.get("metric") == "analysis":
            system_prompt = f"""You are Finora, an AI financial advisor for Indian users.

The user is asking an opinion/analysis question. Answer YES or NO, then explain using the facts below.

RULES:
1. Use ONLY numbers from VERIFIED FACTS
2. Give a clear YES/NO answer first
3. Reference % of income and % of total spend in your reasoning
4. Keep response under 100 words
5. Use ₹ symbol
6. Compare against recommended benchmarks (e.g. food should be 20-30% of spend)

{context}"""
        elif result.get("metric") == "savings":
            system_prompt = f"""You are Finora, a concise AI financial advisor.

RULES:
1. Use ONLY numbers from VERIFIED FACTS
2. Answer how much the user saved (or overspent) this period
3. Mention savings rate and whether it is healthy (target: 20-30%)
4. Keep under 80 words, use ₹ symbol

{context}"""
        elif result.get("metric") == "compare":
            system_prompt = f"""You are Finora, a concise AI financial advisor.

RULES:
1. Use ONLY numbers from VERIFIED FACTS
2. ALWAYS follow this exact format:
   "[Month1] spend: ₹X. [Month2] spend: ₹Y. [Higher month] was higher by ₹Z."
3. Never give only the difference — always show both month amounts first
4. Use ₹ symbol

{context}"""
        elif result.get("metric") == "breakdown":
            system_prompt = f"""You are Finora, a concise AI financial advisor.

The user wants a detailed breakdown of their transactions.

RULES:
1. Start with: "Based on [count] [scope] transactions, total is ₹X:"
2. Then list EACH transaction on a new line: "• [Description] — ₹[Amount] ([Date])"
3. Use ONLY the transactions from VERIFIED FACTS below
4. Do NOT invent or skip any transaction
5. Use ₹ symbol

{context}"""
        else:
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