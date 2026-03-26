"""
rag_engine.py — Hybrid RAG for Finora Chat
Upgrade: Rule-based filtering BEFORE FAISS for accurate retrieval.

Pipeline:
  1. Detect intent from query (category, month, sort)
  2. Filter df BEFORE embedding (rule-based pre-filter)
  3. FAISS cosine search on filtered subset
  4. Return clean context + explainability metadata

Requires: pip install faiss-cpu sentence-transformers
Fallback: full dataset if filter returns empty, then regular chat if FAISS unavailable.
"""

import re
import hashlib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Availability
# ─────────────────────────────────────────────

def _faiss_available() -> bool:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False

# Keep old name so app.py import stays unchanged
_chromadb_available = _faiss_available


# ─────────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────────

_model      = None
_index_hash = None   # hash of last indexed df


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─────────────────────────────────────────────
#  Intent detection
# ─────────────────────────────────────────────

# Synonym maps — expand query keywords to category keywords
CATEGORY_SYNONYMS = {
    "food":           ["food", "dining", "swiggy", "zomato", "restaurant", "bigbasket",
                       "instamart", "dominos", "kfc", "blinkit", "uber eats"],
    "shopping":       ["shopping", "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa"],
    "transport":      ["transport", "uber", "ola", "rapido", "metro", "petrol", "fuel", "cab"],
    "entertainment":  ["entertainment", "netflix", "spotify", "hotstar", "prime", "bookmyshow"],
    "utilities":      ["utilities", "electricity", "airtel", "jio", "broadband", "recharge", "bill"],
    "healthcare":     ["healthcare", "apollo", "medplus", "1mg", "pharmacy", "medicine", "doctor"],
    "subscriptions":  ["netflix", "spotify", "prime", "hotstar", "subscription", "recurring"],
    "credit":         ["credit card", "hdfc", "icici", "sbi", "emi", "loan"],
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


def detect_intent(query: str) -> dict:
    """
    Extract intent signals from the user query.
    Returns dict with: category, month, sort_by, keywords
    """
    q = query.lower().strip()
    intent = {
        "category":  None,   # e.g. "food", "shopping"
        "month":     None,   # integer 1-12
        "sort_by":   None,   # "amount_desc" | "amount_asc"
        "keywords":  [],     # raw keywords for FAISS
    }

    # ── Detect category ───────────────────────────────────────────────────────
    for cat, synonyms in CATEGORY_SYNONYMS.items():
        if any(syn in q for syn in synonyms):
            intent["category"] = cat
            break

    # ── Detect month ──────────────────────────────────────────────────────────
    for month_str, month_num in MONTH_MAP.items():
        if month_str in q:
            intent["month"] = month_num
            break

    # ── Detect sort intent ────────────────────────────────────────────────────
    if any(w in q for w in ["biggest", "largest", "most expensive", "highest", "top"]):
        intent["sort_by"] = "amount_desc"
    elif any(w in q for w in ["smallest", "cheapest", "lowest", "least"]):
        intent["sort_by"] = "amount_asc"

    return intent


# ─────────────────────────────────────────────
#  Pre-filter df based on intent (BEFORE FAISS)
# ─────────────────────────────────────────────

def prefilter_df(df: pd.DataFrame, intent: dict) -> tuple:
    """
    Apply rule-based filters to df BEFORE embedding search.
    Returns (filtered_df, filter_description).
    Falls back to full df if filter returns empty.
    """
    filtered = df.copy()
    desc_parts = []

    # ── Fix date column ───────────────────────────────────────────────────────
    if "Date" in filtered.columns:
        filtered["_date_parsed"] = pd.to_datetime(
            filtered["Date"], dayfirst=True, errors="coerce"
        )

    # ── Month filter ──────────────────────────────────────────────────────────
    if intent["month"] and "_date_parsed" in filtered.columns:
        month_filtered = filtered[
            filtered["_date_parsed"].dt.month == intent["month"]
        ]
        if not month_filtered.empty:
            filtered = month_filtered
            month_name = list(MONTH_MAP.keys())[
                list(MONTH_MAP.values()).index(intent["month"])
            ].capitalize()
            desc_parts.append(f"in {month_name}")

    # ── Category filter ───────────────────────────────────────────────────────
    if intent["category"] and "Category" in filtered.columns:
        synonyms = CATEGORY_SYNONYMS.get(intent["category"], [])
        # Match on Category column OR Description column
        cat_mask  = filtered["Category"].str.lower().str.contains(
            intent["category"], na=False
        )
        desc_mask = filtered["Description"].str.lower().apply(
            lambda x: any(s in x for s in synonyms)
        )
        cat_filtered = filtered[cat_mask | desc_mask]
        if not cat_filtered.empty:
            filtered = cat_filtered
            desc_parts.append(f"in {intent['category'].replace('_', ' ').title()}")

    # ── Sort ──────────────────────────────────────────────────────────────────
    if intent["sort_by"] == "amount_desc":
        filtered = filtered.sort_values("Amount", ascending=False)
        desc_parts.append("sorted by highest amount")
    elif intent["sort_by"] == "amount_asc":
        filtered = filtered.sort_values("Amount", ascending=True)
        desc_parts.append("sorted by lowest amount")

    # ── Fallback to full df if filter returned nothing ─────────────────────────
    used_fallback = False
    if filtered.empty or len(filtered) == 0:
        filtered = df.copy()
        used_fallback = True
        desc_parts = ["all transactions (no matching filter found)"]

    filter_desc = ", ".join(desc_parts) if desc_parts else "all transactions"

    # Drop helper column
    if "_date_parsed" in filtered.columns:
        filtered = filtered.drop(columns=["_date_parsed"])

    return filtered, filter_desc, used_fallback


# ─────────────────────────────────────────────
#  FAISS search on filtered subset
# ─────────────────────────────────────────────

def faiss_search(query: str, df: pd.DataFrame, n_results: int = 10) -> list:
    """
    Build a mini FAISS index from the (already filtered) df,
    search for query, return list of matching metadata dicts.
    """
    if df.empty:
        return []

    import faiss

    # Build documents
    docs = []
    metas = []
    for _, row in df.iterrows():
        text = (
            f"{row.get('Date', '')} | {row.get('Description', '')} | "
            f"₹{float(row.get('Amount', 0)):,.0f} | {row.get('Category', 'Other')}"
        )
        docs.append(text)
        metas.append({
            "date":        str(row.get("Date", "")),
            "description": str(row.get("Description", "")),
            "amount":      float(row.get("Amount", 0)),
            "category":    str(row.get("Category", "Other")),
        })

    model      = _get_model()
    embeddings = model.encode(docs, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    # Build index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Search
    query_vec = model.encode([query], show_progress_bar=False)
    query_vec = np.array(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    k = min(n_results, len(docs))
    _, indices = index.search(query_vec, k)

    # Deduplicate by description+amount
    seen    = set()
    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        m   = metas[idx]
        key = (m["description"], m["amount"])
        if key not in seen:
            seen.add(key)
            results.append(m)

    return results


# ─────────────────────────────────────────────
#  Build context string for LLM
# ─────────────────────────────────────────────

def build_rag_context(
    results: list,
    filter_desc: str,
    monthly_income: float,
    total_transactions: int,
) -> str:
    """
    Format retrieved transactions into a clean context string.
    Includes explainability metadata.
    """
    if not results:
        return ""

    total_amount = sum(r["amount"] for r in results)
    n            = len(results)

    # Summary line (explainability)
    context_lines = [
        f"Retrieved {n} transactions ({filter_desc}).",
        f"Total amount: ₹{total_amount:,.0f}",
        f"User monthly income: ₹{monthly_income:,.0f}",
        "",
        "Transaction details:",
    ]

    for r in results:
        context_lines.append(
            f"  • {r['date']} | {r['description']} | ₹{r['amount']:,.0f} | {r['category']}"
        )

    return "\n".join(context_lines)


# ─────────────────────────────────────────────
#  Public: index_transactions (no-op for FAISS)
# ─────────────────────────────────────────────

def index_transactions(df: pd.DataFrame) -> bool:
    """
    For FAISS we build the index on-demand per query (filtered subset).
    This function just validates availability.
    """
    return _faiss_available()


# ─────────────────────────────────────────────
#  Public: retrieve_relevant_context
# ─────────────────────────────────────────────

def retrieve_relevant_context(query: str, n_results: int = 12) -> str:
    """Legacy interface — used by app.py if called directly."""
    return ""   # handled inside rag_chat now


# ─────────────────────────────────────────────
#  Main: rag_chat
# ─────────────────────────────────────────────

def rag_chat(
    user_message: str,
    chat_history: list,
    df: pd.DataFrame,
    monthly_income: float,
    financial_context: str,
) -> tuple:
    """
    Hybrid RAG chat:
      1. Detect intent → 2. Pre-filter df → 3. FAISS search →
      4. Build clean context → 5. LLM response

    Returns: (reply, updated_chat_history, used_rag: bool)
    """
    from ai.gemini_ai import chat_with_finances, _call_llm_chat

    if not _faiss_available() or df is None or df.empty:
        # Fallback to regular chat
        reply, updated = chat_with_finances(user_message, chat_history, financial_context)
        return reply, updated, False

    try:
        # ── Step 1: Detect intent ─────────────────────────────────────────────
        intent = detect_intent(user_message)

        # ── Step 2: Pre-filter df ─────────────────────────────────────────────
        filtered_df, filter_desc, used_fallback = prefilter_df(df, intent)

        # ── Step 3: FAISS search on filtered subset ───────────────────────────
        results = faiss_search(user_message, filtered_df, n_results=12)

        if not results:
            reply, updated = chat_with_finances(user_message, chat_history, financial_context)
            return reply, updated, False

        # ── Step 4: Build clean context ───────────────────────────────────────
        rag_context = build_rag_context(
            results, filter_desc, monthly_income, len(df)
        )

        # ── Step 5: LLM prompt ────────────────────────────────────────────────
        system_prompt = f"""You are Finora, a smart AI financial advisor for Indian users.
Answer the user's question using ONLY the transaction data below.

RULES:
- Be specific — use exact ₹ amounts and dates from the data
- Always mention: how many transactions, category (if relevant), time period (if relevant)
- Example good answer: "Your total food spending in December is ₹6,620 across 8 transactions."
- Keep answer under 120 words
- Use ₹ symbols
- Do NOT list all transactions unless asked

{rag_context}
---"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            text = msg["parts"][0] if msg["parts"] else ""
            if "USER'S FINANCIAL DATA" in text and role == "user":
                text = text.split("User says:")[-1].strip()
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        reply = _call_llm_chat(messages, max_tokens=300)
        chat_history.append({"role": "user",  "parts": [user_message]})
        chat_history.append({"role": "model", "parts": [reply]})
        return reply, chat_history, True

    except Exception as e:
        print(f"[RAG] Error: {e}")
        reply, updated = chat_with_finances(user_message, chat_history, financial_context)
        return reply, updated, False