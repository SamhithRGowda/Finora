"""
rag_engine.py — FAISS-based RAG for Finora Chat
Replaces ChromaDB with FAISS — lighter, in-memory, no persistent DB needed.

Pipeline:
  1. Each transaction → text string "Date | Description | Amount | Category"
  2. Encode with sentence-transformers (all-MiniLM-L6-v2)
  3. Store in FAISS index
  4. On query → encode query → retrieve top-K similar transactions
  5. Pass retrieved transactions as context to LLM

Requires: pip install faiss-cpu sentence-transformers
Fallback: if FAISS unavailable, falls back to existing full-context chat.
"""

import hashlib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Availability check
# ─────────────────────────────────────────────

def _faiss_available() -> bool:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False

# Keep this name so app.py import stays unchanged
_chromadb_available = _faiss_available


# ─────────────────────────────────────────────
#  Global state (in-memory, resets on restart)
# ─────────────────────────────────────────────

_faiss_index    = None
_documents      = []        # list of text strings, one per transaction
_metadatas      = []        # list of dicts with date/desc/amount/category
_model          = None
_index_hash     = None      # tracks which df is currently indexed


# ─────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─────────────────────────────────────────────
#  Index transactions into FAISS
# ─────────────────────────────────────────────

def index_transactions(df: pd.DataFrame) -> bool:
    """
    Build a FAISS index from transactions.
    Only re-indexes if the DataFrame has changed.

    Returns True if indexed successfully, False if FAISS unavailable.
    """
    global _faiss_index, _documents, _metadatas, _index_hash

    if not _faiss_available():
        return False

    import faiss

    # Skip if already indexed with same data
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    if df_hash == _index_hash:
        return True

    # ── Build text documents ──────────────────────────────────────────────────
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        date   = str(row.get("Date", ""))
        desc   = str(row.get("Description", ""))
        amount = float(row.get("Amount", 0))
        cat    = str(row.get("Category", "Other"))

        # Rich text format for better semantic search
        text = f"{date} | {desc} | ₹{amount:,.0f} | {cat}"
        documents.append(text)
        metadatas.append({
            "date":        date,
            "description": desc,
            "amount":      amount,
            "category":    cat,
        })

    # ── Generate embeddings ───────────────────────────────────────────────────
    model      = _get_model()
    embeddings = model.encode(documents, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # ── Build FAISS index ─────────────────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on normalized vectors
    index.add(embeddings)

    _faiss_index = index
    _documents   = documents
    _metadatas   = metadatas
    _index_hash  = df_hash

    return True


# ─────────────────────────────────────────────
#  Retrieve relevant transactions
# ─────────────────────────────────────────────

def retrieve_relevant_context(query: str, n_results: int = 12) -> str:
    """
    Encode the query, search FAISS, return top matching transactions as text.
    Returns empty string if FAISS not available or index empty.
    """
    if not _faiss_available() or _faiss_index is None or not _documents:
        return ""

    try:
        import faiss

        model        = _get_model()
        query_vec    = model.encode([query], show_progress_bar=False)
        query_vec    = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        k       = min(n_results, len(_documents))
        scores, indices = _faiss_index.search(query_vec, k)

        lines = ["Relevant transactions for your question:"]
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            m = _metadatas[idx]
            lines.append(
                f"  • {m['date']} | {m['description']} | "
                f"₹{m['amount']:,.0f} | {m['category']}"
            )

        return "\n".join(lines)

    except Exception as e:
        print(f"[FAISS] Retrieval error: {e}")
        return ""


# ─────────────────────────────────────────────
#  RAG-powered Chat (same signature as before)
# ─────────────────────────────────────────────

def rag_chat(
    user_message: str,
    chat_history: list,
    df: pd.DataFrame,
    monthly_income: float,
    financial_context: str,
) -> tuple:
    """
    Answer a finance question using FAISS RAG.
    Falls back to regular chat if FAISS is unavailable.

    Returns: (reply, updated_chat_history, used_rag: bool)
    """
    from ai.gemini_ai import chat_with_finances, _call_llm_chat

    # Try to index and retrieve
    indexed   = index_transactions(df)
    retrieved = retrieve_relevant_context(user_message) if indexed else ""

    if indexed and retrieved:
        total_spent = df["Amount"].sum()
        savings     = monthly_income - total_spent

        rag_context = (
            f"USER FINANCIAL SUMMARY:\n"
            f"Monthly Income : ₹{monthly_income:,.0f}\n"
            f"Total Spent    : ₹{total_spent:,.0f}\n"
            f"Savings        : ₹{savings:,.0f}\n"
            f"Transactions   : {len(df)}\n\n"
            f"{retrieved}"
        )

        system_prompt = (
            "You are Finora, a smart AI financial advisor for Indian users.\n"
            "Answer the user's question using ONLY the transaction data provided below.\n"
            "Be specific — reference exact amounts, dates, merchant names.\n"
            "Keep answers concise (under 150 words). Use ₹ symbols.\n\n"
            f"{rag_context}\n---"
        )

        # Build OpenAI-style message list
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            text = msg["parts"][0] if msg["parts"] else ""
            if "USER'S FINANCIAL DATA" in text and role == "user":
                text = text.split("User says:")[-1].strip()
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        try:
            reply = _call_llm_chat(messages, max_tokens=400)
            chat_history.append({"role": "user",  "parts": [user_message]})
            chat_history.append({"role": "model", "parts": [reply]})
            return reply, chat_history, True
        except Exception as e:
            print(f"[RAG] LLM error: {e}")

    # Fallback — regular context injection
    reply, updated = chat_with_finances(user_message, chat_history, financial_context)
    return reply, updated, False