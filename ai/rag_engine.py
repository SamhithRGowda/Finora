"""
rag_engine.py — RAG (Retrieval Augmented Generation) for Finora Chat
Uses ChromaDB to store and retrieve transaction context intelligently.

Instead of dumping all transactions into every prompt, RAG:
  1. Stores transactions as searchable embeddings in ChromaDB
  2. When user asks a question, retrieves only the RELEVANT transactions
  3. Sends those relevant transactions + question to the LLM

This gives better answers AND handles large transaction histories efficiently.

Requires: pip install chromadb sentence-transformers
"""

import os
import json
import hashlib
import pandas as pd

# ─────────────────────────────────────────────
#  Lazy imports — only load if chromadb available
# ─────────────────────────────────────────────

def _chromadb_available() -> bool:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False


# Global ChromaDB client and collection
_chroma_client     = None
_collection        = None
_embedding_model   = None
_collection_hash   = None   # tracks which DataFrame is currently loaded


def _get_embedding_model():
    """Load a lightweight sentence transformer for embeddings."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is small (80MB), fast, and accurate enough for finance
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_collection():
    """Get or create the ChromaDB collection."""
    global _chroma_client, _collection
    if _chroma_client is None:
        import chromadb
        # Persist to disk so it survives Streamlit reruns
        _chroma_client = chromadb.PersistentClient(path=".finora_chroma_db")
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name="finora_transactions",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─────────────────────────────────────────────
#  Index Transactions
# ─────────────────────────────────────────────

def index_transactions(df: pd.DataFrame) -> bool:
    """
    Store transactions in ChromaDB as searchable embeddings.
    Only re-indexes if the DataFrame has changed.

    Args:
        df: Categorized transactions DataFrame.

    Returns:
        True if indexed successfully, False if ChromaDB not available.
    """
    global _collection_hash

    if not _chromadb_available():
        return False

    # Check if we already indexed this exact DataFrame
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    if df_hash == _collection_hash:
        return True   # already up to date

    collection = _get_collection()
    model      = _get_embedding_model()

    # Clear existing data
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Build documents — each transaction becomes a searchable text snippet
    documents = []
    metadatas = []
    ids       = []

    for i, row in df.iterrows():
        # Create a rich text description of each transaction
        doc = (
            f"On {row.get('Date', 'unknown date')}, "
            f"₹{row.get('Amount', 0):,.0f} was spent on "
            f"{row.get('Description', 'unknown')} "
            f"in the category {row.get('Category', 'Other')}."
        )
        documents.append(doc)
        metadatas.append({
            "date":        str(row.get("Date", "")),
            "description": str(row.get("Description", "")),
            "category":    str(row.get("Category", "Other")),
            "amount":      float(row.get("Amount", 0)),
        })
        ids.append(f"txn_{i}")

    # Generate embeddings in batch
    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    # Add to ChromaDB in batches of 100
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=  documents[start:end],
            embeddings= embeddings[start:end],
            metadatas=  metadatas[start:end],
            ids=        ids[start:end],
        )

    _collection_hash = df_hash
    return True


# ─────────────────────────────────────────────
#  Retrieve Relevant Context
# ─────────────────────────────────────────────

def retrieve_relevant_context(query: str, n_results: int = 10) -> str:
    """
    Given a user question, retrieve the most relevant transactions
    from ChromaDB and format them as context for the LLM.

    Args:
        query:     The user's question.
        n_results: Number of relevant transactions to retrieve.

    Returns:
        A formatted string of relevant transactions to inject into the prompt.
    """
    if not _chromadb_available():
        return ""

    try:
        collection = _get_collection()
        model      = _get_embedding_model()

        # Check if collection has data
        if collection.count() == 0:
            return ""

        # Embed the query and search
        query_embedding = model.encode([query]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        # Format retrieved transactions
        lines = ["Relevant transactions for your question:"]
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            lines.append(
                f"  • {meta['date']} | {meta['description']} | "
                f"₹{meta['amount']:,.0f} | {meta['category']}"
            )

        return "\n".join(lines)

    except Exception as e:
        return ""   # fail silently — fallback to regular context


# ─────────────────────────────────────────────
#  RAG-powered Chat
# ─────────────────────────────────────────────

def rag_chat(
    user_message: str,
    chat_history: list,
    df: pd.DataFrame,
    monthly_income: float,
    financial_context: str,
) -> tuple:
    """
    Answer a finance question using RAG — retrieves relevant transactions
    then calls the LLM with targeted context instead of all transactions.

    Falls back to regular chat if ChromaDB is not available.

    Args:
        user_message:      The user's question.
        chat_history:      Conversation history.
        df:                Full transactions DataFrame.
        monthly_income:    User's monthly income.
        financial_context: Pre-built summary string (fallback context).

    Returns:
        (reply, updated_chat_history, used_rag: bool)
    """
    from ai.gemini_ai import chat_with_finances, _call_llm_chat

    # Try to index and retrieve
    indexed = index_transactions(df)

    if indexed:
        # RAG path — retrieve relevant context for this specific question
        retrieved = retrieve_relevant_context(user_message, n_results=15)

        total_spent = df["Amount"].sum()
        savings     = monthly_income - total_spent

        rag_context = f"""USER FINANCIAL SUMMARY:
Monthly Income : ₹{monthly_income:,.0f}
Total Spent    : ₹{total_spent:,.0f}
Savings        : ₹{savings:,.0f}
Transactions   : {len(df)}

{retrieved}"""

        system_prompt = (
            "You are Finora, a smart AI financial advisor for Indian users.\n"
            "Answer the user's question using the transaction data provided below.\n"
            "Be specific — reference exact amounts, dates, and merchant names.\n"
            "Keep answers concise (under 150 words). Use ₹ symbols.\n\n"
            f"{rag_context}\n---"
        )

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            text = msg["parts"][0]
            if "USER'S FINANCIAL DATA" in text and role == "user":
                text = text.split("User says:")[-1].strip()
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        try:
            reply = _call_llm_chat(messages, max_tokens=400)
            chat_history.append({"role": "user",  "parts": [user_message]})
            chat_history.append({"role": "model", "parts": [reply]})
            return reply, chat_history, True
        except Exception:
            pass   # fall through to regular chat

    # Fallback — regular chat without RAG
    reply, updated = chat_with_finances(user_message, chat_history, financial_context)
    return reply, updated, False