"""
gemini_ai.py — Gen AI Layer for Finora
Supports 3 providers — auto-detected from your API key:

  nvapi-...   → NVIDIA NIM (free, no daily limit) ✅ RECOMMENDED
  gsk_...     → Groq (free, fast)
  AIza...     → Google Gemini

All 3 features:
  1. LLM-based transaction categorization
  2. AI-generated personalized spending insights
  3. Chat with your finances
"""

import json
import re
import pandas as pd


# ─────────────────────────────────────────────
#  Provider state
# ─────────────────────────────────────────────

_provider = None   # "nvidia" | "groq" | "gemini"
_api_key  = None


CATEGORIES = [
    "Food & Dining",
    "Shopping",
    "Transport",
    "Entertainment",
    "Utilities & Bills",
    "Healthcare",
    "Credit Card & Banking",
    "Investment & Savings",
    "Other",
]


# ─────────────────────────────────────────────
#  Init — auto-detect provider from key prefix
# ─────────────────────────────────────────────

def init_gemini(api_key: str):
    """
    Initialize the AI provider from your API key.
      nvapi-... → NVIDIA NIM (recommended — free, no daily limit)
      gsk_...   → Groq
      AIza...   → Google Gemini
    """
    global _provider, _api_key
    _api_key = api_key.strip()

    if _api_key.startswith("nvapi-"):
        _provider = "nvidia"
        # quick validation — just import, actual call validates
        from openai import OpenAI  # NVIDIA uses OpenAI-compatible SDK

    elif _api_key.startswith("gsk_"):
        _provider = "groq"
        from groq import Groq
        Groq(api_key=_api_key)

    elif _api_key.startswith("AIza"):
        _provider = "gemini"
        import google.generativeai as genai
        genai.configure(api_key=_api_key)

    else:
        raise ValueError(
            "Unknown key format.\n"
            "• NVIDIA keys start with 'nvapi-'\n"
            "• Groq keys start with 'gsk_'\n"
            "• Gemini keys start with 'AIza'"
        )


# ─────────────────────────────────────────────
#  Internal: single LLM call (all providers)
# ─────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Send a prompt, return the text response."""

    if _provider == "nvidia":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=_api_key,
        )
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    elif _provider == "groq":
        from groq import Groq
        client = Groq(api_key=_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    elif _provider == "gemini":
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        return model.generate_content(prompt).text.strip()

    else:
        raise RuntimeError("AI not initialized. Please enter your API key in the sidebar.")


# ─────────────────────────────────────────────
#  Internal: multi-turn chat (all providers)
# ─────────────────────────────────────────────

def _call_llm_chat(messages: list, max_tokens: int = 400) -> str:
    """Send a full conversation history and return the assistant reply."""

    if _provider == "nvidia":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=_api_key,
        )
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    elif _provider == "groq":
        from groq import Groq
        client = Groq(api_key=_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    elif _provider == "gemini":
        # Gemini doesn't use system role — prepend to first user message
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        # Convert OpenAI-style messages to Gemini history
        gemini_history = []
        for msg in messages[:-1]:  # all but last
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        chat = model.start_chat(history=gemini_history)
        return chat.send_message(messages[-1]["content"]).text.strip()

    else:
        raise RuntimeError("AI not initialized.")


# ─────────────────────────────────────────────
#  Feature 1 — LLM Transaction Categorization
# ─────────────────────────────────────────────

def categorize_with_gemini(df: pd.DataFrame) -> pd.DataFrame:
    """Use LLM to categorize all transactions in one API call."""
    df = df.copy()
    descriptions = df["Description"].tolist()
    numbered = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))

    prompt = f"""You are a financial transaction categorizer for Indian users.

Categorize each transaction into EXACTLY one of these categories:
{', '.join(CATEGORIES)}

Transactions:
{numbered}

Rules:
- Swiggy, Zomato, food apps → Food & Dining
- Amazon, Flipkart, Myntra → Shopping
- Uber, Ola, metro, fuel, IRCTC → Transport
- Netflix, Hotstar, Spotify, cinema → Entertainment
- Electricity, phone bill, internet, recharge → Utilities & Bills
- Hospital, pharmacy, medicine → Healthcare
- Bank transfer, credit card payment, EMI → Credit Card & Banking
- Mutual fund, SIP, stocks, PPF → Investment & Savings
- Anything else → Other

Respond ONLY with a valid JSON array of strings, one category per transaction, same order.
Example: ["Food & Dining", "Shopping", "Transport"]
No explanation. No markdown. Just the JSON array."""

    try:
        raw = _call_llm(prompt, max_tokens=500)
        raw = re.sub(r"```json|```", "", raw).strip()
        categories = json.loads(raw)
        if len(categories) == len(descriptions):
            df["Category"] = categories
        else:
            df["Category"] = (categories + ["Other"] * len(descriptions))[:len(descriptions)]
    except Exception:
        df["Category"] = "Other"

    return df


# ─────────────────────────────────────────────
#  Feature 2 — AI Spending Insights
# ─────────────────────────────────────────────

def generate_ai_insights(
    df: pd.DataFrame,
    monthly_income: float,
    user_name: str = "there",
) -> str:
    """Generate personalized spending insights."""
    summary = df.groupby("Category")["Amount"].agg(["sum", "count"])
    summary.columns = ["Total (₹)", "Transactions"]
    summary = summary.sort_values("Total (₹)", ascending=False)

    total_spent = df["Amount"].sum()
    savings = monthly_income - total_spent
    savings_pct = (savings / monthly_income * 100) if monthly_income > 0 else 0

    prompt = f"""You are Finora, a friendly and smart personal finance advisor for Indian users.

User '{user_name}' has shared their spending data. Give helpful, specific, actionable advice.
Use Indian context (₹, SIP, UPI, Swiggy, etc.).

--- FINANCIAL DATA ---
Monthly Income   : ₹{monthly_income:,.0f}
Total Spent      : ₹{total_spent:,.0f}
Estimated Savings: ₹{savings:,.0f} ({savings_pct:.1f}%)
Transactions     : {len(df)}

Spending by Category:
{summary.to_string()}
----------------------

Write a financial analysis with these markdown sections:

### 🔍 Overall Assessment
2-3 sentences on their financial health.

### 🚨 Areas of Concern
2-3 specific issues with exact ₹ numbers. Be direct but kind.

### ✅ What They're Doing Well
1-2 genuine positives.

### 💡 Top 3 Actionable Tips
Numbered. Specific to their actual data. Mention real Indian alternatives
(cook at home vs Swiggy, start a SIP, use IRCTC instead of flight, etc.)

### 🎯 Savings Goal
Realistic monthly savings target with clear reasoning.

Keep total under 300 words. Use ₹ symbols. Be specific, not generic."""

    try:
        return _call_llm(prompt, max_tokens=1000)
    except Exception as e:
        return f"⚠️ Could not generate AI insights: {str(e)}\n\nPlease check your API key in the sidebar."


# ─────────────────────────────────────────────
#  Feature 3 — Chat with Finances
# ─────────────────────────────────────────────

def build_financial_context(df: pd.DataFrame, monthly_income: float) -> str:
    """Build a compact summary of finances for the chat system prompt."""
    summary = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    total = df["Amount"].sum()
    savings = monthly_income - total

    lines = [
        f"Monthly Income: ₹{monthly_income:,.0f}",
        f"Total Spent: ₹{total:,.0f}",
        f"Savings: ₹{savings:,.0f}",
        f"Transactions: {len(df)}",
        "",
        "Spending by Category:",
    ]
    for cat, amt in summary.items():
        pct = amt / total * 100 if total > 0 else 0
        lines.append(f"  {cat}: ₹{amt:,.0f} ({pct:.1f}%)")

    top5 = df.nlargest(5, "Amount")[["Description", "Amount", "Category"]]
    lines.append("\nTop 5 Transactions:")
    for _, row in top5.iterrows():
        lines.append(f"  {row['Description']}: ₹{row['Amount']:,.0f} ({row['Category']})")

    return "\n".join(lines)


def chat_with_finances(
    user_message: str,
    chat_history: list,
    financial_context: str,
) -> tuple:
    """
    Multi-turn chat about user's finances.
    chat_history uses {"role": "user"/"model", "parts": [text]} format internally.
    """
    system_prompt = (
        "You are Finora, a smart and friendly AI financial advisor for Indian users.\n"
        "You have the user's real financial data below. Answer based on this data.\n"
        "Be specific with exact ₹ numbers. Give practical Indian-context advice.\n"
        "Keep answers under 150 words unless asked for more detail.\n\n"
        f"USER'S FINANCIAL DATA:\n{financial_context}\n---"
    )

    try:
        # Convert internal history format → OpenAI-style messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            role = "assistant" if msg["role"] == "model" else "user"
            # Skip system-context injection from first message
            text = msg["parts"][0]
            if "USER'S FINANCIAL DATA" in text and role == "user":
                text = text.split("User says:")[-1].strip()
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        reply = _call_llm_chat(messages, max_tokens=400)
        chat_history.append({"role": "model", "parts": [reply]})
        chat_history.append({"role": "user", "parts": [user_message]})
        # Reorder: user message was appended after model reply — fix that
        chat_history[-2], chat_history[-1] = chat_history[-1], chat_history[-2]
        return reply, chat_history

    except Exception as e:
        error_msg = f"⚠️ AI error: {str(e)}"
        chat_history.append({"role": "user",  "parts": [user_message]})
        chat_history.append({"role": "model", "parts": [error_msg]})
        return error_msg, chat_history
