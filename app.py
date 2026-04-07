"""
app.py — Finora: AI Financial Intelligence System for India
Complete dashboard with all analysis modules.
"""

import os
import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from parser.bank_parser import parse_bank_statement, load_sample_transactions
from parser.csv_parser import parse_csv_excel
from analysis.spending_analysis import categorize_transactions, generate_spending_summary
from analysis.spending_patterns import detect_spending_patterns
from analysis.subscriptions import detect_subscriptions
from analysis.anomaly_detection import detect_anomalies
from analysis.savings_analysis import analyze_savings_behavior
from analysis.financial_score import calculate_financial_score
from simulation.financial_simulator import simulate_financial_scenarios, simulate_sip
from tax.tax_calculator import compare_tax_regimes
from utils.helpers import format_inr, dataframe_is_valid
from ai.gemini_ai import (
    init_gemini, categorize_with_gemini,
    generate_ai_insights, chat_with_finances, build_financial_context,
)
from ai.action_plan_generator import generate_financial_action_plan
from ai.rag_engine import rag_chat, index_transactions, _chromadb_available


# ─────────────────────────────────────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Finora", page_icon="💎", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);color:#e8e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d0d1f 0%,#1a1a3a 100%);border-right:1px solid rgba(100,100,255,0.15);}
.finora-header{text-align:center;padding:2.5rem 0 1.5rem 0;}
.finora-logo{font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-2px;margin-bottom:.3rem;}
.finora-tagline{font-size:1rem;color:#9ca3af;margin-top:-.2rem;letter-spacing:.3px;}
.finora-badges{display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap;margin-top:.9rem;}
.badge{background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.3);border-radius:999px;padding:.2rem .75rem;font-size:.75rem;color:#c4b5fd;font-family:'DM Sans',sans-serif;}
.section-title{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#c4b5fd;border-left:3px solid #7c3aed;padding-left:.75rem;margin:1.5rem 0 1rem 0;}
.ai-badge{display:inline-block;background:linear-gradient(90deg,#7c3aed,#2563eb);color:white;font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:20px;letter-spacing:.08em;margin-left:8px;vertical-align:middle;}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:1.2rem;margin-bottom:.8rem;}
.alert-card{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.6rem;font-size:.95rem;}
.success-card{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.6rem;font-size:.95rem;}
.insight-card{background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.2);border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.6rem;font-size:.95rem;line-height:1.5;}
.scenario-card{background:rgba(96,165,250,0.07);border:1px solid rgba(96,165,250,0.2);border-radius:12px;padding:1.2rem;margin-bottom:.8rem;}
.scenario-title{font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;margin-bottom:.5rem;color:#60a5fa;}
.chat-user{background:rgba(96,165,250,0.12);border:1px solid rgba(96,165,250,0.25);border-radius:12px 12px 4px 12px;padding:.8rem 1rem;margin:.5rem 0 .5rem 3rem;font-size:.95rem;}
.chat-bot{background:rgba(167,139,250,0.10);border:1px solid rgba(167,139,250,0.25);border-radius:12px 12px 12px 4px;padding:.8rem 1rem;margin:.5rem 3rem .5rem 0;font-size:.95rem;line-height:1.6;}
.chat-label-user{font-size:.72rem;color:#60a5fa;font-weight:600;margin-bottom:4px;}
.chat-label-bot{font-size:.72rem;color:#a78bfa;font-weight:600;margin-bottom:4px;}
.tax-old{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);border-radius:12px;padding:1.5rem;}
.tax-new{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:12px;padding:1.5rem;}
.tax-row{display:flex;justify-content:space-between;font-size:.88rem;padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.tax-total{font-family:'Syne',sans-serif;font-weight:700;font-size:1.2rem;margin-top:.5rem;}
.recommend-banner{background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:12px;padding:1.25rem 1.5rem;text-align:center;margin-top:1.5rem;}
.recommend-text{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:white;}
.recommend-reason{font-size:.9rem;color:rgba(255,255,255,0.8);margin-top:.4rem;}
.api-key-box{background:rgba(250,204,21,0.07);border:1px solid rgba(250,204,21,0.3);border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#2563eb);color:white;border:none;border-radius:8px;font-family:'Syne',sans-serif;font-weight:600;padding:.5rem 1.5rem;transition:opacity .2s;}
.stButton>button:hover{opacity:.85;}
div[data-testid="stMetricValue"]{font-family:'Syne',sans-serif;color:#a78bfa;}
.stProgress>div>div{background:linear-gradient(90deg,#7c3aed,#60a5fa);}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="finora-header">
    <div class="finora-logo">💎 Finora</div>
    <div class="finora-tagline">AI-Powered Financial Intelligence System for India</div>
    <div class="finora-badges">
        <span class="badge">🐍 Python</span>
        <span class="badge">⚡ Streamlit</span>
        <span class="badge">🤖 LLM (Groq / NVIDIA)</span>
        <span class="badge">🔍 RAG Engine</span>
        <span class="badge">📊 Pandas</span>
        <span class="badge">🏦 Indian Banks</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔑 AI API Key")
    st.markdown(
        '<div class="api-key-box">'
        '🆓 <b>NVIDIA</b> (recommended — no limit)<br>'
        '<a href="https://build.nvidia.com" target="_blank" style="color:#a78bfa;">build.nvidia.com</a>'
        ' → key starts with <code>nvapi-</code><br><br>'
        '🆓 <b>Groq</b> (fast &amp; free)<br>'
        '<a href="https://console.groq.com" target="_blank" style="color:#a78bfa;">console.groq.com</a>'
        ' → key starts with <code>gsk_</code>'
        '</div>', unsafe_allow_html=True,
    )
    api_key_input = st.text_input("Paste your API key", type="password",
                                   placeholder="nvapi-... or gsk_...", label_visibility="collapsed")
    gemini_ready = False
    if api_key_input:
        try:
            init_gemini(api_key_input)
            gemini_ready = True
            st.success("✅ AI connected!")
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.caption("Without a key, rule-based fallbacks are used.")

    st.markdown("---")
    st.markdown("### 🗂️ Navigation")
    active_tab = st.radio("Navigation", [
        "🏠 Dashboard",
        "📄 Bank Statement",
        "📊 Spending Analysis",
        "🔄 Subscriptions",
        "🚨 Anomaly Detection",
        "💰 Savings Analysis",
        "🏅 Financial Score",
        "🧮 What-If Simulator",
        "🤖 AI Insights",
        "🎯 Action Plan",
        "💬 Chat",
        "🧾 Tax Optimizer",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("© 2025 Finora · Educational use only")


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "transactions_df":    None,
    "categorized_df":     None,
    "chat_history":       [],
    "financial_context":  "",
    "monthly_income":     50000,
    "user_name":          "there",
    "user_goal":          "Wealth Creation",
    "user_risk":          "Medium",
    "user_savings_target": 10000,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: require data loaded
# ─────────────────────────────────────────────────────────────────────────────

def require_data():
    if not dataframe_is_valid(st.session_state.categorized_df):
        st.info("👈 Go to **Bank Statement** tab first and load your transactions.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  Profile Context Helper
# ─────────────────────────────────────────────────────────────────────────────

def get_profile_actions(goal: str, risk: str, monthly_income: float, savings_target: float) -> list:
    """
    Return 3 profile-aware recommended actions.
    Called by Dashboard and Action Plan — no core logic changed.
    """
    actions = []

    if goal == "Wealth Creation":
        sip = max(savings_target, int(monthly_income * 0.15))
        actions.append(f"📈 Start a SIP of ₹{sip:,.0f}/month in index funds (Nifty 50 / Flexi Cap).")
        actions.append("💹 Invest in ELSS funds — tax saving + wealth creation in one.")
        actions.append("🏦 Build 3-month emergency fund first, then increase SIP annually by 10%.")

    elif goal == "Tax Saving":
        remaining_80c = max(0, 150_000 - savings_target * 12)
        actions.append(f"🧾 Max out 80C (₹1.5L limit) — ₹{remaining_80c:,.0f} more to invest this year.")
        actions.append("🏥 Get health insurance to claim 80D deduction (up to ₹25,000).")
        actions.append("🏠 Consider NPS contribution for additional ₹50,000 deduction under 80CCD(1B).")

    elif goal == "Stability":
        ef_target = monthly_income * 6
        actions.append(f"🏦 Build emergency fund of ₹{ef_target:,.0f} (6× monthly income) in a liquid fund.")
        actions.append("🔄 Automate savings — set up auto-debit on salary day before spending.")
        actions.append("📊 Cut top spending category by 15% and move savings to RD or FD.")

    # Risk-based tweak
    if risk == "Low":
        actions.append("🛡️ Keep 80%+ in debt/FD instruments. Avoid volatile assets.")
    elif risk == "High":
        actions.append("🚀 With high risk tolerance, allocate up to 70% in equity for long-term growth.")

    return actions[:3]


def get_profile_insight_suffix(goal: str, risk: str) -> str:
    """Returns a one-line profile context to append to insights."""
    goal_map = {
        "Wealth Creation": "💡 Focus: grow your surplus through investments.",
        "Tax Saving":      "🧾 Focus: maximize deductions before March 31.",
        "Stability":       "🛡️ Focus: build a safety net before investing.",
    }
    risk_map = {
        "Low":    "Prefer stable, low-risk instruments.",
        "Medium": "Balanced approach recommended.",
        "High":   "You can afford higher-risk, higher-return options.",
    }
    return f"{goal_map.get(goal, '')} {risk_map.get(risk, '')}"


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 0 — Financial Dashboard (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

if active_tab == "🏠 Dashboard":
    st.markdown('<div class="section-title">🏠 Financial Dashboard</div>', unsafe_allow_html=True)

    # ── 1: Your Profile ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Your Profile</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        user_name = st.text_input(
            "Your Name", value=st.session_state.user_name,
            placeholder="e.g. Arjun"
        )
        st.session_state.user_name = user_name

        monthly_income = st.number_input(
            "Monthly Income (₹)", min_value=0, max_value=10_000_000,
            value=st.session_state.monthly_income, step=1000, format="%d"
        )
        st.session_state.monthly_income = monthly_income

    with p2:
        user_goal = st.selectbox(
            "Financial Goal",
            ["Wealth Creation", "Tax Saving", "Stability"],
            index=["Wealth Creation", "Tax Saving", "Stability"].index(st.session_state.user_goal)
        )
        st.session_state.user_goal = user_goal

        user_risk = st.selectbox(
            "Risk Level",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(st.session_state.user_risk)
        )
        st.session_state.user_risk = user_risk

    with p3:
        user_savings_target = st.number_input(
            "Monthly Savings Target (₹)", min_value=0, max_value=1_000_000,
            value=st.session_state.user_savings_target, step=500, format="%d"
        )
        st.session_state.user_savings_target = user_savings_target

        st.markdown("<br>", unsafe_allow_html=True)
        advice_style = "Conservative 🛡️" if user_risk == "Low" else "Balanced ⚖️" if user_risk == "Medium" else "Aggressive 🚀"
        st.markdown(
            f'<div class="card" style="padding:.75rem;text-align:center;">'
            f'<small>Advice Style</small><br><strong>{advice_style}</strong></div>',
            unsafe_allow_html=True
        )

    if user_name and user_name != "there":
        st.markdown(
            f'<div style="color:#a78bfa;font-family:Syne,sans-serif;font-size:1rem;margin:.5rem 0;">'
            f'👋 Welcome, <strong>{user_name}</strong>! Here\'s your financial overview.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    if not dataframe_is_valid(st.session_state.categorized_df):
        st.markdown("""
        <div class="insight-card" style="text-align:center;padding:2.5rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;
                 background:linear-gradient(90deg,#a78bfa,#60a5fa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Ready to analyze your finances 💎
            </div>
            <div style="margin-top:1rem;color:#9ca3af;font-size:1rem;">
                Upload your bank statement to see your complete financial picture.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card" style="text-align:center;">📄<br><strong>Upload PDF</strong><br><small>HDFC / SBI statements</small></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card" style="text-align:center;">📊<br><strong>Upload CSV</strong><br><small>Any bank export</small></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card" style="text-align:center;">🔍<br><strong>Sample Data</strong><br><small>Try with demo data</small></div>', unsafe_allow_html=True)
        st.info("👈 Go to **Bank Statement** tab to get started.")
        st.stop()

    # ── Pull data from existing modules ──────────────────────────────────────
    cat_df = st.session_state.categorized_df

    savings_res  = analyze_savings_behavior(cat_df, monthly_income)
    score_res    = calculate_financial_score(cat_df, monthly_income)
    anomaly_res  = detect_anomalies(cat_df)
    sub_res      = detect_subscriptions(cat_df)
    patterns_res = detect_spending_patterns(cat_df)

    # ── 2: Financial Snapshot ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📸 Financial Snapshot</div>', unsafe_allow_html=True)

    # Show data period label for explainability
    data_period = savings_res.get("data_period", "")
    num_months  = savings_res.get("num_months", 1)
    total_raw   = savings_res.get("total_raw_spend", savings_res["total_spent"])
    if data_period:
        st.markdown(
            f'<div style="font-size:.85rem;color:#9ca3af;margin-bottom:.75rem;">'
            f'📅 Analysis based on <strong style="color:#a78bfa;">{data_period}</strong> of transaction data. '
            f'All values shown are <strong>monthly averages</strong>.</div>',
            unsafe_allow_html=True
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Monthly Income",    format_inr(monthly_income))
    c2.metric("Avg Monthly Spend", format_inr(savings_res["monthly_spend"]),
              help=f"Total ₹{total_raw:,.0f} ÷ {num_months} months")
    c3.metric("Monthly Savings",   format_inr(max(savings_res["savings"], 0)))
    c4.metric("Savings Rate",      f"{savings_res['savings_rate']:.1f}%",
              delta="Good ✅" if savings_res["savings_rate"] >= 20 else "Low ⚠️")
    c5.metric("Health Score",      f"{score_res['score']}/100",
              delta=score_res["grade"])

    score_color = score_res["grade_color"]
    st.markdown(f"""
    <div style="margin:.5rem 0 1.5rem 0;">
        <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;">
            <div style="background:{score_color};width:{score_res['score']}%;
                 height:8px;border-radius:999px;"></div>
        </div>
        <div style="font-size:.8rem;color:#9ca3af;margin-top:.3rem;">
            {score_res['grade_emoji']} {score_res['grade']} Financial Health
            &nbsp;·&nbsp; Total spend across all months: {format_inr(total_raw)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    # ── B: Key Insights ───────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)

        # Collect top insights from existing modules
        all_insights = []

        # From spending patterns
        if patterns_res["patterns"]:
            all_insights.append(("💡", patterns_res["patterns"][0]))

        # From savings analysis
        if savings_res["insights"]:
            all_insights.append(("💡", savings_res["insights"][0]))

        # From subscriptions
        if sub_res["insights"] and sub_res["total_monthly"] > 0:
            all_insights.append(("💡", sub_res["insights"][0]))

        # From score tips
        if score_res["tips"]:
            all_insights.append(("💡", score_res["tips"][0]))

        # Show top 3
        for emoji, insight in all_insights[:3]:
            st.markdown(
                f'<div class="insight-card">{emoji} {insight}</div>',
                unsafe_allow_html=True
            )

    # ── C: Alerts ─────────────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-title">⚠️ Alerts</div>', unsafe_allow_html=True)

        alerts = []

        # Anomaly alerts
        for a in anomaly_res["anomalies"][:2]:
            alerts.append(
                f"🚨 **{a['description']}** — ₹{a['amount']:,.0f} "
                f"({a['multiplier']:.1f}× your avg in {a['category']})"
            )

        # High subscription cost
        if sub_res["total_monthly"] > monthly_income * 0.10:
            alerts.append(
                f"🔄 Subscriptions cost ₹{sub_res['total_monthly']:,.0f}/month "
                f"({sub_res['total_monthly']/monthly_income*100:.1f}% of income)"
            )

        # Low savings rate
        if savings_res["savings_rate"] < 10:
            alerts.append(
                f"💸 Savings rate is only {savings_res['savings_rate']:.1f}% — "
                f"below the recommended 20%"
            )

        # Overspending
        if savings_res["savings"] < 0:
            alerts.append(
                f"🚨 Overspending by ₹{abs(savings_res['savings']):,.0f} this period!"
            )

        if alerts:
            for alert in alerts[:3]:
                st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">✅ No alerts — your finances look healthy!</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── D: Recommended Actions ────────────────────────────────────────────────
    st.markdown('<div class="section-title">✅ Recommended Actions</div>', unsafe_allow_html=True)

    # Profile-aware actions (top 3)
    profile_actions = get_profile_actions(
        user_goal, user_risk,
        monthly_income, user_savings_target
    )
    # Supplement with score tips if not enough
    all_actions = profile_actions + score_res["tips"]
    if savings_res["recommendation"]:
        all_actions.append(f"💡 {savings_res['recommendation']}")

    action_cols = st.columns(3)
    for col, action in zip(action_cols, all_actions[:3]):
        with col:
            st.markdown(
                f'<div class="card" style="min-height:100px;">'
                f'<div style="font-size:.85rem;line-height:1.5;">{action}</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Quick navigation ──────────────────────────────────────────────────────
    # ── 8: Monthly Spending Trend Chart ─────────────────────────────────────
    st.markdown('<div class="section-title">📈 Monthly Spending Trend</div>', unsafe_allow_html=True)
    try:
        trend_df = cat_df.copy()
        trend_df["_date"] = pd.to_datetime(trend_df["Date"], dayfirst=True, errors="coerce")
        trend_df = trend_df.dropna(subset=["_date"])
        trend_df["Month"] = trend_df["_date"].dt.to_period("M")
        monthly_totals = (
            trend_df.groupby("Month")["Amount"].sum()
            .reset_index()
        )
        monthly_totals["Month"] = monthly_totals["Month"].dt.to_timestamp().dt.strftime("%b %Y")
        monthly_totals = monthly_totals.rename(columns={"Amount": "Spend (₹)"})
        st.bar_chart(monthly_totals.set_index("Month"), color="#a78bfa")
    except Exception:
        pass   # silently skip if data not ready

    st.markdown("---")

    st.markdown('<div class="section-title">🔗 Explore Details</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#9ca3af;font-size:.9rem;margin-bottom:1rem;">Use the sidebar to dive deeper into any area.</div>', unsafe_allow_html=True)
    nav_cols = st.columns(5)
    nav_items = [("📊","Spending Analysis"),("🔄","Subscriptions"),("🚨","Anomalies"),("🧮","What-If Simulator"),("🎯","Action Plan")]
    for col, (emoji, label) in zip(nav_cols, nav_items):
        with col:
            st.markdown(f'<div class="card" style="text-align:center;padding:.8rem;">{emoji}<br><small>{label}</small></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Bank Statement
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "📄 Bank Statement":
    st.markdown('<div class="section-title">📄 Upload Bank Statement</div>', unsafe_allow_html=True)

    # ── Upload method selector ────────────────────────────────────────────────
    upload_method = st.radio(
        "Choose how to load your data:",
        ["📊 CSV / Excel  (recommended)", "📄 PDF", "🔍 Use Sample Data"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    df = None

    # ── CSV / Excel path ──────────────────────────────────────────────────────
    if upload_method == "📊 CSV / Excel  (recommended)":
        st.markdown(
            '<div class="insight-card">'
            '📊 <strong>Export your bank statement as CSV or Excel</strong><br>'
            'In your bank app or net banking → Statements → Download → CSV/Excel<br>'
            '<small style="color:#9ca3af;">Supports HDFC, SBI, ICICI, Axis, Kotak and most Indian banks</small>'
            '</div>', unsafe_allow_html=True
        )
        csv_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Export from your bank's net banking or mobile app as CSV or Excel."
        )
        if csv_file:
            with st.spinner("📊 Parsing file..."):
                try:
                    df = parse_csv_excel(csv_file)
                    if df.empty:
                        st.warning("⚠️ File parsed but no valid transactions found. Check your file format.")
                        df = None
                    else:
                        st.success(f"✅ Parsed **{len(df)} transactions** from {csv_file.name}!")
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.markdown(
                        '<div class="insight-card">'
                        '💡 <strong>Tip:</strong> Make sure your file has columns named like:<br>'
                        '&nbsp;&nbsp;• <code>Date</code> or <code>Txn Date</code><br>'
                        '&nbsp;&nbsp;• <code>Description</code> or <code>Narration</code><br>'
                        '&nbsp;&nbsp;• <code>Amount</code> or <code>Withdrawal</code> or <code>Debit</code>'
                        '</div>', unsafe_allow_html=True
                    )

    # ── PDF path ──────────────────────────────────────────────────────────────
    elif upload_method == "📄 PDF":
        st.markdown(
            '<div class="insight-card">'
            '📄 <strong>Upload a PDF bank statement</strong><br>'
            'Works best with text-based PDFs from HDFC and SBI.<br>'
            '<small style="color:#9ca3af;">Scanned/image PDFs are not supported — use CSV instead.</small>'
            '</div>', unsafe_allow_html=True
        )
        pdf_file = st.file_uploader(
            "Upload PDF bank statement",
            type=["pdf"],
            help="HDFC and SBI PDFs work best. Other banks use a generic parser."
        )
        if pdf_file:
            with st.spinner("🔍 Parsing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
                df = parse_bank_statement(tmp_path)
                os.unlink(tmp_path)
            if dataframe_is_valid(df):
                method = df.get("Extraction_Method", pd.Series(["pdfplumber"])).iloc[0] if "Extraction_Method" in df.columns else "pdfplumber"
                st.success(f"✅ Extracted **{len(df)} transactions** from PDF! *(via {method})*")
            elif "_ocr_needed" in df.columns:
                st.warning(
                    "⚠️ This PDF appears to be **scanned** (image-based). "
                    "To read it, install Poppler:\n\n"
                    "```bash\nbrew install poppler\n```\n\n"
                    "Then restart the app and try again. "
                    "Alternatively, export your statement as **CSV** from your bank's app."
                )
            else:
                st.warning(
                    "⚠️ Could not extract transactions from this PDF. "
                    "It may be in an unsupported format. "
                    "Try downloading as **CSV** from your bank instead."
                )

    # ── Sample data path ──────────────────────────────────────────────────────
    elif upload_method == "🔍 Use Sample Data":
        st.markdown(
            '<div class="insight-card">'
            '🔍 <strong>6 months of realistic sample data</strong> — Oct 2024 to Mar 2025<br>'
            'Includes subscriptions, anomalies, and varied spending patterns<br>'
            '<small style="color:#9ca3af;">Perfect for exploring all Finora features</small>'
            '</div>', unsafe_allow_html=True
        )
        if st.button("▶️ Load Sample Data", use_container_width=False):
            df = load_sample_transactions()
            st.success("✅ Loaded 90 sample transactions (6 months)!")

    # ── Categorize and store ──────────────────────────────────────────────────
    if df is not None and dataframe_is_valid(df):
        with st.spinner("🤖 Categorizing..." if gemini_ready else "📂 Categorizing..."):
            cat_df = categorize_with_gemini(df) if gemini_ready else categorize_transactions(df)
        # Apply merchant-based category correction after LLM categorization
        # Fixes cases where Groq/NVIDIA assigns wrong categories (e.g. Flipkart as Food & Dining)
        from utils.categorizer import correct_categories
        cat_df = correct_categories(cat_df)
        st.session_state.transactions_df = df
        st.session_state.categorized_df  = cat_df
        st.session_state.financial_context = ""
        unknown_count = int(df["Needs_Review"].sum()) if "Needs_Review" in df.columns else 0
        if unknown_count:
            st.warning(f"⚠️ **{unknown_count} transactions** have unclear narrations — label them below.")

    if dataframe_is_valid(st.session_state.transactions_df):
        df     = st.session_state.transactions_df
        cat_df = st.session_state.categorized_df

        # ── Manual Labeling UI ────────────────────────────────────────────────
        if "Needs_Review" in df.columns and df["Needs_Review"].sum() > 0:
            unknown_df = df[df["Needs_Review"] == True].copy()
            st.markdown("---")
            st.markdown('<div class="section-title">✏️ Review Unknown Transactions</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="insight-card">'
                f'🔍 <strong>{len(unknown_df)} transaction(s)</strong> could not be identified automatically '
                f'(e.g. P2P transfers, ATM withdrawals).<br><br>'
                f'You can <strong>label them now</strong> for better insights, or '
                f'<strong>skip and continue</strong> — they\'ll be categorized as "Other".'
                f'</div>',
                unsafe_allow_html=True
            )

            # Skip button at the top so it's easy to find
            col_skip, col_space = st.columns([1, 3])
            with col_skip:
                if st.button("⏭️ Skip & Continue", use_container_width=True):
                    # Mark all as reviewed, keep as Other
                    st.session_state.transactions_df["Needs_Review"] = False
                    if "Needs_Review" in st.session_state.categorized_df.columns:
                        st.session_state.categorized_df["Needs_Review"] = False
                    st.success("✅ Skipped — unknown transactions marked as 'Other'.")
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            CATEGORY_OPTIONS = [
                "Food & Dining", "Shopping", "Transport", "Entertainment",
                "Utilities & Bills", "Healthcare", "Credit Card & Banking",
                "Investment & Savings", "Other"
            ]

            for idx in unknown_df.index:
                row = df.loc[idx]
                with st.expander(
                    f"Transaction {idx+1} — ₹{row['Amount']:,.0f} on {row['Date']}  ·  {row['Description']}",
                    expanded=False
                ):
                    st.caption(f"Raw narration: `{row['Description']}`")
                    col_a, col_b = st.columns([2, 2])
                    with col_a:
                        st.text_input(
                            "What was this payment for? (optional)",
                            value="",
                            key=f"desc_{idx}",
                            placeholder="e.g. Rent, Petrol, Friend repayment",
                        )
                    with col_b:
                        st.selectbox(
                            "Category",
                            CATEGORY_OPTIONS,
                            index=8,   # default to "Other"
                            key=f"cat_{idx}",
                        )

            st.markdown("<br>", unsafe_allow_html=True)
            col_save, col_skip2, _ = st.columns([1, 1, 2])
            with col_save:
                if st.button("✅ Save Labels", use_container_width=True):
                    for idx in unknown_df.index:
                        new_desc = st.session_state.get(f"desc_{idx}", "")
                        new_cat  = st.session_state.get(f"cat_{idx}", "Other")
                        if new_desc:
                            st.session_state.transactions_df.at[idx, "Description"] = new_desc
                            st.session_state.categorized_df.at[idx,  "Description"] = new_desc
                        if "Category" in st.session_state.categorized_df.columns:
                            st.session_state.categorized_df.at[idx, "Category"] = new_cat
                        st.session_state.transactions_df.at[idx, "Needs_Review"] = False
                    st.success("✅ Labels saved!")
                    st.rerun()
            with col_skip2:
                if st.button("⏭️ Skip All", use_container_width=True):
                    st.session_state.transactions_df["Needs_Review"] = False
                    if "Needs_Review" in st.session_state.categorized_df.columns:
                        st.session_state.categorized_df["Needs_Review"] = False
                    st.rerun()

        # ── Transactions Table ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📋 Transactions</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions",    len(df))
        c2.metric("Total Spent",     format_inr(df["Amount"].sum()))
        c3.metric("Avg Transaction", format_inr(df["Amount"].mean()))
        c4.metric("Largest",         format_inr(df["Amount"].max()))
        st.markdown("<br>", unsafe_allow_html=True)

        cols_to_show = ["Date", "Description", "Category", "Amount"] \
            if "Category" in cat_df.columns else ["Date", "Description", "Amount"]
        disp = cat_df[cols_to_show].copy()
        disp["Amount"] = disp["Amount"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(disp, use_container_width=True, height=400)

    else:
        st.info("👆 Upload a PDF or click **Use Sample Data** to get started.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Spending Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "📊 Spending Analysis":
    st.markdown('<div class="section-title">📊 Spending Analysis</div>', unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df
    monthly_income = st.number_input("Your monthly income (₹)", min_value=0, max_value=10_000_000,
                                      value=st.session_state.monthly_income, step=1000, format="%d")
    st.session_state.monthly_income = monthly_income

    summary = generate_spending_summary(cat_df)
    total   = summary.sum()
    st.markdown("---")

    col_chart, col_table = st.columns(2)
    with col_chart:
        st.markdown("**Spending by Category**")
        chart_data = summary.reset_index()
        chart_data.columns = ["Category", "Amount"]
        st.bar_chart(chart_data.set_index("Category")["Amount"])
    with col_table:
        st.markdown("**Breakdown**")
        for cat, amt in summary.items():
            pct = amt / total * 100 if total > 0 else 0
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:.5rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.07);">'
                f'<span>{cat}</span><span style="font-family:Syne,sans-serif;font-weight:600;'
                f'color:#a78bfa;">{format_inr(amt)} <span style="color:#9ca3af;font-size:.8rem;">'
                f'({pct:.1f}%)</span></span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Spending Patterns</div>', unsafe_allow_html=True)
    patterns_result = detect_spending_patterns(cat_df)
    for p in patterns_result["patterns"]:
        st.markdown(f'<div class="insight-card">{p}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Subscriptions
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🔄 Subscriptions":
    st.markdown('<div class="section-title">🔄 Subscription Detection</div>', unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df
    result = detect_subscriptions(cat_df)

    c1, c2 = st.columns(2)
    c1.metric("Subscriptions Found", result["count"] if "count" in result else len(result["subscriptions"]))
    c2.metric("Monthly Cost", format_inr(result["total_monthly"]))

    st.markdown("---")
    for insight in result["insights"]:
        card_class = "alert-card" if "⚠️" in insight else "insight-card"
        st.markdown(f'<div class="{card_class}">{insight}</div>', unsafe_allow_html=True)

    if result["subscriptions"]:
        st.markdown('<div class="section-title">📋 Detected Subscriptions</div>', unsafe_allow_html=True)
        sub_df = pd.DataFrame(result["subscriptions"])[["name", "amount", "frequency"]]
        sub_df.columns = ["Service", "Amount (₹)", "Frequency"]
        sub_df["Amount (₹)"] = sub_df["Amount (₹)"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(sub_df, use_container_width=True)

        annual = result["total_monthly"] * 12
        st.markdown(
            f'<div class="alert-card">💡 You spend <strong>₹{result["total_monthly"]:,.0f}/month</strong> '
            f'= <strong>₹{annual:,.0f}/year</strong> on subscriptions. '
            f'Cancelling even one unused service could save ₹1,000+ annually.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🚨 Anomaly Detection":
    st.markdown('<div class="section-title">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df

    threshold = st.slider("Flag transactions above N × category average", 2.0, 5.0, 3.0, 0.5)
    result = detect_anomalies(cat_df, threshold_multiplier=threshold)

    c1, c2 = st.columns(2)
    c1.metric("Anomalies Detected", result["count"])
    c2.metric("Threshold", f"{threshold}× avg")

    st.markdown("---")
    if result["anomalies"]:
        st.markdown('<div class="section-title">🔍 Unusual Transactions</div>', unsafe_allow_html=True)
        for alert in result["alerts"]:
            st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)

        anom_df = pd.DataFrame(result["anomalies"])
        anom_df["amount"]       = anom_df["amount"].apply(lambda x: f"₹{x:,.0f}")
        anom_df["category_avg"] = anom_df["category_avg"].apply(lambda x: f"₹{x:,.0f}")
        anom_df["multiplier"]   = anom_df["multiplier"].apply(lambda x: f"{x:.1f}×")
        anom_df.columns         = ["Date","Description","Category","Amount","Cat. Average","Multiplier"]
        st.dataframe(anom_df, use_container_width=True)
    else:
        for alert in result["alerts"]:
            st.markdown(f'<div class="success-card">{alert}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Savings Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "💰 Savings Analysis":
    st.markdown('<div class="section-title">💰 Savings Behavior Analysis</div>', unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df
    monthly_income = st.number_input("Your monthly income (₹)", min_value=0, max_value=10_000_000,
                                      value=st.session_state.monthly_income, step=1000, format="%d")
    st.session_state.monthly_income = monthly_income
    st.markdown("---")

    result = analyze_savings_behavior(cat_df, monthly_income)

    # Show data period
    if result.get("data_period"):
        st.markdown(
            f'<div style="font-size:.85rem;color:#9ca3af;margin-bottom:.75rem;">'
            f'📅 Data: <strong style="color:#a78bfa;">{result["data_period"]}</strong> · '
            f'All values are <strong>monthly averages</strong>.</div>',
            unsafe_allow_html=True
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend", format_inr(result["monthly_spend"]))
    c2.metric("Total Spend",       format_inr(result["total_raw_spend"]),
              help=f"Across {result['num_months']} months")
    c3.metric("Monthly Savings",   format_inr(result["savings"]))
    c4.metric("Savings Rate",      f"{result['savings_rate']:.1f}%",
              delta="Good ✅" if result["status"] == "good" else "Needs work ⚠️")

    st.markdown("<br>", unsafe_allow_html=True)
    rate = max(min(result["savings_rate"] / 100, 1.0), 0.0)
    st.progress(rate, text=f"Savings Rate: {result['savings_rate']:.1f}% (Target: 30%)")

    st.markdown("---")
    for insight in result["insights"]:
        card = "success-card" if result["status"] == "good" else "alert-card" if result["status"] == "poor" else "insight-card"
        st.markdown(f'<div class="{card}">{insight}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="insight-card">💡 <strong>Recommendation:</strong> {result["recommendation"]}</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — Financial Health Score
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🏅 Financial Score":
    st.markdown('<div class="section-title">🏅 Financial Health Score</div>', unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df
    monthly_income = st.number_input("Your monthly income (₹)", min_value=0, max_value=10_000_000,
                                      value=st.session_state.monthly_income, step=1000, format="%d")
    st.session_state.monthly_income = monthly_income
    st.markdown("---")

    result = calculate_financial_score(cat_df, monthly_income)
    score  = result["score"]
    grade  = result["grade"]
    emoji  = result["grade_emoji"]
    color  = result["grade_color"]

    # ── Big Score Display ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:2rem;background:rgba(255,255,255,0.03);
         border:1px solid {color}40;border-radius:16px;margin-bottom:1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:5rem;font-weight:800;
             color:{color};line-height:1;">{score}</div>
        <div style="font-size:1.1rem;color:#9ca3af;margin-top:.25rem;">out of 100</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;
             color:{color};margin-top:.5rem;">{emoji} {grade}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score progress bar
    st.progress(score / 100, text=f"Financial Health: {score}/100")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="insight-card">{result["explanation"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Score Breakdown ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Score Breakdown</div>', unsafe_allow_html=True)

    breakdown = result["score_breakdown"]
    max_pts   = result["max_breakdown"]

    for factor, pts in breakdown.items():
        mx  = max_pts[factor]
        pct = pts / mx if mx > 0 else 0
        bar_color = "#34d399" if pct >= 0.75 else "#fbbf24" if pct >= 0.5 else "#f87171"

        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">
                <span style="font-weight:600;">{factor}</span>
                <span style="font-family:'Syne',sans-serif;color:{bar_color};font-weight:700;">
                    {pts} / {mx} pts
                </span>
            </div>
            <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:10px;">
                <div style="background:{bar_color};width:{pct*100:.0f}%;height:10px;
                     border-radius:999px;transition:width .3s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Stats ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Savings Rate",    f"{result['savings_rate']:.1f}%")
    c2.metric("Subscription Cost", format_inr(result["sub_cost"]))
    c3.metric("Anomalies Found", result["anomaly_count"])

    st.markdown("---")

    # ── Tips ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 How to Improve Your Score</div>', unsafe_allow_html=True)
    for tip in result["tips"]:
        st.markdown(f'<div class="insight-card">{tip}</div>', unsafe_allow_html=True)

    # ── Grade reference ───────────────────────────────────────────────────────
    with st.expander("📋 Score Grading Reference"):
        st.markdown("""
| Score | Grade | Meaning |
|-------|-------|---------|
| 80–100 | 🏆 Excellent | Strong savings, low debt, stable spending |
| 60–79  | ✅ Good | Decent habits with room to improve |
| 40–59  | ⚠️ Needs Improvement | Multiple areas need attention |
| 0–39   | 🚨 Risky | Immediate financial changes needed |
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — What-If Simulator
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🧮 What-If Simulator":
    st.markdown('<div class="section-title">🧮 Financial What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown("Simulations based on your **actual surplus** — not full income.")

    monthly_income = st.session_state.monthly_income

    # ── Compute realistic surplus from transactions ───────────────────────────
    if dataframe_is_valid(st.session_state.categorized_df):
        df = st.session_state.categorized_df
        total_expense = df["Amount"].sum()

        # Estimate number of months in data
        try:
            dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
            if len(dates) > 1:
                months_in_data = max(1, round((dates.max() - dates.min()).days / 30))
            else:
                months_in_data = 1
        except Exception:
            months_in_data = 1

        monthly_expenses = total_expense / months_in_data
    else:
        monthly_expenses = monthly_income * 0.70   # conservative fallback

    monthly_surplus = max(monthly_income - monthly_expenses, 0)
    usable_amount   = monthly_surplus * 0.70   # 70% of surplus, 30% kept as buffer

    # ── Surplus summary banner ────────────────────────────────────────────────
    surplus_color = "#34d399" if monthly_surplus > 0 else "#f87171"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid {surplus_color}40;
         border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div><small style="color:#9ca3af;">Monthly Income</small><br>
                 <strong style="font-family:Syne,sans-serif;font-size:1.1rem;">{format_inr(monthly_income)}</strong></div>
            <div><small style="color:#9ca3af;">Avg Monthly Expenses</small><br>
                 <strong style="font-family:Syne,sans-serif;font-size:1.1rem;color:#f87171;">{format_inr(monthly_expenses)}</strong></div>
            <div><small style="color:#9ca3af;">Monthly Surplus</small><br>
                 <strong style="font-family:Syne,sans-serif;font-size:1.1rem;color:{surplus_color};">{format_inr(monthly_surplus)}</strong></div>
            <div><small style="color:#9ca3af;">Usable for Goals (70%)</small><br>
                 <strong style="font-family:Syne,sans-serif;font-size:1.1rem;color:#a78bfa;">{format_inr(usable_amount)}</strong></div>
        </div>
        <div style="font-size:.8rem;color:#9ca3af;margin-top:.5rem;">
            ℹ️ 70% of surplus allocated to goals · 30% kept as buffer
        </div>
    </div>
    """, unsafe_allow_html=True)

    if monthly_surplus <= 0:
        st.warning("⚠️ Your expenses exceed your income. Reduce spending before starting investments.")
        st.stop()

    st.markdown("---")

    # ── Custom SIP Calculator ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">💹 Custom SIP Calculator</div>', unsafe_allow_html=True)
    st.caption(f"Based on your finances, you can realistically invest up to **{format_inr(usable_amount)}/month**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        # Default SIP = usable amount, capped sensibly
        sip_default = min(int(usable_amount), 50000)
        sip_amount = st.number_input("Monthly SIP Amount (₹)", 500, 1_000_000, sip_default, 500, format="%d")
        if sip_amount > monthly_surplus:
            st.warning(f"⚠️ ₹{sip_amount:,} exceeds your surplus of {format_inr(monthly_surplus)}.")
        elif sip_amount > usable_amount:
            st.caption(f"⚠️ Dipping into buffer. Recommended max: {format_inr(usable_amount)}")
    with col2:
        sip_years  = st.slider("Investment Duration (Years)", 1, 30, 10)
    with col3:
        sip_return = st.slider("Expected Annual Return (%)", 6.0, 20.0, 12.0, 0.5)

    sip = simulate_sip(sip_amount, sip_years, sip_return)

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Invested",  format_inr(sip["total_invested"]))
    r2.metric("Future Value",    format_inr(sip["future_value"]))
    r3.metric("Wealth Gained",   format_inr(sip["wealth_gained"]))

    milestone_data = {
        "Year":     list(range(1, sip_years + 1)),
        "Value (₹)":[simulate_sip(sip_amount, y, sip_return)["future_value"] for y in range(1, sip_years + 1)],
    }
    st.line_chart(pd.DataFrame(milestone_data).set_index("Year"))

    st.markdown("---")

    # ── Pre-built Scenarios (uses actual surplus) ─────────────────────────────
    st.markdown('<div class="section-title">📋 Pre-built Scenarios</div>', unsafe_allow_html=True)
    st.caption(f"All scenarios use your actual surplus of **{format_inr(monthly_surplus)}/month** (not full income).")
    scenarios = simulate_financial_scenarios(monthly_income, monthly_expenses)

    for scenario in scenarios:
        with st.expander(scenario["title"], expanded=False):
            st.markdown(f"*{scenario['description']}*")
            st.markdown("---")
            for line in scenario["results"]:
                st.markdown(f"**{line}**")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — AI Insights
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🤖 AI Insights":
    st.markdown(
        '<div class="section-title">🤖 AI Insights <span class="ai-badge">LLM</span></div>',
        unsafe_allow_html=True)
    require_data()

    if not gemini_ready:
        st.warning("⚠️ Add your API key in the sidebar to unlock AI insights.")
        st.stop()

    cat_df = st.session_state.categorized_df
    col1, col2 = st.columns([2, 1])
    with col1:
        monthly_income = st.number_input("Your monthly income (₹)", min_value=0, max_value=10_000_000,
                                          value=st.session_state.monthly_income, step=1000, format="%d")
        st.session_state.monthly_income = monthly_income
    with col2:
        user_name = st.text_input("Your name (optional)", value="there")

    if st.button("✨ Generate AI Insights"):
        with st.spinner("🤖 Analyzing your finances..."):
            from ai.gemini_ai import generate_ai_insights
            insights_md = generate_ai_insights(cat_df, monthly_income, user_name)
            st.session_state.financial_context = build_financial_context(cat_df, monthly_income)
        st.markdown("---")
        st.markdown(insights_md)
        st.info("💬 Switch to the **Chat** tab to ask follow-up questions!")
    else:
        st.markdown('<div class="insight-card" style="text-align:center;padding:2rem;">✨ Click the button above for a personalized AI analysis</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 8 — Action Plan
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🎯 Action Plan":
    st.markdown(
        '<div class="section-title">🎯 AI Financial Action Plan <span class="ai-badge">LLM</span></div>',
        unsafe_allow_html=True)
    require_data()

    cat_df = st.session_state.categorized_df
    monthly_income = st.session_state.monthly_income

    monthly_income = st.number_input("Your monthly income (₹)", min_value=0, max_value=10_000_000,
                                      value=monthly_income, step=1000, format="%d")
    st.session_state.monthly_income = monthly_income

    # Read profile from session state (set in Dashboard)
    user_goal          = st.session_state.get("user_goal", "Wealth Creation")
    user_risk          = st.session_state.get("user_risk", "Medium")
    user_savings_target= st.session_state.get("user_savings_target", 10000)

    st.markdown(
        f'<div class="insight-card">👤 <strong>Profile:</strong> Goal — {user_goal} &nbsp;|&nbsp; '
        f'Risk — {user_risk} &nbsp;|&nbsp; Savings Target — {format_inr(user_savings_target)}/month<br>'
        f'<small style="color:#9ca3af;">Update your profile in the 🏠 Dashboard tab.</small></div>',
        unsafe_allow_html=True
    )

    if st.button("🎯 Generate My Action Plan", use_container_width=False):
        with st.spinner("🤖 Building your personalized action plan..."):
            patterns_res  = detect_spending_patterns(cat_df)
            anomaly_res   = detect_anomalies(cat_df)
            sub_res       = detect_subscriptions(cat_df)
            savings_res   = analyze_savings_behavior(cat_df, monthly_income)

            data_summary = {
                "monthly_income":      monthly_income,
                "total_spent":         savings_res["total_spent"],
                "savings_rate":        savings_res["savings_rate"],
                "top_category":        patterns_res["top_category"],
                "anomaly_count":       anomaly_res["count"],
                "subscriptions_cost":  sub_res["total_monthly"],
                "patterns":            patterns_res["patterns"],
                "anomalies":           anomaly_res["anomalies"],
                "subscriptions":       sub_res["subscriptions"],
            }

            plan_md = generate_financial_action_plan(data_summary)

        st.markdown("---")
        st.markdown(plan_md)

        # Profile-specific actions below AI plan
        st.markdown("---")
        st.markdown('<div class="section-title">👤 Profile-Based Actions</div>', unsafe_allow_html=True)
        profile_actions = get_profile_actions(user_goal, user_risk, monthly_income, user_savings_target)
        for action in profile_actions:
            st.markdown(f'<div class="insight-card">{action}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Analysis Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Savings Rate",    f"{savings_res['savings_rate']:.1f}%")
        c2.metric("Anomalies",       anomaly_res["count"])
        c3.metric("Subscriptions",   len(sub_res["subscriptions"]))
        c4.metric("Sub. Cost/Month", format_inr(sub_res["total_monthly"]))
    else:
        st.markdown('<div class="insight-card" style="text-align:center;padding:2rem;">🎯 Click the button above to generate your complete financial roadmap</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 9 — Chat
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "💬 Chat":
    st.markdown(
        '<div class="section-title">💬 Chat with Finora <span class="ai-badge">LLM</span></div>',
        unsafe_allow_html=True)
    require_data()

    if not gemini_ready:
        st.warning("⚠️ Add your API key in the sidebar to use the chat.")
        st.stop()

    if not st.session_state.financial_context:
        st.session_state.financial_context = build_financial_context(
            st.session_state.categorized_df, st.session_state.monthly_income)

    # Show RAG status
    rag_ready = _chromadb_available()
    if rag_ready:
        st.markdown(
            '<div class="success-card" style="padding:.6rem 1rem;margin-bottom:.8rem;">'
            '🧠 <strong>Hybrid RAG enabled</strong> — Finora filters by category/month first, '
            'then uses FAISS to find the most relevant transactions for your question.</div>',
            unsafe_allow_html=True
        )
    else:
        st.caption("💡 Install `faiss-cpu` and `sentence-transformers` to enable smarter RAG-based chat.")

    suggestions = [
        "How much did I spend on food?",
        "Compare January vs February",
        "Which month did I spend the most?",
        "How much did I save?",
        "Transport spend in February",
        "Did I overspend on shopping?",
    ]
    st.markdown("**💡 Try asking:**")
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending"] = sug

    st.markdown("---")

    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg["role"]
        text = msg["parts"][0] if msg["parts"] else ""
        if role == "user" and idx == 0 and "User says:" in text:
            text = text.split("User says:")[-1].strip()
        if role == "user":
            st.markdown(f'<div class="chat-label-user">👤 You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-user">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label-bot">🤖 Finora</div>', unsafe_allow_html=True)
            # Use container with custom style so markdown renders bold/italic correctly
            with st.container():
                st.markdown(
                    f'<div class="chat-bot">' + text.replace("**", "__") + '</div>',
                    unsafe_allow_html=True
                )

    user_input = st.chat_input("Ask anything about your finances...")
    if "_pending" in st.session_state:
        user_input = st.session_state.pop("_pending")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            if rag_ready:
                # RAG path — retrieves relevant transactions first
                reply, updated, used_rag = rag_chat(
                    user_message=user_input,
                    chat_history=st.session_state.chat_history,
                    df=st.session_state.categorized_df,
                    monthly_income=st.session_state.monthly_income,
                    financial_context=st.session_state.financial_context,
                )
            else:
                # Fallback — regular context injection
                reply, updated = chat_with_finances(
                    user_message=user_input,
                    chat_history=st.session_state.chat_history,
                    financial_context=st.session_state.financial_context,
                )
                updated_with_rag = updated
            st.session_state.chat_history = updated
        st.rerun()

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 10 — Tax Optimizer
# ─────────────────────────────────────────────────────────────────────────────

elif active_tab == "🧾 Tax Optimizer":
    st.markdown('<div class="section-title">🧾 Tax Regime Optimizer (FY 2024-25)</div>', unsafe_allow_html=True)
    st.markdown("Compare Old vs New regime. Enter your details below.")
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 💼 Income")
        income = st.number_input("Gross Annual Income (₹)", 0, 100_000_000, 800_000, 10_000, format="%d")
        hra    = st.number_input("HRA Exemption (₹)", 0, 5_000_000, 120_000, 5_000, format="%d")
    with col_r:
        st.markdown("#### 🧩 Deductions (Old Regime)")
        d80c = st.number_input("Section 80C (₹) [Max ₹1,50,000]", 0, 150_000, 150_000, 5_000, format="%d")
        d80d = st.number_input("Section 80D (₹) [Max ₹25,000]",   0, 100_000,  15_000, 1_000, format="%d")

    st.markdown("---")
    result = compare_tax_regimes(income=income, hra=hra, deductions_80c=d80c, deductions_80d=d80d)
    old = result["old_regime"]
    new = result["new_regime"]

    col_old, col_new = st.columns(2)
    with col_old:
        st.markdown(f"""<div class="tax-old">
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#f87171;margin-bottom:.75rem;">🏛️ Old Regime</div>
            <div class="tax-row"><span>Gross Income</span><span>{format_inr(old['gross_income'])}</span></div>
            <div class="tax-row"><span>Std. Deduction</span><span>− {format_inr(old['standard_deduction'])}</span></div>
            <div class="tax-row"><span>HRA</span><span>− {format_inr(old['hra_exemption'])}</span></div>
            <div class="tax-row"><span>80C</span><span>− {format_inr(old['deductions_80c'])}</span></div>
            <div class="tax-row"><span>80D</span><span>− {format_inr(old['deductions_80d'])}</span></div>
            <div class="tax-row" style="font-weight:600;"><span>Taxable Income</span><span>{format_inr(old['taxable_income'])}</span></div>
            <div class="tax-row"><span>Tax</span><span>{format_inr(old['tax_before_cess'])}</span></div>
            <div class="tax-row"><span>Cess 4%</span><span>{format_inr(old['cess'])}</span></div>
            <div class="tax-total" style="color:#f87171;">Total: {format_inr(old['total_tax'])}</div>
        </div>""", unsafe_allow_html=True)
    with col_new:
        st.markdown(f"""<div class="tax-new">
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#34d399;margin-bottom:.75rem;">🆕 New Regime</div>
            <div class="tax-row"><span>Gross Income</span><span>{format_inr(new['gross_income'])}</span></div>
            <div class="tax-row"><span>Std. Deduction</span><span>− {format_inr(new['standard_deduction'])}</span></div>
            <div class="tax-row"><span>HRA</span><span>N/A</span></div>
            <div class="tax-row"><span>80C</span><span>N/A</span></div>
            <div class="tax-row"><span>80D</span><span>N/A</span></div>
            <div class="tax-row" style="font-weight:600;"><span>Taxable Income</span><span>{format_inr(new['taxable_income'])}</span></div>
            <div class="tax-row"><span>Tax</span><span>{format_inr(new['tax_before_cess'])}</span></div>
            <div class="tax-row"><span>Cess 4%</span><span>{format_inr(new['cess'])}</span></div>
            <div class="tax-total" style="color:#34d399;">Total: {format_inr(new['total_tax'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="recommend-banner">
        <div class="recommend-text">🏆 Recommended: {result['recommended']}</div>
        <div class="recommend-reason">{result['reason']}</div>
        <div style="margin-top:.75rem;font-family:'Syne',sans-serif;font-size:1rem;color:rgba(255,255,255,0.9);">
            💰 You save <strong>{format_inr(result['savings'])}</strong> per year
        </div>
    </div>""", unsafe_allow_html=True)