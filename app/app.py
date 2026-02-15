import os
import sys
import streamlit as st
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features, FEATURE_NAMES
from src.feedback_loop import (
    check_url_safe_browsing, log_feedback,
    get_feedback_stats, auto_retrain_if_ready
)

st.set_page_config(
    page_title="PhishGuard-AI | Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def ensure_models_exist():
    import io, contextlib
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    pkl_files = [f for f in os.listdir(models_dir)
                 if f.endswith('.pkl') and f != 'scaler.pkl']
    if not pkl_files:
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        from src.generate_dataset import generate_dataset, save_dataset
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, 'phishing_urls.csv')
        if not os.path.exists(csv_path):
            data = generate_dataset()
            save_dataset(data, csv_path)
        from src.train import main as train_main
        with contextlib.redirect_stdout(io.StringIO()):
            train_main()
    return True

ensure_models_exist()


# ══════════════════════════════════════════════════════════
#  CSS — Dashboard Cards + Tabs + JetBrains Mono
# ══════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Global Font ── */
    *, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stTextInput input, .stSelectbox, button, .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Arka Plan ── */
    .stApp {
        background: linear-gradient(180deg, #0a0e17 0%, #0d1321 50%, #0a0e17 100%);
    }

    /* ══════ ANIMATIONS ══════ */

    /* Staggered card fade-in */
    @keyframes cardFadeIn {
        0% { opacity: 0; transform: translateY(20px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Glow pulse for stat values */
    @keyframes glowPulse {
        0%, 100% { text-shadow: 0 0 5px currentColor; filter: brightness(1); }
        50% { text-shadow: 0 0 20px currentColor, 0 0 40px currentColor; filter: brightness(1.2); }
    }

    /* Scan line across header */
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Blinking cursor */
    @keyframes blink {
        50% { opacity: 0; }
    }

    /* Danger pulse */
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 20px #ff444415; border-color: #ff4444; }
        50% { box-shadow: 0 0 40px #ff444435; border-color: #ff6666; }
    }

    /* Safe glow */
    @keyframes pulse-safe {
        0%, 100% { box-shadow: 0 0 20px #00ff4110; }
        50% { box-shadow: 0 0 35px #00ff4125; }
    }

    /* Result slide in */
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(30px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    /* Verdict pop */
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.8); }
        60% { transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Feature row cascade */
    @keyframes rowReveal {
        0% { opacity: 0; transform: translateX(-10px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    /* Floating particles in header */
    @keyframes float1 {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
        25% { transform: translate(50px, -20px) scale(1.2); opacity: 0.6; }
        50% { transform: translate(100px, 10px) scale(0.8); opacity: 0.4; }
        75% { transform: translate(30px, -10px) scale(1.1); opacity: 0.5; }
    }
    @keyframes float2 {
        0%, 100% { transform: translate(0, 0) scale(0.8); opacity: 0.2; }
        33% { transform: translate(-40px, -30px) scale(1); opacity: 0.5; }
        66% { transform: translate(-80px, 15px) scale(1.3); opacity: 0.3; }
    }

    /* Progress bar shimmer */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ── Header ── */
    .cyber-header {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #00ff4120;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .cyber-header::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 200%; height: 2px;
        background: linear-gradient(90deg, transparent, #00ff41, transparent);
        animation: scan 3s linear infinite;
    }
    .cyber-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: -100%;
        width: 200%; height: 1px;
        background: linear-gradient(90deg, transparent, #58a6ff80, transparent);
        animation: scan 4s linear infinite reverse;
    }
    /* Floating particles */
    .cyber-header .particle-1, .cyber-header .particle-2, .cyber-header .particle-3 {
        position: absolute;
        border-radius: 50%;
        pointer-events: none;
    }
    .cyber-header .particle-1 {
        width: 4px; height: 4px; background: #00ff41;
        top: 30%; left: 15%;
        animation: float1 8s ease-in-out infinite;
    }
    .cyber-header .particle-2 {
        width: 3px; height: 3px; background: #58a6ff;
        top: 50%; right: 20%;
        animation: float2 10s ease-in-out infinite;
    }
    .cyber-header .particle-3 {
        width: 5px; height: 5px; background: #8b5cf6;
        bottom: 25%; left: 60%;
        animation: float1 12s ease-in-out infinite reverse;
    }
    .cyber-header h1 {
        font-size: 2.5rem;
        color: #00ff41;
        text-shadow: 0 0 20px #00ff4140, 0 0 40px #00ff4120;
        margin: 0;
        letter-spacing: 3px;
        font-weight: 800;
        animation: glowPulse 4s ease-in-out infinite;
    }
    .cyber-header .subtitle {
        color: #58a6ff;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        opacity: 0.85;
        font-weight: 400;
    }

    /* ── Dashboard Stat Cards ── */
    .dash-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .dash-card {
        background: linear-gradient(145deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        opacity: 0;
        animation: cardFadeIn 0.6s ease forwards;
    }
    .dash-card:nth-child(1) { animation-delay: 0.1s; }
    .dash-card:nth-child(2) { animation-delay: 0.25s; }
    .dash-card:nth-child(3) { animation-delay: 0.4s; }
    .dash-card:nth-child(4) { animation-delay: 0.55s; }
    .dash-card:hover {
        border-color: #00ff4140;
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 255, 65, 0.1);
    }
    .dash-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0;
        width: 100%; height: 3px;
    }
    .dash-card:nth-child(1)::after { background: linear-gradient(90deg, #00ff41, #00ff4100); }
    .dash-card:nth-child(2)::after { background: linear-gradient(90deg, #58a6ff, #58a6ff00); }
    .dash-card:nth-child(3)::after { background: linear-gradient(90deg, #ff4444, #ff444400); }
    .dash-card:nth-child(4)::after { background: linear-gradient(90deg, #8b5cf6, #8b5cf600); }
    .dash-card .dash-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .dash-card .dash-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        animation: glowPulse 3s ease-in-out infinite;
    }
    .dash-card:nth-child(2) .dash-value { animation-delay: 0.5s; }
    .dash-card:nth-child(3) .dash-value { animation-delay: 1s; }
    .dash-card:nth-child(4) .dash-value { animation-delay: 1.5s; }
    .dash-card .dash-label {
        color: #8b949e;
        font-size: 0.7rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── URL Input Section ── */
    .url-section {
        background: linear-gradient(145deg, #0d1117 0%, #131922 100%);
        border: 1px solid #00ff4125;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        animation: cardFadeIn 0.6s ease forwards;
        animation-delay: 0.7s;
        opacity: 0;
    }
    .url-label {
        color: #00ff41;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
        letter-spacing: 1px;
    }
    .url-label .blink {
        animation: blink 1s step-end infinite;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0d1117;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        font-size: 0.82rem;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c9d1d9;
        background: #161b2280;
    }
    .stTabs [aria-selected="true"] {
        background: #161b22 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff4130 !important;
        box-shadow: 0 0 10px #00ff4110;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }

    /* ── Result Cards ── */
    .result-safe {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #0d2818 100%);
        border: 2px solid #00ff41;
        color: #00ff41;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 0 30px #00ff4110, inset 0 0 30px #00ff4108;
        animation: slideInLeft 0.5s ease, pulse-safe 3s ease-in-out infinite 0.5s;
    }
    .result-danger {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #2d0a0a 100%);
        border: 2px solid #ff4444;
        color: #ff4444;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 0 30px #ff444410, inset 0 0 30px #ff444408;
        animation: slideInLeft 0.5s ease, pulse-danger 2s ease-in-out infinite 0.5s;
    }
    .result-safe .result-label, .result-danger .result-label {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.2rem;
    }
    .result-safe .result-conf, .result-danger .result-conf {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    /* ── Feature Table ── */
    .feat-table {
        width: 100%;
        border-collapse: collapse;
    }
    .feat-table tr {
        border-bottom: 1px solid #21262d;
        transition: all 0.2s ease;
        animation: rowReveal 0.4s ease forwards;
        opacity: 0;
    }
    .feat-table tr:nth-child(1)  { animation-delay: 0s; }
    .feat-table tr:nth-child(2)  { animation-delay: 0.03s; }
    .feat-table tr:nth-child(3)  { animation-delay: 0.06s; }
    .feat-table tr:nth-child(4)  { animation-delay: 0.09s; }
    .feat-table tr:nth-child(5)  { animation-delay: 0.12s; }
    .feat-table tr:nth-child(6)  { animation-delay: 0.15s; }
    .feat-table tr:nth-child(7)  { animation-delay: 0.18s; }
    .feat-table tr:nth-child(8)  { animation-delay: 0.21s; }
    .feat-table tr:nth-child(9)  { animation-delay: 0.24s; }
    .feat-table tr:nth-child(10) { animation-delay: 0.27s; }
    .feat-table tr:nth-child(11) { animation-delay: 0.30s; }
    .feat-table tr:nth-child(12) { animation-delay: 0.33s; }
    .feat-table tr:nth-child(13) { animation-delay: 0.36s; }
    .feat-table tr:nth-child(14) { animation-delay: 0.39s; }
    .feat-table tr:nth-child(15) { animation-delay: 0.42s; }
    .feat-table tr:nth-child(16) { animation-delay: 0.45s; }
    .feat-table tr:nth-child(17) { animation-delay: 0.48s; }
    .feat-table tr:hover {
        background: #161b2280;
        transform: translateX(4px);
    }
    .feat-table td {
        padding: 0.5rem 0.8rem;
        font-size: 0.82rem;
    }
    .feat-table .fname { color: #8b949e; }
    .feat-table .fval { text-align: right; font-weight: 600; }
    .feat-table .fval.danger { color: #ff4444; }
    .feat-table .fval.warn { color: #ff8800; }
    .feat-table .fval.safe { color: #00ff41; }
    .feat-table .fval.neutral { color: #58a6ff; }
    .feat-table .ftag {
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .ftag.tag-danger { background: #ff444420; color: #ff4444; }
    .ftag.tag-warn { background: #ff880020; color: #ff8800; }
    .ftag.tag-ok { background: #00ff4115; color: #00ff41; }

    /* ── API Card ── */
    .api-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        animation: slideInRight 0.5s ease;
    }
    .api-card .api-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #21262d;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    .api-card .api-row:hover {
        background: #1c233380;
        padding-left: 0.3rem;
    }
    .api-card .api-key { color: #8b949e; }
    .api-card .api-val { color: #58a6ff; font-weight: 500; }

    /* ── Match/Mismatch ── */
    .verdict-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        font-size: 0.85rem;
        margin-top: 1rem;
        animation: popIn 0.4s ease forwards;
    }
    .verdict-match {
        background: #0d2818;
        border: 1px solid #00ff4130;
        color: #00ff41;
    }
    .verdict-mismatch {
        background: #2d1a00;
        border: 1px solid #ff880030;
        color: #ff8800;
    }

    /* ── Learning Progress ── */
    .learn-progress {
        background: linear-gradient(135deg, #0d1117, #1a0a2e);
        border: 1px solid #8b5cf620;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #c9d1d9;
        animation: cardFadeIn 0.6s ease forwards;
        animation-delay: 0.3s;
        opacity: 0;
    }
    .learn-progress strong { color: #8b5cf6; }
    .learn-bar-bg {
        width: 100%;
        height: 6px;
        background: #21262d;
        border-radius: 3px;
        margin: 0.8rem 0 0.5rem;
        overflow: hidden;
    }
    .learn-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #8b5cf6, #00ff41, #8b5cf6);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
        transition: width 0.8s ease;
    }

    /* ── History ── */
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #21262d;
        font-size: 0.8rem;
        transition: all 0.2s ease;
        animation: rowReveal 0.3s ease forwards;
        opacity: 0;
    }
    .history-item:nth-child(1)  { animation-delay: 0.05s; }
    .history-item:nth-child(2)  { animation-delay: 0.1s; }
    .history-item:nth-child(3)  { animation-delay: 0.15s; }
    .history-item:nth-child(4)  { animation-delay: 0.2s; }
    .history-item:nth-child(5)  { animation-delay: 0.25s; }
    .history-item:nth-child(6)  { animation-delay: 0.3s; }
    .history-item:nth-child(7)  { animation-delay: 0.35s; }
    .history-item:nth-child(8)  { animation-delay: 0.4s; }
    .history-item:nth-child(9)  { animation-delay: 0.45s; }
    .history-item:nth-child(10) { animation-delay: 0.5s; }
    .history-item:hover {
        background: #161b2280;
        transform: translateX(4px);
    }
    .history-url { color: #58a6ff; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-safe { background: #00ff4120; color: #00ff41; }
    .badge-danger { background: #ff444420; color: #ff4444; }

    /* ── Footer ── */
    .cyber-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #21262d;
        color: #484f58;
        font-size: 0.75rem;
    }
    .cyber-footer .glow {
        color: #00ff41;
        text-shadow: 0 0 10px #00ff4140;
        animation: glowPulse 3s ease-in-out infinite;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #00ff41;
        font-size: 0.9rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }
    .stMarkdown { color: #c9d1d9; }
    .stMarkdown h3 { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="cyber-header">
    <div class="particle-1"></div>
    <div class="particle-2"></div>
    <div class="particle-3"></div>
    <h1>⛨ PHISHGUARD-AI</h1>
    <div class="subtitle">&gt; Self-Learning Phishing URL Detection System_</div>
</div>
""", unsafe_allow_html=True)


def load_model(model_path=None):
    import joblib
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    if model_path is None:
        model_path = os.path.join(models_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        for f in os.listdir(models_dir):
            if f.endswith('.pkl') and f != 'scaler.pkl':
                model_path = os.path.join(models_dir, f)
                break
    model = joblib.load(model_path)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## > MODEL_CONFIG")
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir)
                       if f.endswith('.pkl') and f != 'scaler.pkl']
    selected_model = st.selectbox("Classifier:", model_files,
                                  index=0 if model_files else None)

    st.markdown("---")
    st.markdown("## > API_CONFIG")
    api_key = st.text_input("Safe Browsing API Key:", type="password",
                            help="Opsiyonel. Yoksa heuristic analiz kullanilir.")
    if not api_key:
        st.caption("_// heuristic mode active_")

    st.markdown("---")
    st.markdown("## > TEST_URLS")
    examples = {
        "✅ google.com": "https://www.google.com/search?q=python",
        "✅ youtube.com": "https://www.youtube.com/watch?v=abc123",
        "⚠️ IP-based": "http://192.168.1.1/admin/login.php",
        "⚠️ fake-tld": "http://paypal-secure-login.tk/verify",
        "⚠️ shortener": "https://bit.ly/3xYz123",
        "⚠️ @-attack": "http://google.com@malicious.tk/steal",
    }
    for label, url in examples.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state['example_url'] = url

    st.markdown("---")
    st.markdown("## > AI_LEARNING")
    stats = get_feedback_stats()
    if stats['total'] > 0:
        progress = min(stats['total'] / 50, 1.0)
        st.progress(progress, text=f"Retrain: {stats['total']}/50")
        if stats['ready_to_retrain']:
            st.success("Ready to retrain!")
            if st.button(">> RETRAIN MODEL <<", type="primary"):
                with st.spinner("Retraining..."):
                    result = auto_retrain_if_ready(force=True)
                    if result['retrained']:
                        st.success(f"{result['old_accuracy']:.1%} -> {result['new_accuracy']:.1%}")
                        st.balloons()


# ══════════════════════════════════════════════════════════
#  DASHBOARD STAT CARDS
# ══════════════════════════════════════════════════════════
stats = get_feedback_stats()
st.markdown(f"""
<div class="dash-grid">
    <div class="dash-card">
        <div class="dash-icon">📡</div>
        <div class="dash-value" style="color:#00ff41">{stats['total']}</div>
        <div class="dash-label">Total Scans</div>
    </div>
    <div class="dash-card">
        <div class="dash-icon">✅</div>
        <div class="dash-value" style="color:#58a6ff">{stats['correct']}</div>
        <div class="dash-label">Correct</div>
    </div>
    <div class="dash-card">
        <div class="dash-icon">❌</div>
        <div class="dash-value" style="color:#ff4444">{stats['incorrect']}</div>
        <div class="dash-label">Errors</div>
    </div>
    <div class="dash-card">
        <div class="dash-icon">🎯</div>
        <div class="dash-value" style="color:#8b5cf6">{f"{stats['accuracy']:.0%}" if stats['total'] > 0 else "--"}</div>
        <div class="dash-label">Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  URL INPUT
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="url-section">
    <div class="url-label">&gt; TARGET_URL<span class="blink">_</span></div>
</div>
""", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    default_url = st.session_state.get('example_url', '')
    url_input = st.text_input("URL", value=default_url,
                              placeholder="http://suspicious-link.tk/login",
                              label_visibility="collapsed")
    if 'example_url' in st.session_state:
        del st.session_state['example_url']
with col_btn:
    analyze_btn = st.button("⚡ SCAN", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════
#  ANALYSIS & TABBED RESULTS
# ══════════════════════════════════════════════════════════
if analyze_btn and url_input and selected_model:
    with st.spinner("Scanning target..."):
        try:
            model_path = os.path.join(models_dir, selected_model)
            model, scaler = load_model(model_path)
            features = extract_features(url_input)
            X = np.array([[features[f] for f in FEATURE_NAMES]])
            X_scaled = scaler.transform(X) if scaler else X
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = max(proba) * 100

            api_result = check_url_safe_browsing(url_input, api_key or None)
            is_correct, api_label = log_feedback(
                url_input, prediction, confidence / 100, api_result, features
            )

            # Save to history
            if 'scan_history' not in st.session_state:
                st.session_state['scan_history'] = []
            st.session_state['scan_history'].insert(0, {
                'url': url_input,
                'prediction': prediction,
                'confidence': confidence,
                'is_correct': is_correct,
            })
            if len(st.session_state['scan_history']) > 20:
                st.session_state['scan_history'] = st.session_state['scan_history'][:20]

            # ── TABS ──
            tab_result, tab_features, tab_api, tab_history = st.tabs([
                "🔍 Sonuç", "📊 Feature Analizi", "🛡️ API Doğrulama", "📜 Geçmiş"
            ])

            # ── TAB 1: SONUÇ ──
            with tab_result:
                col_model, col_verify = st.columns(2)

                with col_model:
                    st.markdown("### > MODEL_PREDICTION")
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-danger">
                            &#9888; THREAT DETECTED
                            <div class="result-label">Classification: PHISHING</div>
                            <div class="result-conf">{confidence:.1f}%</div>
                            <div class="result-label">confidence</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            &#10004; SECURE URL
                            <div class="result-label">Classification: LEGITIMATE</div>
                            <div class="result-conf">{confidence:.1f}%</div>
                            <div class="result-label">confidence</div>
                        </div>""", unsafe_allow_html=True)

                with col_verify:
                    source = api_result['source'].split('(')[0].strip()
                    st.markdown(f"### > VERIFICATION [{source}]")
                    if api_result['is_malicious']:
                        st.markdown(f"""
                        <div class="result-danger">
                            &#9888; MALICIOUS
                            <div class="result-label">{api_result['details']}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            &#10004; CLEAN
                            <div class="result-label">{api_result['details']}</div>
                        </div>""", unsafe_allow_html=True)

                # Match/Mismatch
                if is_correct:
                    st.markdown("""
                    <div class="verdict-box verdict-match">
                        ✅ [MATCH] Model prediction confirmed by verification source.
                        Result logged for continuous learning.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="verdict-box verdict-mismatch">
                        ⚠️ [MISMATCH] Model disagrees with verification source!
                        This sample has been flagged for retraining. The AI will learn from this error.
                    </div>""", unsafe_allow_html=True)

                # Learning Progress
                updated_stats = get_feedback_stats()
                remaining = updated_stats['needed_for_retrain']
                fill_pct = min(updated_stats['total'] / 50 * 100, 100)
                st.markdown(f"""
                <div class="learn-progress">
                    <strong>&gt; AI_LEARNING_CYCLE</strong><br><br>
                    Total: <strong>{updated_stats['total']}</strong> &nbsp;|&nbsp;
                    Accuracy: <strong>{updated_stats['accuracy']:.1%}</strong> &nbsp;|&nbsp;
                    Errors: <strong>{updated_stats['incorrect']}</strong>
                    <div class="learn-bar-bg">
                        <div class="learn-bar-fill" style="width:{fill_pct}%"></div>
                    </div>
                    {f'Retrain in <strong>{remaining}</strong> more scans...' if remaining > 0
                     else '<strong style="color:#00ff41">READY TO RETRAIN! Check sidebar →</strong>'}
                </div>
                """, unsafe_allow_html=True)

            # ── TAB 2: FEATURE ANALİZİ ──
            with tab_features:
                st.markdown("### > EXTRACTED_FEATURES [16 vectors]")

                danger_keys = ['has_ip', 'has_at_sign', 'has_double_slash',
                               'has_shortener', 'suspicious_tld']
                warn_keys = ['has_dash']
                no_https_warn = True

                rows_html = ""
                for k, v in features.items():
                    if k in danger_keys and v == 1:
                        css = "danger"
                        tag = '<span class="ftag tag-danger">RISK</span>'
                    elif (k in warn_keys and v == 1) or (k == 'has_https' and v == 0):
                        css = "warn"
                        tag = '<span class="ftag tag-warn">WARN</span>'
                    elif k in danger_keys and v == 0:
                        css = "safe"
                        tag = '<span class="ftag tag-ok">OK</span>'
                    else:
                        css = "neutral"
                        tag = ""

                    rows_html += f"""
                    <tr>
                        <td class="fname">{k}</td>
                        <td class="fval {css}">{v}</td>
                        <td>{tag}</td>
                    </tr>"""

                st.markdown(f"""
                <table class="feat-table">
                    <tr style="border-bottom:2px solid #30363d;">
                        <td class="fname" style="font-weight:600;color:#58a6ff">Feature</td>
                        <td class="fval" style="color:#58a6ff">Value</td>
                        <td style="color:#58a6ff;font-size:0.75rem">Status</td>
                    </tr>
                    {rows_html}
                </table>
                """, unsafe_allow_html=True)

            # ── TAB 3: API DOĞRULAMA ──
            with tab_api:
                st.markdown("### > VERIFICATION_DETAILS")

                api_rows = {
                    "Source": api_result['source'].split('(')[0].strip(),
                    "Malicious": "⚠️ YES" if api_result['is_malicious'] else "✅ NO",
                    "Confidence": f"{api_result.get('confidence', 0):.0%}",
                    "Threat Type": api_result.get('threat_type', 'N/A'),
                    "Details": api_result.get('details', 'N/A'),
                }

                api_html = ""
                for k, v in api_rows.items():
                    api_html += f"""
                    <div class="api-row">
                        <span class="api-key">{k}</span>
                        <span class="api-val">{v}</span>
                    </div>"""

                st.markdown(f'<div class="api-card">{api_html}</div>',
                            unsafe_allow_html=True)

                if not api_key:
                    st.markdown("""
                    <div style="margin-top:1rem;padding:0.8rem;background:#161b22;
                         border:1px solid #30363d;border-radius:8px;font-size:0.8rem;
                         color:#8b949e">
                        ℹ️ Heuristic mod aktif. Google Safe Browsing API key ekleyerek
                        daha güvenilir sonuçlar alabilirsiniz. (Sidebar → API_CONFIG)
                    </div>""", unsafe_allow_html=True)

            # ── TAB 4: GEÇMİŞ ──
            with tab_history:
                st.markdown("### > SCAN_HISTORY")
                history = st.session_state.get('scan_history', [])
                if not history:
                    st.markdown("""
                    <div style="text-align:center;color:#484f58;padding:2rem;
                         font-size:0.85rem">
                        // No scans yet. Analyze a URL to build history.
                    </div>""", unsafe_allow_html=True)
                else:
                    items_html = ""
                    for item in history:
                        badge_css = "badge-danger" if item['prediction'] == 1 else "badge-safe"
                        badge_text = "PHISHING" if item['prediction'] == 1 else "SAFE"
                        match_icon = "✅" if item['is_correct'] else "⚠️"
                        items_html += f"""
                        <div class="history-item">
                            <span class="history-url">{item['url']}</span>
                            <span>
                                {match_icon}
                                <span class="history-badge {badge_css}">{badge_text}</span>
                                <span style="color:#8b949e;font-size:0.7rem;margin-left:4px">
                                    {item['confidence']:.0f}%
                                </span>
                            </span>
                        </div>"""
                    st.markdown(items_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ERROR: {str(e)}")

elif analyze_btn and not url_input:
    st.warning("Enter a target URL to scan.")
elif analyze_btn and not selected_model:
    st.warning("Select a model from sidebar.")


# ── Footer ──
st.markdown("""
<div class="cyber-footer">
    <span class="glow">[PHISHGUARD-AI]</span> v3.0 — Dashboard Edition<br>
    // Self-Learning Cyber Threat Detection | JetBrains Mono_
</div>
""", unsafe_allow_html=True)
