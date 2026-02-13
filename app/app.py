"""
PhishGuard-AI — Streamlit Web Arayuzu
======================================
Dark Cybersecurity Theme + Self-Learning AI
"""

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

# ── Sayfa Ayarlari ──
st.set_page_config(
    page_title="PhishGuard-AI | Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  DARK CYBERSECURITY THEME — CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Ana Arka Plan ── */
    .stApp {
        background: linear-gradient(180deg, #0a0e17 0%, #0d1321 50%, #0a0e17 100%);
    }

    /* ── Header ── */
    .cyber-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #00ff4120;
        border-radius: 16px;
        margin-bottom: 2rem;
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
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .cyber-header h1 {
        font-family: 'Courier New', monospace;
        font-size: 2.8rem;
        color: #00ff41;
        text-shadow: 0 0 20px #00ff4140, 0 0 40px #00ff4120;
        margin: 0;
        letter-spacing: 3px;
    }
    .cyber-header .subtitle {
        color: #58a6ff;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
        opacity: 0.8;
    }
    .cyber-header .tagline {
        color: #8b949e;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-family: 'Courier New', monospace;
    }

    /* ── Terminal Input Box ── */
    .terminal-box {
        background: #0d1117;
        border: 1px solid #00ff4130;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .terminal-label {
        color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    /* ── Sonuc Kartlari ── */
    .result-safe {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #0d2818 100%);
        border: 2px solid #00ff41;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 0 20px #00ff4115, inset 0 0 20px #00ff4108;
    }
    .result-danger {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #2d0a0a 100%);
        border: 2px solid #ff4444;
        color: #ff4444;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 0 20px #ff444415, inset 0 0 20px #ff444408;
        animation: pulse-danger 2s ease-in-out infinite;
    }
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 20px #ff444415; }
        50% { box-shadow: 0 0 30px #ff444430; }
    }

    /* ── Bilgi Kartlari ── */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-card h4 {
        color: #58a6ff;
        font-family: 'Courier New', monospace;
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
    }

    /* ── Ogrenme Kutusu ── */
    .learn-box {
        background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 100%);
        border: 1px solid #8b5cf620;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #c9d1d9;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .learn-box strong { color: #8b5cf6; }

    /* ── Match/Mismatch Kutusu ── */
    .match-box {
        background: #0d2818;
        border: 1px solid #00ff4130;
        border-radius: 10px;
        padding: 1rem;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    .mismatch-box {
        background: #2d1a00;
        border: 1px solid #ff880030;
        border-radius: 10px;
        padding: 1rem;
        color: #ff8800;
        font-family: 'Courier New', monospace;
        text-align: center;
    }

    /* ── Stat Card ── */
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-card .stat-value {
        color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .stat-card .stat-label {
        color: #8b949e;
        font-size: 0.75rem;
        font-family: 'Courier New', monospace;
    }

    /* ── Feature Grid ── */
    .feat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0.5rem;
        border-bottom: 1px solid #21262d;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .feat-name { color: #8b949e; }
    .feat-val { color: #58a6ff; font-weight: bold; }
    .feat-danger { color: #ff4444; font-weight: bold; }
    .feat-warn { color: #ff8800; font-weight: bold; }
    .feat-safe { color: #00ff41; font-weight: bold; }

    /* ── Footer ── */
    .cyber-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #21262d;
        color: #484f58;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    .cyber-footer .glow { color: #00ff41; text-shadow: 0 0 10px #00ff4140; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }

    /* ── Genel Metin ── */
    .stMarkdown { color: #c9d1d9; }
    .stMarkdown h3 {
        color: #58a6ff;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="cyber-header">
    <h1>[PHISHGUARD-AI]</h1>
    <div class="subtitle">&gt; Self-Learning Phishing URL Detection System_</div>
    <div class="tagline">// AI-powered threat analysis with real-time verification</div>
</div>
""", unsafe_allow_html=True)


def load_model(model_path=None):
    """Modeli ve scaler'i yukler."""
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
    api_key = st.text_input("Safe Browsing API Key:",
                            type="password",
                            help="Opsiyonel. Yoksa heuristic analiz kullanilir.")
    if not api_key:
        st.caption("_// heuristic mode active_")

    # ── Ogrenme Istatistikleri ──
    st.markdown("---")
    st.markdown("## > AI_LEARNING_STATUS")
    stats = get_feedback_stats()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-label">TOTAL SCANS</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:#00ff41">{stats['correct']}</div>
            <div class="stat-label">CORRECT</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        acc_str = f"{stats['accuracy']:.0%}" if stats['total'] > 0 else "--"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:#58a6ff">{acc_str}</div>
            <div class="stat-label">ACCURACY</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:#ff4444">{stats['incorrect']}</div>
            <div class="stat-label">ERRORS</div>
        </div>""", unsafe_allow_html=True)

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

    # ── Ornek URL'ler ──
    st.markdown("---")
    st.markdown("## > TEST_URLS")
    examples = {
        "[SAFE] google.com": "https://www.google.com/search?q=python",
        "[SAFE] youtube.com": "https://www.youtube.com/watch?v=abc123",
        "[THREAT] IP-based": "http://192.168.1.1/admin/login.php",
        "[THREAT] fake-tld": "http://paypal-secure-login.tk/verify",
        "[THREAT] shortener": "https://bit.ly/3xYz123",
        "[THREAT] @-attack": "http://google.com@malicious.tk/steal",
    }
    for label, url in examples.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state['example_url'] = url


# ══════════════════════════════════════════════════════════
#  ANA ICERIK
# ══════════════════════════════════════════════════════════

# Input alani
st.markdown('<div class="terminal-label">&gt; TARGET_URL:</div>', unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
with col_input:
    default_url = st.session_state.get('example_url', '')
    url_input = st.text_input("URL", value=default_url,
                              placeholder="http://suspicious-link.tk/login",
                              label_visibility="collapsed")
    if 'example_url' in st.session_state:
        del st.session_state['example_url']
with col_btn:
    analyze_btn = st.button("SCAN", type="primary", use_container_width=True)


# ── Analiz ──
if analyze_btn and url_input and selected_model:
    with st.spinner("Scanning target..."):
        try:
            # 1. Model tahmini
            model_path = os.path.join(models_dir, selected_model)
            model, scaler = load_model(model_path)
            features = extract_features(url_input)
            X = np.array([[features[f] for f in FEATURE_NAMES]])
            X_scaled = scaler.transform(X) if scaler else X
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = max(proba) * 100

            # 2. API / Heuristic dogrulama
            api_result = check_url_safe_browsing(url_input, api_key or None)

            # 3. Feedback kaydet
            is_correct, api_label = log_feedback(
                url_input, prediction, confidence / 100,
                api_result, features
            )

            # ── SONUCLAR ──
            st.markdown("---")

            col_model, col_api = st.columns(2)

            with col_model:
                st.markdown("### > MODEL_PREDICTION")
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-danger">
                        &#9888; THREAT DETECTED<br>
                        <span style="font-size:0.9rem">Classification: PHISHING</span><br>
                        <span style="font-size:1.5rem">{confidence:.1f}%</span>
                        <span style="font-size:0.8rem">confidence</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        &#10004; SECURE URL<br>
                        <span style="font-size:0.9rem">Classification: LEGITIMATE</span><br>
                        <span style="font-size:1.5rem">{confidence:.1f}%</span>
                        <span style="font-size:0.8rem">confidence</span>
                    </div>""", unsafe_allow_html=True)

            with col_api:
                source = api_result['source'].split('(')[0].strip()
                st.markdown(f"### > VERIFICATION [{source}]")
                if api_result['is_malicious']:
                    st.markdown(f"""
                    <div class="result-danger">
                        &#9888; MALICIOUS<br>
                        <span style="font-size:0.8rem">{api_result['details']}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        &#10004; CLEAN<br>
                        <span style="font-size:0.8rem">{api_result['details']}</span>
                    </div>""", unsafe_allow_html=True)

            # Uyum durumu
            st.markdown("")
            if is_correct:
                st.markdown("""
                <div class="match-box">
                    [MATCH] Model prediction confirmed by verification source.
                    Result logged for continuous learning.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="mismatch-box">
                    [MISMATCH] Model disagrees with verification source!
                    This sample has been flagged for retraining.
                    The AI will LEARN from this error.
                </div>""", unsafe_allow_html=True)

            # Ogrenme durumu
            updated_stats = get_feedback_stats()
            remaining = updated_stats['needed_for_retrain']
            st.markdown(f"""
            <div class="learn-box">
                <strong>&gt; AI_LEARNING_CYCLE</strong><br><br>
                Total scans: <strong>{updated_stats['total']}</strong> |
                Accuracy: <strong>{updated_stats['accuracy']:.1%}</strong> |
                Errors: <strong>{updated_stats['incorrect']}</strong><br><br>
                {f'Retrain in <strong>{remaining}</strong> more scans...' if remaining > 0
                 else '<strong style="color:#00ff41">READY TO RETRAIN! Check sidebar.</strong>'}<br>
                <br>
                <em>// Every scan feeds the AI learning pipeline.</em><br>
                <em>// Model auto-improves after 50 verified samples.</em>
            </div>
            """, unsafe_allow_html=True)

            # Feature detaylari
            with st.expander(">> EXTRACTED_FEATURES [16 vectors]"):
                for k, v in features.items():
                    is_danger = (k in ['has_ip', 'has_at_sign', 'has_double_slash',
                                       'has_shortener', 'suspicious_tld'] and v == 1)
                    is_warn = ((k == 'has_dash' and v == 1) or
                               (k == 'has_https' and v == 0))
                    if is_danger:
                        css = "feat-danger"
                        icon = "[!]"
                    elif is_warn:
                        css = "feat-warn"
                        icon = "[~]"
                    else:
                        css = "feat-safe"
                        icon = "[+]"
                    st.markdown(f"""
                    <div class="feat-row">
                        <span class="feat-name">{icon} {k}</span>
                        <span class="{css}">{v}</span>
                    </div>""", unsafe_allow_html=True)

            # Otomatik retrain
            if updated_stats['ready_to_retrain']:
                st.info("Model ready for retraining. Use sidebar button.")

        except Exception as e:
            st.error(f"ERROR: {str(e)}")

elif analyze_btn and not url_input:
    st.warning("Enter a target URL to scan.")
elif analyze_btn and not selected_model:
    st.warning("Select a model from sidebar.")

# ── Footer ──
st.markdown("""
<div class="cyber-footer">
    <span class="glow">[PHISHGUARD-AI]</span> v2.0 — Self-Learning Cyber Threat Detection<br>
    // Powered by Machine Learning | Active Learning Pipeline<br>
    // Every scan makes the AI smarter_
</div>
""", unsafe_allow_html=True)
