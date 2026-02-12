"""
PhishGuard-AI — Streamlit Web Arayuzu (Self-Learning)
======================================================
Phishing URL tespiti + otomatik ogrenme dongusu.
"""

import os
import sys
import streamlit as st
import numpy as np

# Proje kokunu path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features, FEATURE_NAMES
from src.feedback_loop import (
    check_url_safe_browsing, log_feedback,
    get_feedback_stats, auto_retrain_if_ready
)

# ── Sayfa Ayarlari ──
st.set_page_config(
    page_title="PhishGuard-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Stilleri ──
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.8; margin: 0.5rem 0 0 0; }
    .result-safe {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white; font-size: 1.3rem; font-weight: bold;
    }
    .result-danger {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        background: linear-gradient(135deg, #e53935, #ff6f00);
        color: white; font-size: 1.3rem; font-weight: bold;
    }
    .feedback-box {
        padding: 1rem; border-radius: 10px;
        background: linear-gradient(135deg, #1e3a5f, #2d5986);
        color: white; margin-top: 1rem;
    }
    .learn-box {
        padding: 1rem; border-radius: 10px;
        background: linear-gradient(135deg, #4a148c, #7b1fa2);
        color: white; margin-top: 1rem;
    }
    .stat-card {
        padding: 1rem; border-radius: 10px; text-align: center;
        background: #1a1a2e; color: white; margin: 0.5rem 0;
    }
    .stat-card h3 { margin: 0; font-size: 2rem; }
    .stat-card p { margin: 0; opacity: 0.7; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🛡️ PhishGuard-AI</h1>
    <p>Self-Learning Phishing URL Detection — Kendi Kendine Ogrenen Yapay Zeka</p>
</div>
""", unsafe_allow_html=True)


def load_model(model_path=None):
    """Modeli ve scaler'i yukler."""
    import joblib
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    if model_path is None:
        model_path = os.path.join(models_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        # Herhangi bir pkl dosyasi bul
        for f in os.listdir(models_dir):
            if f.endswith('.pkl') and f != 'scaler.pkl':
                model_path = os.path.join(models_dir, f)
                break
    model = joblib.load(model_path)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")

    # Model secimi
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir)
                       if f.endswith('.pkl') and f != 'scaler.pkl']

    selected_model = st.selectbox("Model Sec", model_files,
                                  index=0 if model_files else None)

    # API key (opsiyonel)
    st.markdown("---")
    st.markdown("## 🔑 API Ayarlari")
    api_key = st.text_input("Google Safe Browsing API Key (opsiyonel)",
                            type="password",
                            help="API key olmadan da calisir (heuristic mod)")
    if not api_key:
        st.info("API key girilmedi. Heuristic dogrulama kullanilacak.")

    # Feedback istatistikleri
    st.markdown("---")
    st.markdown("## 🧠 Ogrenme Durumu")
    stats = get_feedback_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Analiz", stats['total'])
        st.metric("Dogrusu", stats['correct'])
    with col2:
        st.metric("Model Dogrulugu",
                   f"{stats['accuracy']:.0%}" if stats['total'] > 0 else "-%")
        st.metric("Hatali", stats['incorrect'])

    # Yeniden egitim durumu
    if stats['total'] > 0:
        progress = min(stats['total'] / 50, 1.0)
        st.progress(progress, text=f"Yeniden egitim: {stats['total']}/50")

        if stats['ready_to_retrain']:
            st.success("Yeterli veri birikti!")
            if st.button("🔄 Modeli Yeniden Egit", type="primary"):
                with st.spinner("Model yeniden egitiliyor..."):
                    result = auto_retrain_if_ready(force=True)
                    if result['retrained']:
                        st.success(f"Dogruluk: {result['old_accuracy']:.2%} → {result['new_accuracy']:.2%}")
                        st.balloons()
                    else:
                        st.warning(result['message'])

    # Ornek URL'ler
    st.markdown("---")
    st.markdown("## 🔗 Ornek URL'ler")
    examples = {
        "Google (Guvenli)": "https://www.google.com/search?q=python",
        "YouTube (Guvenli)": "https://www.youtube.com/watch?v=abc123",
        "Phishing (IP)": "http://192.168.1.1/admin/login.php",
        "Phishing (TLD)": "http://paypal-secure-login.tk/verify",
        "Phishing (Kisaltma)": "https://bit.ly/3xYz123",
        "Phishing (@)": "http://google.com@malicious.tk/steal",
    }
    for label, url in examples.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state['example_url'] = url


# ── Ana Icerik ──
col_input, col_btn = st.columns([4, 1])
with col_input:
    default_url = st.session_state.get('example_url', '')
    url_input = st.text_input("🔗 URL girin:", value=default_url,
                              placeholder="https://example.com/login")
    if 'example_url' in st.session_state:
        del st.session_state['example_url']

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)


if analyze_btn and url_input and selected_model:
    with st.spinner("Analiz ediliyor..."):
        try:
            # 1. Model tahmini
            model_path = os.path.join(models_dir, selected_model)
            model, scaler = load_model(model_path)
            features = extract_features(url_input)
            X = np.array([[features[f] for f in FEATURE_NAMES]])
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
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

            # ── Sonuclari Goster ──
            st.markdown("---")

            # Model Sonucu
            col_model, col_api = st.columns(2)

            with col_model:
                st.markdown("### 🤖 Model Tahmini")
                if prediction == 1:
                    st.markdown(f'<div class="result-danger">🚨 PHISHING TESPIT EDILDI!<br>'
                                f'Guven: %{confidence:.1f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe">✅ GUVENLI URL<br>'
                                f'Guven: %{confidence:.1f}</div>', unsafe_allow_html=True)

            with col_api:
                st.markdown(f"### 🔍 Dogrulama ({api_result['source'].split('(')[0].strip()})")
                if api_result['is_malicious']:
                    st.markdown(f'<div class="result-danger">🚨 TEHLIKELI<br>'
                                f'{api_result["details"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe">✅ TEMIZ<br>'
                                f'{api_result["details"]}</div>', unsafe_allow_html=True)

            # Uyum Durumu
            st.markdown("---")
            if is_correct:
                st.success("✅ **Model ve dogrulama kaynagi ayni sonuca ulasti.** Model bu ornegi dogruladi.")
            else:
                st.warning("⚠️ **Model ve dogrulama kaynagi farkli sonuc verdi!** "
                           "Bu ornek feedback olarak kaydedildi. "
                           "Model yeniden egitildiginde bu hatadan OGRENECEK.")

            # Ogrenme Dongusu Bilgisi
            updated_stats = get_feedback_stats()
            st.markdown(f"""
            <div class="learn-box">
                <strong>🧠 Yapay Zeka Ogrenme Dongusu</strong><br>
                <br>
                📊 Toplam analiz: <strong>{updated_stats['total']}</strong> |
                ✅ Dogru: <strong>{updated_stats['correct']}</strong> |
                ❌ Hatali: <strong>{updated_stats['incorrect']}</strong><br>
                <br>
                📈 Model dogrulugu: <strong>{updated_stats['accuracy']:.1%}</strong> |
                Yeniden egitim icin: <strong>{updated_stats['needed_for_retrain']}</strong> analiz daha<br>
                <br>
                <em>Her analiz modelin ogrenme verisine eklenir.
                {50} analize ulasinca model otomatik yeniden egitilir ve DAHA IYI olur!</em>
            </div>
            """, unsafe_allow_html=True)

            # Oznitlikler
            with st.expander("📋 Cikarilan Oznitelikler (16 adet)"):
                feat_col1, feat_col2 = st.columns(2)
                feat_items = list(features.items())
                mid = len(feat_items) // 2
                with feat_col1:
                    for k, v in feat_items[:mid]:
                        risk = "🔴" if (k in ['has_ip', 'has_at_sign', 'has_double_slash',
                                               'has_shortener', 'suspicious_tld'] and v == 1) else \
                               "🟡" if (k == 'has_dash' and v == 1) or \
                                       (k == 'has_https' and v == 0) else "🟢"
                        st.text(f"{risk} {k}: {v}")
                with feat_col2:
                    for k, v in feat_items[mid:]:
                        risk = "🔴" if (k in ['has_ip', 'has_at_sign', 'has_double_slash',
                                               'has_shortener', 'suspicious_tld'] and v == 1) else \
                               "🟡" if (k == 'has_dash' and v == 1) or \
                                       (k == 'has_https' and v == 0) else "🟢"
                        st.text(f"{risk} {k}: {v}")

            # Otomatik retrain kontrolu
            if updated_stats['ready_to_retrain']:
                st.info("🔄 Yeterli veri birikti! Sol menuden 'Modeli Yeniden Egit' butonuna tiklayabilirsiniz.")

        except Exception as e:
            st.error(f"Hata: {str(e)}")

elif analyze_btn and not url_input:
    st.warning("Lutfen bir URL girin.")
elif analyze_btn and not selected_model:
    st.warning("Lutfen bir model secin.")

# ── Alt Bilgi ──
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; font-size: 0.85rem;">
    <strong>PhishGuard-AI</strong> — Self-Learning Phishing URL Detection System<br>
    🧠 Yapay zeka ile kendi kendine ogrenen guvenlik sistemi<br>
    Her analiz modelin ogrenme verisine eklenir ve model surekli iyilesir.
</div>
""", unsafe_allow_html=True)
