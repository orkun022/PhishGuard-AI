"""
Streamlit Web Arayüzü
=====================
Phishing URL Tespiti için interaktif web arayüzü.
Çalıştırma: streamlit run app/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features, FEATURE_NAMES
from src.predict import load_model, predict_url


# ─────────────────────────────────────────────
# Sayfa Yapılandırması
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Phishing URL Tespiti",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS Stilleri
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-safe {
        background: linear-gradient(135deg, #00C853, #69F0AE);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .result-danger {
        background: linear-gradient(135deg, #FF1744, #FF5252);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 0.3rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stButton > button {
        width: 100%;
        padding: 0.6rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Başlık
# ─────────────────────────────────────────────
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🛡️ Phishing URL Tespiti")
st.markdown("**Machine Learning ile URL güvenlik analizi**")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")

    # Model seçimi
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    available_models = []
    if os.path.exists(models_dir):
        available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f != 'scaler.pkl']

    if available_models:
        selected_model = st.selectbox(
            "Model Seçin:",
            available_models,
            index=available_models.index('best_model.pkl') if 'best_model.pkl' in available_models else 0
        )
    else:
        selected_model = None
        st.warning("⚠️ Eğitilmiş model bulunamadı!\n\n`python src/train.py` komutunu çalıştırın.")

    st.markdown("---")

    st.header("📚 Hakkında")
    st.markdown("""
    Bu uygulama, URL'lerden çıkarılan **16 öznitelik** kullanarak
    bir URL'nin **phishing** (kimlik avı) olup olmadığını tespit eder.

    **Kullanılan Öznitelikler:**
    - URL uzunluğu
    - Domain analizi
    - IP adresi kontrolü
    - Özel karakter sayısı
    - Alt domain derinliği
    - Ve daha fazlası...
    """)

    st.markdown("---")
    st.markdown("🎓 *Bilgisayar Mühendisliği Projesi*")


# ─────────────────────────────────────────────
# Ana Alan
# ─────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔗 URL Girin")
    url_input = st.text_input(
        "Analiz edilecek URL:",
        placeholder="https://example.com/login",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

with col2:
    st.subheader("📌 Örnek URL'ler")
    example_urls = {
        "✅ Google": "https://www.google.com/search?q=python",
        "✅ GitHub": "https://github.com/features",
        "🚨 Sahte PayPal": "http://paypal-secure-login.tk/update",
        "🚨 IP Tabanlı": "http://192.168.1.1/admin/login.php",
        "🚨 Kısaltılmış": "http://bit.ly/xk9f2m",
    }

    for label, url in example_urls.items():
        if st.button(label, key=f"example_{label}", use_container_width=True):
            url_input = url
            analyze_btn = True


# ─────────────────────────────────────────────
# Analiz Sonucu
# ─────────────────────────────────────────────
if analyze_btn and url_input and selected_model:
    with st.spinner("🔄 Analiz ediliyor..."):
        try:
            model_path = os.path.join(models_dir, selected_model)
            model, scaler = load_model(model_path)
            result = predict_url(url_input, model, scaler)

            st.markdown("---")

            # Sonuç gösterimi
            if result['prediction'] == 1:
                st.markdown(
                    f"<div class='result-danger'>"
                    f"🚨 PHISHING TESPİT EDİLDİ!<br>"
                    f"<small>Güven: {result['confidence']:.1f}%</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-safe'>"
                    f"✅ GÜVENLİ URL<br>"
                    f"<small>Güven: {result['confidence']:.1f}%</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Metrikler
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Sonuç", result['label'])
            with col_b:
                st.metric("Güven", f"{result['confidence']:.1f}%")
            with col_c:
                risk_level = "Yüksek" if result['prediction'] == 1 else "Düşük"
                st.metric("Risk Seviyesi", risk_level)

            # Öznitelik detayları
            st.markdown("---")
            st.subheader("📊 Çıkarılan Öznitelikler")

            features = result['features']
            feat_df = pd.DataFrame([
                {"Öznitelik": k, "Değer": v}
                for k, v in features.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Risk faktörleri
            st.subheader("⚠️ Risk Faktörleri")
            risk_factors = []
            if features.get('has_ip'):
                risk_factors.append("🔴 URL'de IP adresi kullanılıyor")
            if features.get('has_at_sign'):
                risk_factors.append("🔴 URL'de '@' işareti var")
            if features.get('has_double_slash'):
                risk_factors.append("🟡 Çift '//' yönlendirmesi tespit edildi")
            if features.get('has_shortener'):
                risk_factors.append("🟡 URL kısaltma servisi kullanılıyor")
            if features.get('suspicious_tld'):
                risk_factors.append("🔴 Şüpheli TLD kullanılıyor")
            if features.get('has_dash'):
                risk_factors.append("🟡 Domain'de '-' işareti var")
            if not features.get('has_https'):
                risk_factors.append("🟡 HTTPS kullanılmıyor")
            if features.get('url_length', 0) > 75:
                risk_factors.append("🟡 URL normalden uzun")
            if features.get('subdomain_count', 0) > 2:
                risk_factors.append("🔴 Çok fazla alt domain")
            if features.get('num_special_chars', 0) > 5:
                risk_factors.append("🟡 Çok fazla özel karakter")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.success("Belirgin bir risk faktörü tespit edilmedi.")

        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Bir hata oluştu: {str(e)}")

elif analyze_btn and not url_input:
    st.warning("⚠️ Lütfen bir URL girin.")

elif analyze_btn and not selected_model:
    st.error("❌ Eğitilmiş model bulunamadı. Önce `python src/train.py` komutunu çalıştırın.")


# ─────────────────────────────────────────────
# Alt Bilgi
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "🛡️ Phishing URL Tespiti — ML Projesi | "
    "Algoritmalar: Random Forest, XGBoost, SVM, Logistic Regression"
    "</div>",
    unsafe_allow_html=True
)
