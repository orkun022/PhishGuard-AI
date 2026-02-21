# How I Built an AI-Powered Phishing URL Detector

> A deep technical walkthrough of PhishGuard-AI — from feature engineering to self-learning deployment.

---

## The Problem

Phishing attacks are the #1 attack vector in cybersecurity. Every **11 seconds**, a new phishing site goes live. Traditional blacklists can only catch **known** threats — they're reactive by nature. What if we could teach a machine to recognize phishing URLs it has **never seen before**?

That's exactly what PhishGuard-AI does.

---

## The Approach: 3-Layer Detection Pipeline

Instead of relying on a single detection method, I built a **3-layer verification system**:

### Layer 1: Machine Learning Engine

The core of PhishGuard-AI is a feature-based ML classifier. I extract **16 handcrafted features** from raw URLs:

```python
features = {
    "url_length": len(url),              # Phishing URLs tend to be longer
    "has_ip": has_ip_address(url),       # Legitimate sites use domain names
    "has_at_sign": "@" in url,           # Browsers ignore text before @
    "suspicious_tld": tld in SUSPICIOUS, # .tk, .ml, .ga are free/abused
    "num_dots": url.count("."),          # Excessive subdomains = suspicious
    # ... 11 more features
}
```

**Why these features?** Each one captures a known phishing pattern from cybersecurity literature. For example:
- `has_ip` — no legitimate bank uses `http://192.168.1.1/login`
- `has_at_sign` — `http://google.com@evil.com` actually navigates to `evil.com`
- `suspicious_tld` — free TLDs like `.tk` are heavily abused for phishing

I compared **4 ML algorithms**:

| Algorithm | How It Works | F1-Score |
|-----------|-------------|----------|
| **Logistic Regression** | Linear decision boundary | 1.0000 |
| **Random Forest** | Ensemble of decision trees | 1.0000 |
| **SVM (RBF)** | Kernel-based classification | 1.0000 |
| **XGBoost** | Gradient boosting | 0.9950 |

All performed excellently, but **Random Forest** was selected as the default due to its interpretability and feature importance output.

### Layer 2: API Verification

After the ML model makes its prediction, PhishGuard-AI cross-checks the URL against the **Google Safe Browsing API**. This creates a multi-source verification:

```
ML Prediction:  PHISHING (97% confidence)
API Verdict:    THREAT_DETECTED
Result:         ✅ Match — high confidence phishing
```

When the API is unavailable, the system falls back to **heuristic analysis** — checking the URL against known suspicious patterns (IP addresses, URL shorteners, suspicious TLDs).

### Layer 3: Self-Learning Feedback Loop

This is what makes PhishGuard-AI unique. After each scan:

1. The system compares the ML prediction with the API result
2. Logs the feedback as a **verified training sample** (JSON)
3. After collecting **50 verified samples**, automatically retrains the model

```python
feedback_log.append({
    "url": url,
    "ml_prediction": prediction,
    "api_result": api_label,
    "is_correct": ml_matches_api,
    "features": extracted_features
})

if len(feedback_log) >= 50:
    retrain_model(feedback_log)  # Auto-retrain
```

This means the model gets **smarter over time** — a concept inspired by online learning in production ML systems.

---

## Technical Decisions

### Why Feature Engineering over Deep Learning?

I chose handcrafted features over raw character-level models (LSTM/CNN) for several reasons:

1. **Interpretability** — I can explain *why* a URL was flagged (e.g., "contains IP address")
2. **Speed** — Feature extraction + Random Forest prediction takes <50ms
3. **Small data regime** — With 1,000 training samples, deep learning would overfit
4. **Deployment simplicity** — No GPU required, runs on Streamlit Cloud's free tier

### Why StandardScaler?

Features like `url_length` (range: 10-500) and `has_ip` (binary: 0/1) have vastly different scales. Without normalization, distance-based algorithms (SVM) would be dominated by the high-magnitude features. StandardScaler ensures each feature contributes equally.

### Why 5-Fold Cross-Validation?

A single train/test split can be misleading. Cross-validation gives a more robust estimate of model performance:
- Trains on 80% of data, tests on 20% — repeated 5 times
- Reports mean ± standard deviation
- Ensures the model generalizes, not just memorizes

---

## Deployment

The app is deployed on **Streamlit Cloud** with zero infrastructure overhead:

- **Live Demo:** [phishguard-ai.streamlit.app](https://phishguard-ai.streamlit.app)
- **Auto-deploy:** Every push to `main` triggers automatic redeployment
- **Self-contained:** If no pre-trained model exists, the app generates a demo dataset and trains models on first launch

---

## Lessons Learned

1. **Feature engineering > more data** — 16 well-chosen features outperformed throwing raw URLs at a neural network
2. **Multi-layer verification** builds real confidence — ML alone has blind spots, API alone has coverage gaps
3. **Self-learning is powerful** — Even a simple feedback loop dramatically improves model relevance over time
4. **Build for deployment** — Making the app auto-train on first launch removed the biggest deployment pain point

---

## What's Next

- Browser extension for real-time URL scanning
- WHOIS-based features (domain age, registrar)
- Deep learning model comparison (LSTM on raw URL characters)
- Docker containerization for enterprise deployment

---

*Built with Python, Scikit-learn, XGBoost, and Streamlit.*

**GitHub:** [github.com/orkun022/PhishGuard-AI](https://github.com/orkun022/PhishGuard-AI)
**Live Demo:** [phishguard-ai.streamlit.app](https://phishguard-ai.streamlit.app)
