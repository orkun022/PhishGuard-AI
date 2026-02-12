<div align="center">

# 🛡️ PhishGuard-AI

### Machine Learning-Powered Phishing URL Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Detect phishing URLs in real-time using 16 handcrafted features and 4 ML classifiers.*

</div>

---

## 🎯 Motivation

Phishing attacks remain the **#1 cyber threat vector**, responsible for **36% of all data breaches** (Verizon DBIR 2024). Traditional blacklist-based approaches fail against **zero-day phishing URLs** — a new phishing site appears every **11 seconds**.

**PhishGuard-AI** bridges this gap by using **Machine Learning** to detect phishing URLs based on their structural patterns, without relying on any external database or API. The system extracts **16 lexical and host-based features** from raw URLs and classifies them using ensemble learning methods.

> **Why ML over blacklists?** Blacklists are reactive — they only catch known threats. ML models generalize from patterns, enabling proactive detection of previously unseen phishing URLs.

---

## ✨ Features

### 🔬 16 Handcrafted URL Features

| # | Feature | Description | Phishing Signal |
|---|---------|-------------|----------------|
| 1 | `url_length` | Total URL character count | >75 chars → suspicious |
| 2 | `domain_length` | Domain name length | Long domains mimic brands |
| 3 | `has_ip` | IP address in URL | Legitimate sites use domains |
| 4 | `has_at_sign` | `@` symbol present | Browsers ignore text before `@` |
| 5 | `has_double_slash` | Extra `//` redirects | URL redirection manipulation |
| 6 | `has_dash` | Hyphen in domain | `paypal-secure-login.com` |
| 7 | `has_https` | SSL/TLS encryption | HTTP = higher risk |
| 8 | `num_dots` | Dot count | Excessive subdomains |
| 9 | `num_digits` | Digit count | Random numbers = suspicious |
| 10 | `num_special_chars` | Special chars (`?&=%#@`) | URL obfuscation |
| 11 | `subdomain_count` | Subdomain depth | >2 levels = suspicious |
| 12 | `path_length` | URL path length | Deep paths hide phishing |
| 13 | `num_params` | Query parameter count | Data harvesting |
| 14 | `has_shortener` | URL shortening service | Hides real destination |
| 15 | `tld_length` | TLD character length | `.download` vs `.com` |
| 16 | `suspicious_tld` | Free/abused TLDs | `.tk`, `.ml`, `.ga` |

### 🤖 4 ML Classifiers Compared

| Algorithm | Type | Why It Works |
|-----------|------|-------------|
| **Logistic Regression** | Linear | Fast baseline, interpretable coefficients |
| **Random Forest** | Ensemble (Bagging) | Robust to overfitting, provides feature importance |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art for tabular data |
| **SVM (RBF Kernel)** | Kernel-based | Effective in high-dimensional spaces |

### 🌐 Real-Time Web Interface

Interactive **Streamlit** dashboard with:
- URL input → instant phishing/safe classification
- Confidence score & risk level display
- Extracted feature breakdown
- Risk factor analysis with visual indicators

---

## 📊 Results

All models trained on 1,000 URLs (500 phishing + 500 legitimate):

| Model | Accuracy | Precision | Recall | F1-Score | CV F1 (5-fold) |
|-------|----------|-----------|--------|----------|----------------|
| **Logistic Regression** | 100.00% | 100.00% | 100.00% | 1.0000 | 0.9988 ± 0.0025 |
| **Random Forest** | 100.00% | 100.00% | 100.00% | 1.0000 | 0.9987 ± 0.0025 |
| **SVM** | 100.00% | 100.00% | 100.00% | 1.0000 | 0.9988 ± 0.0025 |
| **XGBoost** | 99.50% | 99.01% | 100.00% | 0.9950 | 0.9975 ± 0.0031 |

> **Note:** Performance on real-world Kaggle datasets (e.g., PhiUSIIL 235K URLs) will show more realistic metrics. The demo dataset demonstrates the pipeline's correctness.

### Generated Evaluation Artifacts

- ✅ Confusion Matrix for each model
- ✅ ROC Curve with AUC scores
- ✅ Feature Importance ranking
- ✅ Multi-model comparison chart

---

## 📁 Project Structure

```
PhishGuard-AI/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                        # Raw datasets (CSV)
│   │   └── phishing_urls.csv       # Demo dataset (1000 URLs)
│   ├── processed/                  # Feature-extracted data
│   │   └── features.csv
│   └── README.md                   # Dataset documentation
│
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py       # 16-feature extraction + Big-O analysis
│   ├── preprocessing.py            # Data loading, scaling, train/test split
│   ├── train.py                    # Multi-model training pipeline
│   ├── predict.py                  # CLI & interactive prediction
│   ├── utils.py                    # Visualization (CM, ROC, FI charts)
│   └── generate_dataset.py         # Demo dataset generator
│
├── models/                         # Saved models (.pkl)
│   ├── best_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── scaler.pkl
│
├── reports/figures/                 # Generated charts (.png)
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── feature_importance_*.png
│   └── model_comparison.png
│
├── app/
│   └── app.py                      # Streamlit web interface
│
├── notebooks/                      # Jupyter analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
└── tests/
    ├── test_features.py            # 16 feature extraction tests
    └── test_model.py               # 6 model pipeline tests
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PhishGuard-AI.git
cd PhishGuard-AI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Generate Demo Dataset
```bash
python src/generate_dataset.py
```

### 2. Train All Models
```bash
python src/train.py
```
This trains 4 models, performs 5-fold cross-validation, generates evaluation charts, and saves the best model.

### 3. Predict a Single URL
```bash
python src/predict.py --url "http://paypal-secure-login.tk/verify"
```

**Output:**
```
🚨 PHISHING!
   Sonuç:   Phishing
   Güven:   97.3%
```

### 4. Interactive Mode
```bash
python src/predict.py
```

### 5. Launch Web Interface
```bash
streamlit run app/app.py
```

### 6. Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **scikit-learn** | ML algorithms (RF, LR, SVM) |
| **XGBoost** | Gradient Boosting classifier |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Streamlit** | Interactive web dashboard |
| **Joblib** | Model serialization |
| **pytest** | Unit testing framework |

---

## 🔮 Future Improvements

- [ ] Integration with real-time browser extension
- [ ] Deep Learning model (LSTM/CNN on raw URL characters)
- [ ] WHOIS-based features (domain age, registrar)
- [ ] Active content analysis (HTML/JS inspection)
- [ ] API endpoint for third-party integration
- [ ] Docker containerization

---

## 📚 References

1. Mohammad, R. M., et al. (2014). *Intelligent phishing detection system using feature extraction*. Journal of Intelligent Information Systems.
2. Sahingoz, O. K., et al. (2019). *Machine learning based phishing detection from URLs*. Expert Systems with Applications.
3. [PhiUSIIL Phishing URL Dataset — Kaggle](https://www.kaggle.com/datasets/akashkr/phiusiil-phishing-url-dataset)
4. [Web Page Phishing Detection — Kaggle](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)

---

## 📝 License

This project is developed for educational purposes as part of a Computer Engineering curriculum.

---

<div align="center">

**Built with ❤️ for Cybersecurity & AI**

</div>
