"""
Visualization & Utility Module
===============================
Confusion matrix, ROC curve, feature importance ve model karşılaştırma
grafikleri oluşturur.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmeyen backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')


def ensure_figures_dir():
    """Figures klasörünün var olduğundan emin ol."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name: str = 'Model', save: bool = True):
    """
    Confusion matrix görselleştirmesi.

    Parameters
    ----------
    y_true : array-like
        Gerçek etiketler.
    y_pred : array-like
        Tahmin edilen etiketler.
    model_name : str
        Model adı (grafik başlığı için).
    save : bool
        Grafik kaydedilsin mi.
    """
    ensure_figures_dir()

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legitimate', 'Phishing'],
        yticklabels=['Legitimate', 'Phishing'],
        ax=ax, linewidths=0.5, linecolor='gray'
    )
    ax.set_xlabel('Tahmin Edilen', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Kaydedildi: {path}")

    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_proba, model_name: str = 'Model', save: bool = True):
    """
    ROC Curve görselleştirmesi.

    Parameters
    ----------
    y_true : array-like
        Gerçek etiketler.
    y_proba : array-like
        Pozitif sınıf olasılıkları.
    model_name : str
        Model adı.
    save : bool
        Grafik kaydedilsin mi.
    """
    ensure_figures_dir()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Kaydedildi: {path}")

    plt.close(fig)
    return roc_auc


def plot_feature_importance(importances, feature_names, model_name: str = 'Model', save: bool = True):
    """
    Feature importance görselleştirmesi.

    Parameters
    ----------
    importances : array-like
        Öznitelik önem değerleri.
    feature_names : list of str
        Öznitelik isimleri.
    model_name : str
        Model adı.
    save : bool
        Grafik kaydedilsin mi.
    """
    ensure_figures_dir()

    # Sırala (büyükten küçüğe)
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = ax.barh(
        range(len(sorted_names)), sorted_importances[::-1],
        color=colors[::-1], edgecolor='white', linewidth=0.5
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel('Önem Değeri', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Kaydedildi: {path}")

    plt.close(fig)


def plot_model_comparison(results: dict, save: bool = True):
    """
    Birden fazla modelin performans karşılaştırması.

    Parameters
    ----------
    results : dict
        {model_name: {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}}
    save : bool
        Grafik kaydedilsin mi.
    """
    ensure_figures_dir()

    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(model_names))
    width = 0.18
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        # Değerleri barların üstüne yaz
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'model_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [✓] Kaydedildi: {path}")

    plt.close(fig)


def print_classification_report(y_true, y_pred, model_name: str = 'Model'):
    """
    Classification report yazdırır.

    Parameters
    ----------
    y_true : array-like
        Gerçek etiketler.
    y_pred : array-like
        Tahmin edilen etiketler.
    model_name : str
        Model adı.
    """
    print(f"\n{'=' * 50}")
    print(f"Classification Report — {model_name}")
    print('=' * 50)
    print(classification_report(
        y_true, y_pred,
        target_names=['Legitimate', 'Phishing']
    ))
