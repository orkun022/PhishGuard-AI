import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save=True):
    ensure_figures_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'], ax=ax)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gercek')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_proba, model_name='Model', save=True):
    ensure_figures_dir()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_feature_importance(importances, feature_names, model_name='Model', save=True):
    ensure_figures_dir()
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
    ax.barh(range(len(sorted_names)), sorted_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.set_xlabel('Onem Degeri')
    ax.set_title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, save=True):
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
        bars = ax.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Skor')
    ax.set_title('Model Karsilastirmasi')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.12)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'model_comparison.png')
        fig.savefig(path, dpi=150)
    plt.close(fig)


def print_classification_report(y_true, y_pred, model_name='Model'):
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))
