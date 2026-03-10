"""
Visualization functions for recidivism prediction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional
import os


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig


def plot_roc_curves(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        results: {model_name: {'y_prob': np.ndarray, ...}}
        y_true: True labels
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, (name, res) in enumerate(results.items()):
        y_prob = res.get('risk_scores', res.get('y_prob'))
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=colors[i % 10], lw=2, label=f'{name} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).
    Perfect calibration = diagonal line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot(prob_pred, prob_true, 's-', label=model_name, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration Curve — {model_name}')
    ax.legend()
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_uncertainty_distribution(
    uncertainty_scores: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot histogram of uncertainty scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncertainty_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.median(uncertainty_scores), color='red', linestyle='--', label='Median')
    ax.set_xlabel('Predictive Variance (Uncertainty)')
    ax.set_ylabel('Count')
    ax.set_title(f'Uncertainty Distribution — {model_name}')
    ax.legend()
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot error rate vs uncertainty.
    Should show: higher uncertainty → higher error rate.
    """
    errors = (np.round(risk_scores) != y_true).astype(float)
    bins = np.percentile(uncertainty_scores, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    bin_centers, bin_errors = [], []

    for i in range(len(bins) - 1):
        mask = (uncertainty_scores >= bins[i]) & (uncertainty_scores < bins[i + 1])
        if i == len(bins) - 2:
            mask = (uncertainty_scores >= bins[i]) & (uncertainty_scores <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_errors.append(errors[mask].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, bin_errors, 'o-', color='tomato', lw=2, markersize=8)
    ax.set_xlabel('Predictive Variance (Uncertainty)')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'Error Rate vs Uncertainty — {model_name}')
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_risk_coverage_curve(
    selective_results: List[Dict],
    model_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot AUC vs coverage (selective prediction).
    Should show: lower coverage (more selective) → higher AUC.
    """
    coverages = [r['coverage'] for r in selective_results]
    aucs = [r['auc'] for r in selective_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, aucs, 'o-', color='mediumseagreen', lw=2, markersize=8)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('AUC')
    ax.set_title(f'Selective Prediction (Risk-Coverage Curve) — {model_name}')
    ax.set_xlim([0.4, 1.05])
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_calibration_by_uncertainty(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration curves split by uncertainty level (high vs low).
    Should show: low uncertainty predictions are better calibrated.
    """
    median_unc = np.median(uncertainty_scores)
    low_mask = uncertainty_scores <= median_unc
    high_mask = ~low_mask

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    for mask, label, color in [
        (low_mask, 'Low Uncertainty', 'steelblue'),
        (high_mask, 'High Uncertainty', 'tomato'),
    ]:
        if mask.sum() < 10:
            continue
        prob_true, prob_pred = calibration_curve(y_true[mask], risk_scores[mask], n_bins=10)
        ax.plot(prob_pred, prob_true, 'o-', color=color, lw=2, label=label)

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration by Uncertainty — {model_name}')
    ax.legend()
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_prediction_examples(
    all_predictions: np.ndarray,
    y_true: np.ndarray,
    n_examples: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot violin plots showing prediction distribution for sample individuals.
    Shows how MC Dropout produces a distribution, not a single point.
    """
    # Pick n_examples/2 correct and n_examples/2 incorrect predictions
    mean_preds = all_predictions.mean(axis=1)
    errors = np.round(mean_preds) != y_true

    correct_idx = np.where(~errors)[0]
    wrong_idx = np.where(errors)[0]

    rng = np.random.default_rng(42)
    n_each = n_examples // 2
    chosen_correct = rng.choice(correct_idx, size=min(n_each, len(correct_idx)), replace=False)
    chosen_wrong = rng.choice(wrong_idx, size=min(n_each, len(wrong_idx)), replace=False)
    chosen = np.concatenate([chosen_correct, chosen_wrong])

    fig, axes = plt.subplots(1, len(chosen), figsize=(2.5 * len(chosen), 5), sharey=True)
    if len(chosen) == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen):
        preds_i = all_predictions[idx]
        ax.violinplot(preds_i, positions=[0], showmedians=True)
        status = 'Correct' if not errors[idx] else 'Wrong'
        label = f'True={int(y_true[idx])}\n{status}'
        ax.set_title(label, fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticks([])
        ax.set_ylabel('Predicted Prob.' if ax == axes[0] else '')

    fig.suptitle('MC Dropout Prediction Distributions', y=1.02)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def create_full_report(
    results: Dict,
    y_true: np.ndarray,
    output_dir: str = 'results/figures',
) -> None:
    """
    Generate all figures for the final report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ROC curves for all models
    plot_roc_curves(results, y_true, save_path=os.path.join(output_dir, 'roc_curves.png'))

    for name, res in results.items():
        safe_name = name.replace(' ', '_').replace('+', 'plus')
        y_prob = res.get('risk_scores', res.get('y_prob'))
        if y_prob is None:
            continue

        plot_calibration_curve(
            y_true, y_prob, name,
            save_path=os.path.join(output_dir, f'calibration_{safe_name}.png'),
        )

        if 'uncertainty_scores' in res:
            unc = res['uncertainty_scores']
            risk = res['risk_scores']

            plot_uncertainty_distribution(
                unc, name,
                save_path=os.path.join(output_dir, f'uncertainty_dist_{safe_name}.png'),
            )
            plot_uncertainty_vs_error(
                y_true, risk, unc, name,
                save_path=os.path.join(output_dir, f'uncertainty_vs_error_{safe_name}.png'),
            )
            plot_calibration_by_uncertainty(
                y_true, risk, unc, name,
                save_path=os.path.join(output_dir, f'calibration_by_unc_{safe_name}.png'),
            )

            sp = res['metrics'].get('selective_prediction')
            if sp:
                plot_risk_coverage_curve(
                    sp, name,
                    save_path=os.path.join(output_dir, f'risk_coverage_{safe_name}.png'),
                )

            if 'all_predictions' in res:
                plot_prediction_examples(
                    res['all_predictions'], y_true,
                    save_path=os.path.join(output_dir, f'prediction_examples_{safe_name}.png'),
                )

    print(f"  Figures saved to: {output_dir}")
