"""
Main experiment script for uncertainty-aware recidivism prediction.

Run from cs229finalproject/:
    python main.py
"""

import numpy as np
import torch
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

# Ensure src is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_pipeline
from src.models import (
    create_logistic_regression,
    RecidivismMLP,
    RecidivismLSTM,
    create_sequences_for_lstm,
)
from src.train import train_pytorch_model, compute_class_weight
from src.mc_dropout import mc_dropout_predict, mc_dropout_predict_lstm
from src.ensemble import (
    train_ensemble_mlp_members,
    train_ensemble_lstm_members,
    ensemble_predict_mlp,
    ensemble_predict_lstm,
)
from src.evaluation import (
    compute_standard_metrics,
    compute_ece,
    compute_uncertainty_metrics,
    selective_prediction_analysis,
)
from src.visualization import create_full_report, set_style

# ============================================
# Configuration
# ============================================

CONFIG = {
    'data_path': os.path.join(os.path.dirname(__file__), 'data', 'nij_training.csv'),
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.1,   # Fraction of training set used for validation

    # MLP config
    'mlp_hidden_sizes': [128, 64, 32],
    'mlp_dropout': 0.3,
    'mlp_epochs': 100,
    'mlp_batch_size': 64,
    'mlp_lr': 0.001,
    'mlp_patience': 10,

    # LSTM config
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.3,
    'lstm_epochs': 100,
    'lstm_batch_size': 64,
    'lstm_lr': 0.001,
    'lstm_patience': 10,
    'lstm_weight_decay': 0.0,

    # MC Dropout config
    'mc_n_passes': 50,

    # Deep Ensemble config
    'ensemble_n_members': 5,
    'ensemble_base_seed': 100,

    # Output
    'results_dir': os.path.join(os.path.dirname(__file__), 'results'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def _save_metrics(metrics_dict: dict, path: str) -> None:
    """Save metrics dict as a formatted JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=float)


def main():
    np.random.seed(CONFIG['random_state'])
    torch.manual_seed(CONFIG['random_state'])

    print('=' * 60)
    print('Uncertainty-Aware Recidivism Prediction')
    print('=' * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results_dir = CONFIG['results_dir']
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)

    # ============================================
    # Step 1: Load and preprocess data
    # ============================================
    print('Step 1: Preprocessing data...')
    data = preprocess_pipeline(
        CONFIG['data_path'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
    )

    X_train_full = data['X_train']
    X_test = data['X_test']
    y_train_full = data['y_train']
    y_test = data['y_test']

    # Year-level labels (available for LSTM)
    y_train_year1 = data.get('y_train_year1', y_train_full)
    y_train_year2 = data.get('y_train_year2', y_train_full)
    y_train_year3 = data.get('y_train_year3', y_train_full)
    y_test_year1  = data.get('y_test_year1',  y_test)
    y_test_year2  = data.get('y_test_year2',  y_test)
    y_test_year3  = data.get('y_test_year3',  y_test)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=CONFIG['val_size'],
        random_state=CONFIG['random_state'],
        stratify=y_train_full,
    )

    # LSTM year labels aligned to train split
    n_train = len(X_train_full)
    train_idx = np.arange(n_train)
    sub_train_idx, _ = train_test_split(
        train_idx,
        test_size=CONFIG['val_size'],
        random_state=CONFIG['random_state'],
        stratify=y_train_full,
    )
    y_tr_y1 = y_train_year1[sub_train_idx]
    y_tr_y2 = y_train_year2[sub_train_idx]
    y_tr_y3 = y_train_year3[sub_train_idx]

    print(f'  Train: {X_train.shape[0]} samples')
    print(f'  Val:   {X_val.shape[0]} samples')
    print(f'  Test:  {X_test.shape[0]} samples')
    print(f'  Features: {X_train.shape[1]}')
    print(f'  Positive rate (train): {y_train.mean():.2%}')
    print()

    all_results = {}

    # ============================================
    # Step 2: Logistic Regression
    # ============================================
    print('Step 2: Training Logistic Regression...')

    lr_model = create_logistic_regression(random_state=CONFIG['random_state'])
    lr_model.fit(X_train, y_train)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    lr_metrics = compute_standard_metrics(y_test, lr_probs)
    lr_metrics['ece'] = compute_ece(y_test, lr_probs)

    all_results['Logistic Regression'] = {
        'y_prob': lr_probs,
        'metrics': lr_metrics,
    }

    print(f"  AUC:   {lr_metrics['auc']:.4f}")
    print(f"  Brier: {lr_metrics['brier_score']:.4f}")
    print(f"  ECE:   {lr_metrics['ece']:.4f}")
    print()

    # ============================================
    # Step 3: MLP + MC Dropout
    # ============================================
    print('Step 3: Training MLP...')

    mlp_model = RecidivismMLP(
        input_size=X_train.shape[1],
        hidden_sizes=CONFIG['mlp_hidden_sizes'],
        dropout=CONFIG['mlp_dropout'],
    )

    class_weight = compute_class_weight(y_train)

    mlp_result = train_pytorch_model(
        mlp_model, X_train, y_train, X_val, y_val,
        epochs=CONFIG['mlp_epochs'],
        batch_size=CONFIG['mlp_batch_size'],
        learning_rate=CONFIG['mlp_lr'],
        patience=CONFIG['mlp_patience'],
        device=CONFIG['device'],
        class_weight=class_weight,
    )

    mlp_model = mlp_result['model']

    # Standard (deterministic) prediction
    mlp_model.eval()
    with torch.no_grad():
        mlp_probs = torch.sigmoid(
            mlp_model(torch.FloatTensor(X_test).to(CONFIG['device']))
        ).cpu().numpy()

    mlp_metrics = compute_standard_metrics(y_test, mlp_probs)
    mlp_metrics['ece'] = compute_ece(y_test, mlp_probs)

    print(f"  AUC:   {mlp_metrics['auc']:.4f}")
    print(f"  Brier: {mlp_metrics['brier_score']:.4f}")
    print(f"  ECE:   {mlp_metrics['ece']:.4f}")
    print()

    # MC Dropout
    print(f"  Running MC Dropout ({CONFIG['mc_n_passes']} passes)...")
    mc_result = mc_dropout_predict(
        mlp_model, X_test,
        n_passes=CONFIG['mc_n_passes'],
        device=CONFIG['device'],
    )

    mc_metrics = compute_standard_metrics(y_test, mc_result['risk_scores'])
    mc_metrics['ece'] = compute_ece(y_test, mc_result['risk_scores'])
    mc_uncertainty_metrics = compute_uncertainty_metrics(
        y_test, mc_result['risk_scores'], mc_result['uncertainty_scores']
    )
    mc_metrics.update(mc_uncertainty_metrics)

    all_results['MLP'] = {
        'y_prob': mlp_probs,
        'metrics': mlp_metrics,
    }

    all_results['MLP + MC Dropout'] = {
        'y_prob': mc_result['risk_scores'],
        'risk_scores': mc_result['risk_scores'],
        'uncertainty_scores': mc_result['uncertainty_scores'],
        'all_predictions': mc_result['all_predictions'],
        'metrics': mc_metrics,
    }

    print(f"  MC Dropout AUC:              {mc_metrics['auc']:.4f}")
    print(f"  Uncertainty-Error Corr:      {mc_metrics['uncertainty_error_correlation']:.4f}")
    print(f"  Error Rate (Low Unc):        {mc_metrics['error_rate_low_uncertainty']:.4f}")
    print(f"  Error Rate (High Unc):       {mc_metrics['error_rate_high_uncertainty']:.4f}")
    print()

    # Save MLP model
    torch.save(
        mlp_model.state_dict(),
        os.path.join(results_dir, 'models', 'mlp_best.pt'),
    )

    # ============================================
    # Step 3b: MLP + Deep Ensemble
    # ============================================
    print(f"Step 3b: Training MLP Deep Ensemble ({CONFIG['ensemble_n_members']} members)...")

    mlp_ensemble_models = train_ensemble_mlp_members(
        input_size=X_train.shape[1],
        hidden_sizes=CONFIG['mlp_hidden_sizes'],
        dropout=CONFIG['mlp_dropout'],
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        n_members=CONFIG['ensemble_n_members'],
        base_seed=CONFIG['ensemble_base_seed'],
        epochs=CONFIG['mlp_epochs'],
        batch_size=CONFIG['mlp_batch_size'],
        learning_rate=CONFIG['mlp_lr'],
        patience=CONFIG['mlp_patience'],
        device=CONFIG['device'],
        class_weight=class_weight,
    )

    mlp_ens_result = ensemble_predict_mlp(mlp_ensemble_models, X_test, device=CONFIG['device'])

    mlp_ens_metrics = compute_standard_metrics(y_test, mlp_ens_result['risk_scores'])
    mlp_ens_metrics['ece'] = compute_ece(y_test, mlp_ens_result['risk_scores'])
    mlp_ens_metrics.update(compute_uncertainty_metrics(
        y_test, mlp_ens_result['risk_scores'], mlp_ens_result['uncertainty_scores']
    ))

    all_results['MLP + Deep Ensemble'] = {
        'y_prob': mlp_ens_result['risk_scores'],
        'risk_scores': mlp_ens_result['risk_scores'],
        'uncertainty_scores': mlp_ens_result['uncertainty_scores'],
        'all_predictions': mlp_ens_result['all_predictions'],
        'metrics': mlp_ens_metrics,
    }

    print(f"  AUC:                         {mlp_ens_metrics['auc']:.4f}")
    print(f"  Uncertainty-Error Corr:      {mlp_ens_metrics['uncertainty_error_correlation']:.4f}")
    print(f"  Error Rate (Low Unc):        {mlp_ens_metrics['error_rate_low_uncertainty']:.4f}")
    print(f"  Error Rate (High Unc):       {mlp_ens_metrics['error_rate_high_uncertainty']:.4f}")
    print()

    for i, m in enumerate(mlp_ensemble_models):
        torch.save(m.state_dict(),
                   os.path.join(results_dir, 'models', f'mlp_ensemble_{i}.pt'))

    # ============================================
    # Step 4: LSTM + MC Dropout (comparison)
    # ============================================
    print('Step 4: Training LSTM (for comparison)...')

    # Build sequences for train / val / test
    # Val year labels: use y_train_yearN from the val split indices
    val_idx_in_full = np.setdiff1d(np.arange(n_train), sub_train_idx)
    y_val_y1 = y_train_year1[val_idx_in_full]
    y_val_y2 = y_train_year2[val_idx_in_full]
    y_val_y3 = y_train_year3[val_idx_in_full]

    X_train_seq, y_train_seq = create_sequences_for_lstm(X_train, y_tr_y1, y_tr_y2, y_tr_y3)
    X_val_seq,   y_val_seq   = create_sequences_for_lstm(X_val,   y_val_y1, y_val_y2, y_val_y3)
    X_test_seq,  y_test_seq  = create_sequences_for_lstm(X_test,  y_test_year1, y_test_year2, y_test_year3)

    lstm_model = RecidivismLSTM(
        input_size=X_train_seq.shape[2],
        hidden_size=CONFIG['lstm_hidden_size'],
        num_layers=CONFIG['lstm_num_layers'],
        dropout=CONFIG['lstm_dropout'],
    )

    # Class weight from per-year labels (much lower positive rate than 3-yr aggregate)
    y_lstm_all_years = np.concatenate([y_tr_y1, y_tr_y2, y_tr_y3])
    lstm_class_weight = compute_class_weight(y_lstm_all_years)

    # For LSTM we train on all 3 time steps; validation uses year-3 AUC
    lstm_result = train_pytorch_model(
        lstm_model,
        X_train_seq, y_train_seq,
        X_val_seq,   y_val_seq,
        epochs=CONFIG['lstm_epochs'],
        batch_size=CONFIG['lstm_batch_size'],
        learning_rate=CONFIG['lstm_lr'],
        patience=CONFIG['lstm_patience'],
        device=CONFIG['device'],
        class_weight=lstm_class_weight,
        lstm_mode=True,
        weight_decay=CONFIG['lstm_weight_decay'],
    )

    lstm_model = lstm_result['model']

    # Standard LSTM prediction via survival rule:
    # P(recidivate within 3yr) = 1 - (1-p1)(1-p2)(1-p3)
    lstm_model.eval()
    with torch.no_grad():
        lstm_logits = lstm_model(torch.FloatTensor(X_test_seq).to(CONFIG['device']))
        lstm_year_probs = torch.sigmoid(lstm_logits).cpu().numpy()  # (n, 3)
    lstm_probs = 1 - np.prod(1 - lstm_year_probs, axis=1)

    lstm_metrics = compute_standard_metrics(y_test, lstm_probs)
    lstm_metrics['ece'] = compute_ece(y_test, lstm_probs)

    print(f"  LSTM AUC:   {lstm_metrics['auc']:.4f}")
    print(f"  LSTM Brier: {lstm_metrics['brier_score']:.4f}")
    print()

    # LSTM MC Dropout
    print(f"  Running LSTM MC Dropout ({CONFIG['mc_n_passes']} passes)...")
    lstm_mc_result = mc_dropout_predict_lstm(
        lstm_model, X_test_seq,
        n_passes=CONFIG['mc_n_passes'],
        device=CONFIG['device'],
    )

    lstm_mc_metrics = compute_standard_metrics(y_test, lstm_mc_result['risk_scores'])
    lstm_mc_metrics['ece'] = compute_ece(y_test, lstm_mc_result['risk_scores'])
    lstm_unc_metrics = compute_uncertainty_metrics(
        y_test, lstm_mc_result['risk_scores'], lstm_mc_result['uncertainty_scores']
    )
    lstm_mc_metrics.update(lstm_unc_metrics)

    all_results['LSTM'] = {
        'y_prob': lstm_probs,
        'metrics': lstm_metrics,
    }

    all_results['LSTM + MC Dropout'] = {
        'y_prob': lstm_mc_result['risk_scores'],
        'risk_scores': lstm_mc_result['risk_scores'],
        'uncertainty_scores': lstm_mc_result['uncertainty_scores'],
        'all_predictions': lstm_mc_result['all_predictions'],
        'metrics': lstm_mc_metrics,
    }

    torch.save(
        lstm_model.state_dict(),
        os.path.join(results_dir, 'models', 'lstm_best.pt'),
    )

    # ============================================
    # Step 4b: LSTM + Deep Ensemble
    # ============================================
    print(f"Step 4b: Training LSTM Deep Ensemble ({CONFIG['ensemble_n_members']} members)...")

    lstm_ensemble_models = train_ensemble_lstm_members(
        input_size=X_train_seq.shape[2],
        hidden_size=CONFIG['lstm_hidden_size'],
        num_layers=CONFIG['lstm_num_layers'],
        dropout=CONFIG['lstm_dropout'],
        X_train=X_train_seq, y_train=y_train_seq,
        X_val=X_val_seq, y_val=y_val_seq,
        n_members=CONFIG['ensemble_n_members'],
        base_seed=CONFIG['ensemble_base_seed'],
        epochs=CONFIG['lstm_epochs'],
        batch_size=CONFIG['lstm_batch_size'],
        learning_rate=CONFIG['lstm_lr'],
        patience=CONFIG['lstm_patience'],
        device=CONFIG['device'],
        class_weight=lstm_class_weight,
        weight_decay=CONFIG['lstm_weight_decay'],
    )

    lstm_ens_result = ensemble_predict_lstm(lstm_ensemble_models, X_test_seq, device=CONFIG['device'])

    lstm_ens_metrics = compute_standard_metrics(y_test, lstm_ens_result['risk_scores'])
    lstm_ens_metrics['ece'] = compute_ece(y_test, lstm_ens_result['risk_scores'])
    lstm_ens_metrics.update(compute_uncertainty_metrics(
        y_test, lstm_ens_result['risk_scores'], lstm_ens_result['uncertainty_scores']
    ))

    all_results['LSTM + Deep Ensemble'] = {
        'y_prob': lstm_ens_result['risk_scores'],
        'risk_scores': lstm_ens_result['risk_scores'],
        'uncertainty_scores': lstm_ens_result['uncertainty_scores'],
        'all_predictions': lstm_ens_result['all_predictions'],
        'metrics': lstm_ens_metrics,
    }

    print(f"  AUC:                         {lstm_ens_metrics['auc']:.4f}")
    print(f"  Uncertainty-Error Corr:      {lstm_ens_metrics['uncertainty_error_correlation']:.4f}")
    print(f"  Error Rate (Low Unc):        {lstm_ens_metrics['error_rate_low_uncertainty']:.4f}")
    print(f"  Error Rate (High Unc):       {lstm_ens_metrics['error_rate_high_uncertainty']:.4f}")
    print()

    for i, m in enumerate(lstm_ensemble_models):
        torch.save(m.state_dict(),
                   os.path.join(results_dir, 'models', f'lstm_ensemble_{i}.pt'))

    # ============================================
    # Step 5: Save results and generate report
    # ============================================
    print('Step 5: Generating report and saving results...')

    metrics_summary = {name: res['metrics'] for name, res in all_results.items()}
    _save_metrics(metrics_summary, os.path.join(results_dir, 'metrics', 'summary.json'))

    set_style()
    create_full_report(all_results, y_test, os.path.join(results_dir, 'figures'))

    # ============================================
    # Final console output
    # ============================================
    print()
    print('=' * 65)
    print('FINAL RESULTS')
    print('=' * 65)
    print(f"{'Model':<30} {'AUC':<10} {'Brier':<10} {'ECE':<10}")
    print('-' * 65)
    for name, res in all_results.items():
        m = res['metrics']
        ece_val = m.get('ece', float('nan'))
        print(f"{name:<30} {m['auc']:<10.4f} {m['brier_score']:<10.4f} {ece_val:<10.4f}")

    for mc_name in ('MLP + MC Dropout', 'LSTM + MC Dropout',
                    'MLP + Deep Ensemble', 'LSTM + Deep Ensemble'):
        if mc_name not in all_results:
            continue
        mc_m = all_results[mc_name]['metrics']
        print()
        print('=' * 65)
        print(f'UNCERTAINTY ANALYSIS — {mc_name}')
        print('=' * 65)
        print(f"  Uncertainty-Error Correlation: {mc_m['uncertainty_error_correlation']:.4f}")
        print(f"  Error Rate (Low Uncertainty):  {mc_m['error_rate_low_uncertainty']:.4f}")
        print(f"  Error Rate (High Uncertainty): {mc_m['error_rate_high_uncertainty']:.4f}")
        print()
        print('  Selective Prediction (AUC at different coverage levels):')
        for sp in mc_m.get('selective_prediction', []):
            print(f"    Coverage {sp['coverage']:.0%}: AUC = {sp['auc']:.4f}")

    print()
    print(f"Done! Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
