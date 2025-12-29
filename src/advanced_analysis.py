# advanced_analysis.py
# Script to produce Statistical Analysis (Comp 5) and Interpretability (SHAP)
# Requirement: pip install shap scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Import our MLP class
from src.mlp_model import MLP

# --- CHAMPION CONFIGURATION (From tune.py) ---
OURS_BEST_CONFIG = {
    'layer_sizes_hidden': [16],  # [20, 16, 1]
    'learning_rate': 0.005,
    'num_epochs': 2500
}

RANDOM_SEED = 42
BATCH_SIZE = 32
N_SPLITS = 5  # 5 Folds for statistics


def calculate_confidence_interval(data, confidence=0.95):
    """Calculates mean, margin of error and confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h  # Return mean and margin (m +/- h)


def run_statistical_analysis(X, y):
    print("\n--- 1. STARTING STATISTICAL ANALYSIS (5-Fold CV) ---")
    print("Goal: Compute Standard Deviation and 95% Confidence Intervals")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Store metrics from each fold
    metrics_history = {
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'AUC-ROC': []
    }

    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"  > Processing Fold {fold}/{N_SPLITS}...")

        # Split
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Scale (fit on train, transform on val) to avoid strict Data Leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Convert y to numpy
        y_train_np = y_train_fold.values.reshape(-1, 1)
        y_val_np = y_val_fold.values.reshape(-1, 1)

        # Instantiate and train our Model
        n_features = X_train_scaled.shape[1]
        layers = [n_features] + OURS_BEST_CONFIG['layer_sizes_hidden'] + [1]

        mlp = MLP(layer_sizes=layers, seed=RANDOM_SEED)
        mlp.fit(X_train_scaled, y_train_np,
                num_epochs=OURS_BEST_CONFIG['num_epochs'],
                learning_rate=OURS_BEST_CONFIG['learning_rate'],
                batch_size=BATCH_SIZE, verbose=False)

        # Evaluate
        preds = mlp.predict(X_val_scaled)
        probs, _ = mlp.forward_pass(X_val_scaled)

        # Save metrics
        metrics_history['Accuracy'].append(accuracy_score(y_val_np, preds))
        metrics_history['Precision'].append(precision_score(y_val_np, preds))
        metrics_history['Recall'].append(recall_score(y_val_np, preds))
        metrics_history['F1-Score'].append(f1_score(y_val_np, preds))
        metrics_history['AUC-ROC'].append(roc_auc_score(y_val_np, probs))

        fold += 1

    # Final Statistics Calculation
    print("\n--- STATISTICAL RESULTS (Our Model) ---")
    results_df = pd.DataFrame(index=metrics_history.keys(),
                              columns=['Mean', 'Std Dev', '95% CI Margin', 'Lower Bound', 'Upper Bound'])

    for metric, values in metrics_history.items():
        mean, margin = calculate_confidence_interval(values)
        std_dev = np.std(values)

        results_df.loc[metric, 'Mean'] = mean
        results_df.loc[metric, 'Std Dev'] = std_dev
        results_df.loc[metric, '95% CI Margin'] = margin
        results_df.loc[metric, 'Lower Bound'] = mean - margin
        results_df.loc[metric, 'Upper Bound'] = mean + margin

        print(f"{metric:<10}: {mean:.4f} (+/- {margin:.4f}) | Std: {std_dev:.4f}")

    # Save to CSV for use in the LaTeX report
    results_df.to_csv('statistical_results_ours.csv')
    print("Statistical results saved to 'statistical_results_ours.csv'")


def run_shap_analysis(X, y):
    print("\n--- 2. STARTING EXPLAINABILITY ANALYSIS (SHAP) ---")
    print("Goal: Generate feature importance plots for the From-Scratch Model")

    # For SHAP we need a trained model. We'll use all data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_np = y.values.reshape(-1, 1)

    n_features = X_scaled.shape[1]
    layers = [n_features] + OURS_BEST_CONFIG['layer_sizes_hidden'] + [1]

    # Train final model
    print("  > Training final model for inspection...")
    model_for_shap = MLP(layer_sizes=layers, seed=RANDOM_SEED)
    model_for_shap.fit(X_scaled, y_np,
                       num_epochs=OURS_BEST_CONFIG['num_epochs'],
                       learning_rate=OURS_BEST_CONFIG['learning_rate'],
                       batch_size=BATCH_SIZE, verbose=False)

    # --- SHAP setup for custom model ---
    # KernelExplainer needs a function that receives X and returns probabilities
    def model_predict_proba(data):
        # data is numpy array
        probs, _ = model_for_shap.forward_pass(data)
        return probs.flatten()  # SHAP prefers 1D arrays for simple outputs

    print("  > Computing SHAP values (this may take a few minutes)...")

    # Use a summary of the dataset (kmeans) as background to speed up computation
    # Using the whole dataset would be very slow with KernelExplainer
    background_summary = shap.kmeans(X_scaled, 50)

    explainer = shap.KernelExplainer(model_predict_proba, background_summary)

    # Compute SHAP values for a representative sample (e.g. 100 samples)
    # If it takes too long, reduce to 50.
    sample_X = X_scaled[:100]
    shap_values = explainer.shap_values(sample_X)

    # --- Plot 1: Summary Plot (Beeswarm) ---
    print("  > Generating plots...")
    plt.figure()
    shap.summary_plot(shap_values, sample_X, feature_names=X.columns, show=False)
    plt.title("SHAP Summary Plot (Impact on Prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_beeswarm.png')
    plt.close()

    # --- Plot 2: Bar Plot (Global Importance) ---
    plt.figure()
    # For bar plot, shap_values may need adjustment depending on the version
    # If shap_values is a list (for multi-class), take index 0
    if isinstance(shap_values, list):
        vals = shap_values[0]
    else:
        vals = shap_values

    shap.summary_plot(vals, sample_X, feature_names=X.columns, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_importance_bar.png')
    plt.close()

    print("SHAP plots saved: 'shap_summary_beeswarm.png' and 'shap_importance_bar.png'")


def main():
    # Load data
    try:

        # Forma profesional (robusta)
        # 1. Obtiene la ruta donde está ESTE script (src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Sube un nivel para ir a la raíz (..) y luego entra en data
        csv_path = os.path.join(current_dir, '..', 'data', 'processed_heart_disease_for_ml.csv')

        # 3. Carga el archivo
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: processed dataset not found.")
        return

    X = data.drop('target', axis=1)
    y = data['target']

    # 1. Run statistical analysis (Mean +/- Std)
    run_statistical_analysis(X, y)

    # 2. Run explainability (SHAP)
    run_shap_analysis(X, y)

    print("\n--- ADVANCED ANALYSIS COMPLETED ---")


if __name__ == "__main__":
    main()