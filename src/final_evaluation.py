# final_evaluation.py
#
# This is the final script.
# It takes the BEST configurations found during the tuning phase (tune.py),
# re-trains the champion models on the ENTIRE training set,
# and evaluates them ONE-AND-ONLY-ONCE on the "pristine" test set.
# This gives us our final, unbiased, and honest performance report.

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import warnings
from sklearn.exceptions import ConvergenceWarning

# Import our custom model
from mlp_model import MLP

# --- 1. CHAMPION CONFIGURATIONS (from tune.py) ---
# We manually enter the winning hyperparameters found in our tuning script.

OURS_BEST_CONFIG = {
    # Winner from 'Our MLP' Leaderboard
    # Arch=[20, 16, 1], LR=0.005, Epochs=2500 | F1-Score: 0.8422
    'layer_sizes_hidden': [16],
    'learning_rate': 0.005,
    'num_epochs': 2500
}

SKLEARN_BEST_CONFIG = {
    # Winner from 'scikit-learn' Leaderboard
    # Best Params: {'hidden_layer_sizes': [32], 'learning_rate_init': 0.01, 'max_iter': 2000}
    'hidden_layer_sizes': (32,),
    'learning_rate_init': 0.01,
    'max_iter': 2000
}

# --- 2. Fixed Parameters ---
RANDOM_SEED = 42
BATCH_SIZE = 32


def main():
    """
    Main function to run the final evaluation.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Starting FINAL EVALUATION...")

    # --- 3. Load Data and Perform the "Golden Split" ---
    try:

        # Forma profesional (robusta)
        # 1. Obtiene la ruta donde está ESTE script (src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Sube un nivel para ir a la raíz (..) y luego entra en data
        csv_path = os.path.join(current_dir, '..', 'data', 'processed_heart_disease_for_ml.csv')

        # 3. Carga el archivo
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: 'processed_heart_disease_for_ml.csv' not found.")
        return

    X = data.drop('target', axis=1)
    y = data['target']

    # We split the data exactly as we did in tune.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # --- 4. Scale the Data ---
    # We fit the scaler ONLY on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # We ONLY transform the test data (using the scaler fit on train)
    X_test_scaled = scaler.transform(X_test)

    # Convert 'y' to NumPy arrays in the correct shape
    y_train_np = y_train.values.reshape(-1, 1)
    y_test_np = y_test.values.reshape(-1, 1)

    print(f"Data prepared. Training champions on {len(y_train)} samples.")
    print(f"Final evaluation will be on {len(y_test)} pristine test samples.")

    # --- 5. Train "Our" Champion Model ---
    print("Training 'Our MLP' (Best Config)...")

    n_features = X_train_scaled.shape[1]  # This will be 20

    # Build the full architecture list: [20] + [16] + [1]
    ours_layers = [n_features] + OURS_BEST_CONFIG['layer_sizes_hidden'] + [1]

    mlp_ours = MLP(layer_sizes=ours_layers,
                   seed=RANDOM_SEED)

    # Train the model on the ENTIRE training set
    mlp_ours.fit(X_train_scaled, y_train_np,
                 num_epochs=OURS_BEST_CONFIG['num_epochs'],
                 learning_rate=OURS_BEST_CONFIG['learning_rate'],
                 batch_size=BATCH_SIZE,
                 verbose=False)  # No need to print cost, we just want the final model

    print("'Our MLP' training complete.")

    # --- 6. Train the 'scikit-learn' Champion Model ---
    print("Training 'scikit-learn' (Best Config)...")

    sklearn_mlp = MLPClassifier(
        hidden_layer_sizes=SKLEARN_BEST_CONFIG['hidden_layer_sizes'],
        learning_rate_init=SKLEARN_BEST_CONFIG['learning_rate_init'],
        max_iter=SKLEARN_BEST_CONFIG['max_iter'],
        batch_size=BATCH_SIZE,
        activation='relu',
        solver='adam',
        random_state=RANDOM_SEED,
        verbose=False
    )

    # Train on the ENTIRE training set (sklearn prefers 1D y_train)
    sklearn_mlp.fit(X_train_scaled, y_train)

    print("'scikit-learn' training complete.")

    # --- 7. The Final Judgement ---
    print(f"\nEvaluating performance on the pristine {len(y_test)} sample TEST SET...")

    # Get predictions and probabilities from OUR model
    y_pred_ours = mlp_ours.predict(X_test_scaled)
    y_prob_ours, _ = mlp_ours.forward_pass(X_test_scaled)

    # Get predictions and probabilities from SKLEARN
    y_pred_sklearn = sklearn_mlp.predict(X_test_scaled)
    y_prob_sklearn = sklearn_mlp.predict_proba(X_test_scaled)[:, 1]

    # --- 8. Calculate All Metrics ---
    metrics = {
        "Accuracy": (accuracy_score, False),
        "Precision": (precision_score, False),
        "Recall": (recall_score, False),
        "F1-Score": (f1_score, False),
        "AUC-ROC": (roc_auc_score, True)  # AUC needs probabilities
    }

    results = {"Our MLP": {}, "scikit-learn": {}}

    for metric_name, (metric_func, use_probs) in metrics.items():
        if use_probs:
            # Use probabilities for AUC-ROC
            results["Our MLP"][metric_name] = metric_func(y_test_np, y_prob_ours)
            results["scikit-learn"][metric_name] = metric_func(y_test_np, y_prob_sklearn)
        else:
            # Use binary predictions (0/1) for all other metrics
            results["Our MLP"][metric_name] = metric_func(y_test_np, y_pred_ours)
            results["scikit-learn"][metric_name] = metric_func(y_test_np, y_pred_sklearn)

    # --- 9. Print the Final Report Table ---
    print("\n--- FINAL PROJECT REPORT (Test Set Performance) ---")
    print(f"{'Metric':<12} | {'Our MLP':<12} | {'scikit-learn':<12}")
    print("-" * 41)
    for metric_name in metrics:
        print(
            f"{metric_name:<12} | {results['Our MLP'][metric_name]:<12.4f} | {results['scikit-learn'][metric_name]:<12.4f}")

    # --- 10. Generate Final Confusion Matrices ---
    print("\nGenerating final confusion matrix plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Final Model Performance on Pristine Test Set', fontsize=16)

    # Matrix for Our MLP
    cm_ours = confusion_matrix(y_test_np, y_pred_ours)
    sns.heatmap(cm_ours, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred Neg (0)', 'Pred Pos (1)'],
                yticklabels=['True Neg (0)', 'True Pos (1)'])
    axes[0].set_title('Confusion Matrix - Our MLP (From Scratch)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Matrix for scikit-learn
    cm_sklearn = confusion_matrix(y_test_np, y_pred_sklearn)
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Pred Neg (0)', 'Pred Pos (1)'],
                yticklabels=['True Neg (0)', 'True Pos (1)'])
    axes[1].set_title('Confusion Matrix - scikit-learn MLP')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.savefig('final_confusion_matrices.png')
    print("Plot 'final_confusion_matrices.png' saved.")


if __name__ == "__main__":
    main()