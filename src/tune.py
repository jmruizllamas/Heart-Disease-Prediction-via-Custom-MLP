# tune.py
# This script finds the optimal hyperparameters for our models
# using K-Fold Cross-Validation, while keeping the final test set "pristine".

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score  # We use F1-Score as our optimization metric
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# Import our custom model
from src.mlp_model import MLP

# --- 1. Define the Hyperparameter Search Space ---
# This dictionary defines all the combinations we want to test.
PARAM_GRID = {
    'architectures': [
        [16],  # -> e.g., [20, 16, 1]
        [32],  # -> e.g., [20, 32, 1]
        [16, 8],  # -> e.g., [20, 16, 8, 1]
    ],
    'learning_rates': [0.01, 0.005],
    'num_epochs': [2000, 2500, 3000, 3500]  # Test 2000 vs 3000 epochs
}

# --- 2. Define Fixed Parameters ---
N_SPLITS = 5  # 5 Folds for K-Fold Cross-Validation
RANDOM_SEED = 42
BATCH_SIZE = 32


def main():
    """
    Main function to run the entire tuning pipeline.
    """

    # Suppress warnings from scikit-learn about convergence
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # --- 3. Load Data ---
    try:
        data = pd.read_csv('../data/processed_heart_disease_for_ml.csv')
    except FileNotFoundError:
        print("Error: 'processed_heart_disease_for_ml.csv' not found.")
        print("Please ensure the file is in the './data/' directory.")
        return

    X = data.drop('target', axis=1)
    y = data['target']

    # --- 4. The Golden Split: Train vs. Test ---
    # We split the data ONCE.
    # X_test_final, y_test_final are kept in a "vault" and NOT used for tuning.
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    print(f"Dataset split: {len(y_train_full)} samples for K-Fold tuning, {len(y_test_final)} for Final Test.")

    # --- 5. Tuning 'Our MLP' (Manual K-Fold Loop) ---
    print("\n--- Starting Optimization of 'Our MLP' (Manual K-Fold) ---")

    # Convert to NumPy arrays for our model
    X_train_np = X_train_full.values
    y_train_np = y_train_full.values.reshape(-1, 1)
    n_features = X_train_np.shape[1]

    # Initialize the K-Fold splitter
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results_ours = []

    # Iterate over every combination in our PARAM_GRID
    for arch_hidden in PARAM_GRID['architectures']:
        for lr in PARAM_GRID['learning_rates']:
            for epochs in PARAM_GRID['num_epochs']:

                fold_scores = []
                arch_full = [n_features] + arch_hidden + [1]
                config_str = f"Arch={arch_full}, LR={lr}, Epochs={epochs}"
                print(f"Testing: {config_str}")

                # The K-Fold loop (runs 5 times)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
                    # 1. Get the data for this fold
                    X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
                    y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

                    # 2. Scale the data (CRITICAL: fit on train_fold, transform on val_fold)
                    scaler = StandardScaler()
                    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                    X_val_fold_scaled = scaler.transform(X_val_fold)

                    # 3. Initialize and train our model
                    mlp = MLP(layer_sizes=arch_full, seed=RANDOM_SEED)
                    mlp.fit(X_train_fold_scaled, y_train_fold,
                            num_epochs=epochs,
                            learning_rate=lr,
                            batch_size=BATCH_SIZE,
                            verbose=False)  # verbose=False to avoid flooding the console

                    # 4. Evaluate on the validation fold
                    y_pred_val = mlp.predict(X_val_fold_scaled)
                    f1 = f1_score(y_val_fold, y_pred_val)
                    fold_scores.append(f1)

                # 5. Calculate the average F1-Score across all 5 folds
                avg_f1 = np.mean(fold_scores)
                results_ours.append({
                    'config': config_str,
                    'avg_f1_score': avg_f1
                })

    # Print the final leaderboard for our model
    print("\n--- 'Our MLP' Leaderboard (Mean F1-Score in K-Fold) ---")
    results_ours.sort(key=lambda x: x['avg_f1_score'], reverse=True)
    for res in results_ours:
        print(f"{res['config']:<35} | F1-Score: {res['avg_f1_score']:.4f}")

    # --- 6. Tuning 'scikit-learn' (The Easy Way: GridSearchCV) ---
    print("\n--- Starting Optimization of 'scikit-learn' (GridSearchCV) ---")

    # We must scale the *entire* full training set before passing it to GridSearchCV
    # GridSearchCV is smart enough to handle the scaling *within* its internal K-Folds
    # (Note: A more robust way is using a `Pipeline`, but this is fine)
    scaler_sklearn = StandardScaler()
    X_train_full_scaled = scaler_sklearn.fit_transform(X_train_full)

    # The parameter grid for scikit-learn
    sklearn_param_grid = {
        'hidden_layer_sizes': PARAM_GRID['architectures'],
        'learning_rate_init': PARAM_GRID['learning_rates'],
        'max_iter': PARAM_GRID['num_epochs']
    }

    # Initialize the MLPClassifier
    sklearn_mlp = MLPClassifier(batch_size=BATCH_SIZE,
                                activation='relu',
                                solver='adam',
                                random_state=RANDOM_SEED)

    # The magic tool!
    # cv=N_SPLITS performs the K-Fold cross-validation for us
    # scoring='f1' tells it to optimize for F1-Score
    # n_jobs=-1 uses all available CPU cores to speed up the search
    grid_search = GridSearchCV(sklearn_mlp, sklearn_param_grid,
                               cv=N_SPLITS,
                               scoring='f1',
                               verbose=1,  # Show progress
                               n_jobs=-1)

    start_time = time.time()
    # GridSearchCV wants the 1D version of y_train
    grid_search.fit(X_train_full_scaled, y_train_full)
    end_time = time.time()

    print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

    # Print the final leaderboard for scikit-learn
    print("\n--- 'scikit-learn' Leaderboard (Mean F1-Score in K-Fold) ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Mean F1-Score: {grid_search.best_score_:.4f}")

    # --- 7. Conclusion ---
    print("\nOptimization completed.")
    print("The next step is to use the best configurations found here")
    print("in the 'final_evaluation.py' script to run the final, single")
    print("comparison on the 'X_test_final' set.")


if __name__ == "__main__":
    main()