# python
# train.py
# Extended script to train, evaluate and compare
# our MLP against scikit-learn.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- NEW IMPORTS! ---
# Import scikit-learn's MLP implementation
from sklearn.neural_network import MLPClassifier

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our MLP class
from src.mlp_model import MLP


def plot_learning_curve(costs, learning_rate, num_epochs):
    """
    Helper function to plot the cost learning curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title(f'Learning Curve (Cost)\nLR={learning_rate}, Epochs={num_epochs}')
    plt.xlabel('Iterations (x100 epochs)')
    plt.ylabel('Cost (Binary Cross Entropy)')
    plt.grid(True)
    # --- ADJUSTMENT ---
    plt.savefig('learning_curve.png')  # Save the plot
    print("Plot 'learning_curve.png' saved to project folder.")
    # plt.show() # Commented out to avoid stopping the script


def main():
    """
    Main function to run the training pipeline.
    """
    print("Starting training process...")

    # --- 1. Load Data ---
    try:
        data = pd.read_csv('../data/processed_heart_disease_for_ml.csv')
        print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns.")
    except FileNotFoundError:
        print("Error: 'processed_heart_disease_for_ml.csv' not found.")
        return

    # --- 2. Prepare Data (Preprocessing Pipeline) ---
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing completed.")

    # --- 3. Convert to NumPy and ensure correct dimensions ---
    y_train_np = y_train.values.reshape(-1, 1)
    y_test_np = y_test.values.reshape(-1, 1)

    # --- 4. Define Architecture and Hyperparameters ---
    n_features = X.shape[1]
    # This is the architecture we will optimize!
    architecture = [n_features, 16, 1]

    learning_rate = 0.01
    num_epochs = 2000
    batch_size = 32

    print(f"Network Architecture: {architecture}")
    print(f"Hyperparameters: LR={learning_rate}, Epochs={num_epochs}, Batch Size={batch_size}")

    # --- 5. Initialize and Train our Model ---
    mlp_ours = MLP(layer_sizes=architecture, seed=42)
    print("Starting training of our MLP...")
    costs = mlp_ours.fit(X_train_scaled, y_train_np,
                         num_epochs=num_epochs,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         verbose=True)
    print("Training of our MLP completed.")

    # --- 6. Visualize Results of Our Model ---
    if costs:
        plot_learning_curve(costs, learning_rate, num_epochs)

    # --- 7. (NEW) Train scikit-learn Model for Comparison ---
    print("\nStarting training of scikit-learn model...")

    # Create the sklearn MLP.
    # Important! Use the SAME architecture for a FAIR comparison.
    # (16,) means 1 hidden layer with 16 neurons.
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(16,),
                                activation='relu',  # Equivalent to our ReLU
                                solver='adam',  # An optimizer (like SGD)
                                max_iter=num_epochs,  # Equivalent to num_epochs
                                learning_rate_init=learning_rate,  # Equivalent to learning_rate
                                batch_size=batch_size,
                                random_state=42,
                                verbose=False)  # Set to True if you want to see its log

    # scikit-learn prefers y_train as a 1D vector, not a column.
    sklearn_mlp.fit(X_train_scaled, y_train)

    print("scikit-learn training completed.")

    # --- 8. (NEW) Evaluation and Performance Comparison ---
    print("\nEvaluating performance on the TEST SET...")

    # Get predictions from OUR model
    y_pred_ours = mlp_ours.predict(X_test_scaled)
    # Get probabilities (for AUC)
    y_prob_ours, _ = mlp_ours.forward_pass(X_test_scaled)

    # Get predictions from the SKLEARN model
    y_pred_sklearn = sklearn_mlp.predict(X_test_scaled)
    # Get probabilities (for AUC)
    y_prob_sklearn = sklearn_mlp.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        "Accuracy": (accuracy_score, False),
        "Precision": (precision_score, False),
        "Recall": (recall_score, False),
        "F1-Score": (f1_score, False),
        "AUC-ROC": (roc_auc_score, True)  # AUC needs probabilities, not preds.
    }

    results = {"Our MLP": {}, "scikit-learn": {}}

    for metric_name, (metric_func, use_probs) in metrics.items():
        if use_probs:
            results["Our MLP"][metric_name] = metric_func(y_test_np, y_prob_ours)
            results["scikit-learn"][metric_name] = metric_func(y_test_np, y_prob_sklearn)
        else:
            results["Our MLP"][metric_name] = metric_func(y_test_np, y_pred_ours)
            results["scikit-learn"][metric_name] = metric_func(y_test_np, y_pred_sklearn)

    # --- 9. (NEW) Print Final Comparison Report ---
    print("\n--- FINAL COMPARISON REPORT (Test Set) ---")
    print(f"{'Metric':<12} | {'Our MLP':<12} | {'scikit-learn':<12}")
    print("-" * 41)
    for metric_name in metrics:
        print(
            f"{metric_name:<12} | {results['Our MLP'][metric_name]:<12.4f} | {results['scikit-learn'][metric_name]:<12.4f}")


if __name__ == "__main__":
    main()