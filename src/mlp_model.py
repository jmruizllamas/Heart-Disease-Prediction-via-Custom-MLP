# mlp_model.py
# This file contains the blueprint of our MLP model.

import numpy as np

class MLP:
    """
    Implementation of a Multi-Layer Perceptron (MLP) from scratch using NumPy.
    This class supports initialization, forward pass, and backward pass
    for training a neural network for binary classification tasks.
    """

    def __init__(self, layer_sizes, seed=42):
        """
        Constructor for the MLP class.
        Arguments:
            layer_sizes (list): List containing the number of neurons
                                in each layer of the network.
                                Example: [num_features, 16, 1]
            seed (int): Seed for random number generation to ensure reproducibility.
        Its job is to initialize the weights and biases for each layer.
        """

        # 1. Set the random seed so results are consistent each time the model is initialized.
        rng = np.random.default_rng(seed)

        # 2. Save the architecture
        self.layer_sizes = layer_sizes

        # 3. Initialize lists to store parameters
        self.weights = []
        self.biases = []

        # 4. Parameter initialization loop
        #    Iterate from the first hidden layer (index 1) to the last.
        #    No weights are needed for layer 0 (input).
        for i in range(1, len(self.layer_sizes)):
            # --- Weights Initialization ---
            # Create a weight matrix W for the connection between
            # layer (i-1) and layer (i).
            #
            # Shape: (neurons_prev_layer, neurons_current_layer)
            # E.g. W1 for [20, 16, 1] will be (20, 16)

            # Use the rng (Generator) instance instead of np.random.randn to avoid warnings
            w = rng.standard_normal((self.layer_sizes[i - 1], self.layer_sizes[i])) * 0.01
            self.weights.append(w)

            # --- Biases Initialization ---
            # Create a bias vector b for layer (i).
            #
            # Shape: (1, neurons_current_layer)
            # E.g. b1 for [20, 16, 1] will be (1, 16)

            b = np.zeros((1, self.layer_sizes[i]))
            self.biases.append(b)

    @staticmethod
    def _sigmoid(z):

        """
        Sigmoid activation function.
        Squashes any given z value to a value between 0 and 1.
        Used for the out layer in binary classification.
        """
        # np.clip() ensures stability by limiting extreme values of z

        z = np.clip(z, -500, 500) # Avoid overflow
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(z):
        """
        Derivative of the sigmoid function.
        Calculated as s * (1 - s), where 's' is the sigmoid output.
        Used during backpropagation.
        """
        s = MLP._sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def _relu(z):
        """
        ReLU (Rectified Linear Unit) activation function.
        Returns the input value if it's positive; otherwise, returns 0.
        Used for hidden layers to introduce non-linearity.
        """
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        """
        Derivative of the ReLU function.
        Returns 1 for positive input values and 0 for non-positive values.
        Used during backpropagation.
        """
        return (z > 0).astype(int)

    @staticmethod
    def _binary_cross_entropy(y_true, y_pred):
        """
        Binary Cross-Entropy Loss function.
        Measures the performance of a classification model
        whose output is a probability value between 0 and 1.
        """
        m = y_true.shape[0]
        # We add a small constant to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.squeeze(loss)

    def forward_pass(self, X):
            """
            Makes a forward pass through the MLP.
            Computes the output of the network for input data X.
            Arguments:
                X (np.ndarray): Input data of shape (num_samples, num_features).
            Returns:
                A_final (np.ndarray): Output of the network after the forward pass.
                cache (dict): Dictionary containing intermediate values (Z and A)
                              for each layer, useful for backpropagation.

            """

            # The 'cache' will store intermediate values
            cache = {}

            # 'A' (Activation) initially is the input X itself
            A = X
            cache['A0'] = X  # Save the input as Activation 0

            # Iterate through all layers (except the last) to apply ReLU activation
            num_hidden_layers = len(self.weights) - 1

            for i in range(num_hidden_layers):
                # Retrieve weights W and biases b for this layer
                W = self.weights[i]
                b = self.biases[i]

                # --- Step 1: Linear computation (Z) ---
                # Z = (input to this layer * weights) + bias
                # Z = A * W + b
                Z = np.dot(A, W) + b

                # --- Step 2: Activation computation (A) ---
                # Apply ReLU activation
                A = self._relu(Z)

                # --- Step 3: Save to cache ---
                cache[f'Z{i + 1}'] = Z
                cache[f'A{i + 1}'] = A

            # --- Output Layer Computation ---

            # Retrieve weights W and biases b of the FINAL layer
            W_final = self.weights[-1]  # -1 selects the last element
            b_final = self.biases[-1]

            # --- Final Step 1: Linear computation (Z final) ---
            Z_final = np.dot(A, W_final) + b_final

            # --- Final Step 2: Activation computation (A final) ---
            # Apply Sigmoid activation for the output
            A_final = self._sigmoid(Z_final)

            # --- Final Step 3: Save to cache ---
            cache[f'Z{len(self.weights)}'] = Z_final
            cache[f'A{len(self.weights)}'] = A_final

            return A_final, cache

    def backward_pass(self, y_true, A_final, cache):
        """
        Performs backpropagation to compute gradients of the loss
        with respect to weights and biases.

        Arguments:
            Y_true (np.ndarray): True labels of shape (num_samples, 1).
            A_final (np.ndarray): Output from the forward pass.
            cache (dict): Intermediate values from the forward pass.

        Returns:
            grads (dict): Dictionary containing gradients for weights and biases.
        """

        grads = {}  # Dictionary to store gradients
        m = y_true.shape[0]  # Number of samples

        # Ensure y_true has the same shape as A_final
        y_true = y_true.reshape(A_final.shape)

        # --- 1. Output Layer (Layer L) ---
        # The initial gradient dZ is simple thanks to the combination of Sigmoid and Binary Cross-Entropy.
        # dZ_final = A_final - y_true

        L = len(self.weights)  # Number of weight/bias layers (e.g. 2)

        dZ = A_final - y_true  # Gradient of layer L

        # Activation of the previous layer (L-1)
        A_prev = cache[f'A{L - 1}']

        # Gradients for the final layer
        grads[f'dW{L}'] = 1 / m * np.dot(A_prev.T, dZ)
        grads[f'db{L}'] = 1 / m * np.sum(dZ, axis=0, keepdims=True)

        # --- 2. Backward Loop (Layers L-1 ... 1) ---

        # Iterate in reverse from layer L-1 down to layer 1
        for l in reversed(range(L - 1)):
            # l goes (e.g. 0)
            # l+1 is the current layer (e.g. 1)
            # l+2 is the next layer (e.g. 2)

            # Retrieve the weights W and Z of the next and current layers
            W_next = self.weights[l + 1]  # Weights of the forward layer
            Z_current = cache[f'Z{l + 1}']  # Z of this layer
            A_prev = cache[f'A{l}']  # A of the previous layer (for dW)

            # --- Propagate the gradient backwards ---
            # dA = dZ_next * W_next.T
            dA = np.dot(dZ, W_next.T)

            # --- Compute dZ for this layer (ReLU) ---
            # dZ = dA * relu_derivative(Z)
            dZ = dA * self._relu_derivative(Z_current)

            # --- Compute gradients for this layer ---
            grads[f'dW{l + 1}'] = 1 / m * np.dot(A_prev.T, dZ)
            grads[f'db{l + 1}'] = 1 / m * np.sum(dZ, axis=0, keepdims=True)

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Updates the weights and biases using the computed gradients.

        W_new = W_old - learning_rate * dW
        b_new = b_old - learning_rate * db

        Arguments:
            grads (dict): Dictionary containing gradients for weights and biases.
            learning_rate (float): Learning rate for the update step.
        """

        L = len(self.weights)  # Number of layers

        for l in range(L):
            # Update weights
            self.weights[l] -= learning_rate * grads[f'dW{l + 1}']
            # Update biases
            self.biases[l] -= learning_rate * grads[f'db{l + 1}']

    def fit(self, X_train, Y_train, num_epochs, learning_rate, batch_size=32, verbose=True):
        """
        Trains the MLP using mini-batch gradient descent.

        Iterates over the training data for a specified number of epochs,
        performing forward and backward passes, and updating parameters.

        Arguments:
            X_train (np.ndarray): Training input data.
            Y_train (np.ndarray): True labels for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for parameter updates.
            batch_size (int): Size of each mini-batch.
            verbose (bool): If True, prints loss every 100 epochs.

        Returns:
            costs (list): List of loss values for each epoch.
        """
        costs = []  # To store the cost history
        m = X_train.shape[0]  # Total number of training samples

        # Ensure y_train has shape (m, 1)
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # Main training loop (epochs)
        for epoch in range(num_epochs):

            epoch_cost = 0.0  # Accumulated cost for this epoch

            # --- Mini-batch creation ---
            # First, shuffle the data so each batch is random
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation, :]
            y_shuffled = Y_train[permutation, :]

            # Split shuffled data into batches
            # Use np.array_split to handle the last batch if it's not perfect size
            num_batches = m // batch_size + (1 if m % batch_size != 0 else 0)

            for i in range(num_batches):
                # Extract current batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, m)
                X_batch = X_shuffled[start_idx:end_idx, :]
                y_batch = y_shuffled[start_idx:end_idx, :]

                # --- 1. Forward Pass (Predict) ---
                A_final, cache = self.forward_pass(X_batch)

                # --- 2. Cost Function (Measure Error) ---
                cost = self._binary_cross_entropy(y_batch, A_final)
                epoch_cost += cost

                # --- 3. Backward Pass (Compute Gradients) ---
                grads = self.backward_pass(y_batch, A_final, cache)

                # --- 4. Update Parameters (Learn) ---
                self.update_parameters(grads, learning_rate)

            # --- End of Epoch ---
            avg_epoch_cost = epoch_cost / num_batches

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch}/{num_epochs} - Cost: {avg_epoch_cost:.6f}")
            if epoch % 100 == 0:
                costs.append(avg_epoch_cost)

        return costs

    def predict(self, X):
        """
        Makes predictions for input data X.

        Arguments:
            X (np.ndarray): Input data of shape (num_samples, num_features).
        Returns:
            predictions (np.ndarray): Predicted class labels (0 or 1).
        """
        # 1. Forward pass to get probabilities
        A_final, _ = self.forward_pass(X)
        # 2. Convert probabilities to class labels (0 or 1)
        predictions = (A_final >= 0.5).astype(int)
        return predictions
