import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# -------------------- Task 1: Generate a Complex Non-linear Dataset --------------------
def generate_complex_data(n_samples=5000):
    """
    Task 1: Implement the function to generate a complex non-linear dataset.

    Description:
    Generate a dataset based on a non-linear function with added noise.
    The function is defined as: y = 0.1x^3 - 0.5x^2 + 0.2x + 3 + sin(2x) + noise

    Args:
        n_samples: Number of samples to generate.

    Returns:
        X: Input feature array (n_samples,)
        y: Target values (n_samples,)
    """
    # TODO: Implement the dataset generation logic using the defined complex function
    x = np.linspace(-10,10,n_samples)

    noise = np.random.normal(0, 10, n_samples) # mean = 0, standard deviation = 10

    y = (0.1 * (x ** 3)) - (0.5 * (x ** 2)) + (0.2 * x) + 3 + np.sin(2*x) + noise 

    return x, y


# -------------------- Activation Functions and Their Derivatives --------------------
def relu(x):
    """
    Task 2: Implement the ReLU activation function.
    """
    # TODO: Implement ReLU activation function
    return np.maximum(0,x)


def relu_derivative(x):
    """
    Task 2: Implement the derivative of the ReLU activation function.
    """
    # TODO: Implement ReLU derivative
    return np.where(x > 0,1,0)


def tanh(x):
    """
    Task 2: Implement the Tanh activation function.
    """
    # TODO: Implement Tanh activation function
    return np.tanh(x)


def tanh_derivative(x):
    """
    Task 2: Implement the derivative of the Tanh activation function.
    """
    # TODO: Implement Tanh derivative
    return 1 - np.tanh(x)**2


# -------------------- Mean Squared Error Loss Function --------------------
def mse_loss(y_true, y_pred):
    """
    Task 3: Implement the Mean Squared Error (MSE) loss function.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        loss: Computed mean squared error.
    """
    # TODO: Calculate the mean squared error
    return np.mean((y_true - y_pred) ** 2)


# -------------------- Neural Network Class Definition --------------------
class TwoLayerMLP:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """
        Task 4: Initialize the neural network parameters (weights and biases).

        Description:
        Define and initialize weights and biases for three layers:
            1. Input to Hidden Layer 1
            2. Hidden Layer 1 to Hidden Layer 2
            3. Hidden Layer 2 to Output

        Args:
            input_size: Number of input features.
            hidden1_size: Number of neurons in the first hidden layer.
            hidden2_size: Number of neurons in the second hidden layer.
            output_size: Number of output neurons.
        """
        # TODO: Initialize weights and biases with appropriate dimensions
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))


    def forward(self, X):
        """
        Task 5: Implement the forward propagation logic.

        Args:
            X: Input features.

        Returns:
            output: Final output of the network.
        """
        # TODO: Implement forward propagation logic for each layer
        # Input to Hidden Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1  # takes the vector dot product
        self.a1 = relu(self.z1)  # Apply ReLU activation on z1

        # Hidden Layer 1 to Hidden Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = tanh(self.z2)  # Apply Tanh activation on z2

        # Hidden Layer 2 to Output
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        output = self.z3  # No activation in the output layer for regression
        return output

    def backward(self, X, y, output, lr=0.0005):
        """
        Task 6: Implement backpropagation to calculate gradients and update weights.

        Args:
            X: Input features.
            y: Ground truth target values.
            output: Network predictions.
            lr: Learning rate.
        """
        # TODO: Calculate output layer error and backpropagate through the network
        m = X.shape[0]  # number of samples

        # Calculate the error at the output layer (for regression, usually MSE)
        output_error = output - y.reshape(-1, 1)  # assuming y is of shape (n_samples,)

        # Backpropagate to Hidden Layer 2
        dW3 = (1 / m) * np.dot(self.a2.T, output_error)
        db3 = (1 / m) * np.sum(output_error, axis=0, keepdims=True)
        a2_error = np.dot(output_error, self.W3.T)  # propagate the error to a2
        dZ2 = a2_error * tanh_derivative(self.z2)   # derivative for Tanh activation

        # Backpropagate to Hidden Layer 1
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        a1_error = np.dot(dZ2, self.W2.T)  # propagate the error to a1
        dZ1 = a1_error * relu_derivative(self.z1)   # derivative for ReLU activation

        # Backpropagate to Input Layer
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=5000, lr=0.0005, batch_size=64):
        """
        Task 7: Implement the training process using mini-batch gradient descent.

        Args:
            X: Input features.
            y: Ground truth target values.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Size of each mini-batch.

        Returns:
            loss_history: List of training loss values over epochs.
        """
        loss_history = []
        num_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle the dataset at the start of each epoch -- this avoids biased data split
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)

                # Perform backpropagation
                self.backward(X_batch, y_batch, output, lr)

            # Calculate training loss for the entire training set
            epoch_loss = mse_loss(y, self.forward(X))

            # Test loss - using the same data as a proxy for simplicity, or you can use validation data if available
            test_loss = mse_loss(y, self.forward(X))  # You can adjust this if you have a separate test set

            # Store the loss for plotting later
            loss_history.append(epoch_loss)

            # Print epoch, training loss, and test loss every 100 epoch
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        return loss_history

    def predict(self, X):  # this will help with determining accuaracy of the model
        """
        Task 8: Implement the prediction logic using the trained model.

        Args:
            X: Input features.

        Returns:
            output: Predictions of the network.
        """
        # TODO: Implement the prediction using the forward pass
        return self.forward(X) #for the prediction, we can use the forward pass in the first step of the backpropagation algorithm


# -------------------- Task 9: Data Preparation and Model Training --------------------
# Prepare the dataset
X, y = generate_complex_data()
X = X.reshape(-1, 1)  # Reshape X to a 2D array
y = y.reshape(-1, 1)  # Reshape y to a 2D array

# Create and train the model
model = TwoLayerMLP(input_size=1, hidden1_size=128, hidden2_size=64, output_size=1)
loss_history = model.train(X, y, epochs=5000, lr=0.0005, batch_size=64)

# -------------------- Task 10: Visualization of Training Results --------------------
# TODO: Plot the training loss over epochs using matplotlib
# Plot training loss
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Use the trained model to make predictions
y_pred = model.predict(X)

# Plot true vs predicted values
plt.scatter(X, y, label='True Data', color='blue')
plt.plot(X, y_pred, color='red', label='Predicted Data')
plt.title("True vs Predicted Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

y_pred = model.predict(X)

# def hyperparameter_tuning(X, y, learning_rates, batch_sizes, epochs=1000):
#     """
#     Task 5: Experiment with different hyperparameters and record the performance.
    
#     Args:
#         X: Input features.
#         y: Ground truth target values.
#         learning_rates: List of learning rates to experiment with.
#         batch_sizes: List of batch sizes to experiment with.
#         epochs: Number of training epochs.
        
#     Returns:
#         None: Displays the loss curves for each configuration.
#     """
#     # Loop through each combination of learning rate and batch size
#     for lr in learning_rates:
#         for batch_size in batch_sizes:
            
#             # Create and train the model
#             model = TwoLayerMLP(input_size=1, hidden1_size=128, hidden2_size=64, output_size=1)
#             loss_history = model.train(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
            
#             # Plot the loss curve for this configuration
#             plt.plot(loss_history, label=f"LR: {lr}, Batch Size: {batch_size}")
#             plt.title(f"Loss Curve for LR: {lr}, Batch Size: {batch_size}")
#             plt.xlabel("Epochs")
#             plt.ylabel("Loss")
#             plt.legend()
#             plt.show()

# # -------------------- Running the Hyperparameter Tuning --------------------
# X, y = generate_complex_data()  # Generate the dataset
# X = X.reshape(-1, 1)  # Reshape X to a 2D array
# y = y.reshape(-1, 1)  # Reshape y to a 2D array

# learning_rates = [0.0001, 0.001, 0.01]
# batch_sizes = [32, 64, 128]

# # Perform hyperparameter tuning and show only the graphs
# hyperparameter_tuning(X, y, learning_rates, batch_sizes, epochs=1000)
