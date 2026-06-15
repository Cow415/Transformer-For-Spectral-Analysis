# Data structure node formats to use
import numpy as np

# Perceptrion Node
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Initialize weights to zeros (or small random numbers) and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Iteratively update parameters over specified epochs
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Calculate the linear summation
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Apply the Step Activation Function (outputs 1 or 0)
                y_predicted = 1 if linear_output >= 0 else 0
                
                # Perceptron learning rule for updating weights and bias
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # Compute predictions for a matrix of test inputs
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
# Tree Nodes




# Splitted Arrays