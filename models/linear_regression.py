from models.model import Model
import numpy as np
class Linear_Regression(Model):
    """Ordinary least-squares Linear Regression via batch gradient descent.

    Attributes:
        X (np.ndarray): Design matrix with bias row, shape (n_features+1, n_samples).
        Y (np.ndarray): Target vector, shape (n_samples,).
        theta (np.ndarray): Learned parameters, shape (n_features+1,).
        alpha (float): Learning rate.
        epochs (int): Number of passes over the training data.
    """
    def __init__(self, alpha=0.025, epochs=100):
        """Initialize hyperparameters and internal placeholders.

        Args:
            alpha (float): Step size for gradient updates.
            epochs (int): Number of passes through the data.
        """
        self.X = None
        self.Y = None
        self.theta = None
        self.alpha = alpha
        self.epochs = epochs

    def _hypothesis(self, X_i):
        """Compute the linear prediction for a single sample.

        Args:
            X_i: Feature-vector with bias term, shape (n_features+1,).

        Returns:
            Predicted value (float).
        """
        return np.dot(self.theta,X_i)

    def fit(self, X, Y):
        """Learn θ to minimize mean squared error on (X, Y).

        This builds the design matrix, initializes θ to zero, and
        runs batch gradient descent for `self.epochs` iterations.

        Args:
            X: Training data matrix, shape (n_samples, n_features).
            Y: Target vector, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        inputs = np.ones((n_samples, n_features+1))
        inputs[:,1:] = X
        self.X = inputs
        self.Y = Y
        self.theta = np.zeros(n_features + 1) # initial parameters are set to 0
        for i in range(self.epochs):
            self._lms_gradient_descent()

    def _lms_gradient_descent(self):
        """Perform one step of Least-Mean-Squares batch gradient descent.

        Updates self.theta in place over all samples.
        """
        n_samples, n_features = self.X.shape
        self.theta -= (self.alpha / n_samples) * self.X.T.dot(self.X.dot(self.theta) - self.Y)

    def predict(self, X):
        """Predict target values for new data.

        Args:
            X: New data matrix.
               If single sample: shape (n_features+1,) including bias.
               Otherwise: supply the design matrix with bias term.

        Returns:
            Predicted value(s). For a single X_i, returns float; for matrix, returns np.ndarray.
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted: theta is None.")
        y = self._hypothesis(X)
        return y



