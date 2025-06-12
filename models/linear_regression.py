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
        return np.dot(self.theta.T, X_i)

    def fit(self, X, Y):
        """Learn θ to minimize mean squared error on (X, Y).

        This builds the design matrix, initializes θ to zero, and
        runs batch gradient descent for `self.epochs` iterations.

        Args:
            X: Training data matrix, shape (n_samples, n_features).
            Y: Target vector, shape (n_samples,).
        """
        self.X = np.ones((X.shape[0]+1,X.shape[1]))
        self.X[1:1+np.shape(X)[0],:np.shape(X)[1]] = X
        self.Y = Y
        self.theta = np.zeros(self.X.shape[0]) # initial parameters are set to 0
        for i in range(self.epochs):
            self._lms_gradient_descent()

    def _lms_gradient_descent(self):
        """Perform one step of Least-Mean-Squares batch gradient descent.

        Updates self.theta in place over all samples.
        """
        for j in range(self.theta.shape[0]):
            sum_delta = 0
            for i in range(self.X.shape[1]):
                sum_delta += (self.Y[i] - self._hypothesis(self.X[:,i])) * self.X[j,i]
            self.theta[j] += self.alpha * sum_delta

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



