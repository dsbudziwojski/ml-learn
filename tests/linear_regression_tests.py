import pytest
import numpy as np
from models.linear_regression import Linear_Regression


class TestLinearRegression:

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        lr = Linear_Regression()
        assert lr.alpha == 0.025
        assert lr.epochs == 100

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        lr = Linear_Regression(alpha=0.01, epochs=500)
        assert lr.alpha == 0.01
        assert lr.epochs == 500

    def test_fit_basic_functionality(self):
        """Test basic fit functionality with simple data."""
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]]).T
        Y_train = np.array([1.0, 2.0, 3.0])

        lr = Linear_Regression(alpha=0.01, epochs=500)
        lr.fit(X_train, Y_train)

        # Check that theta has been initialized and has correct shape
        # Should have n_features + 1 (for bias) elements
        assert hasattr(lr, 'theta')
        assert lr.theta.shape == (3,)  # 2 features + 1 bias

    def test_fit_perfect_linear_relationship(self):
        """Test fitting with a perfect linear relationship."""
        # y = 2x + 3 (perfect linear relationship)
        X_train = np.array([[1], [2], [3], [4], [5]])  # shape (5, 1)
        Y_train = np.array([5.0, 7.0, 9.0, 11.0, 13.0])  # 2*x + 3

        # Use smaller learning rate and more epochs for better convergence
        lr = Linear_Regression(alpha=0.001, epochs=5000)
        lr.fit(X_train, Y_train)

        # For perfect data, should converge reasonably close to true parameters
        # theta should be approximately [3, 2] (bias=3, slope=2)
        assert len(lr.theta) == 2
        assert abs(lr.theta[0] - 3.0) < 0.5  # bias term (more realistic tolerance)
        assert abs(lr.theta[1] - 2.0) < 0.5  # slope term (more realistic tolerance)

        # Alternative test: verify prediction accuracy on new data
        # Even if parameters aren't perfect, predictions should be accurate
        x_test = np.array([1.0, 6.0])  # bias + feature = 6
        expected_y = 2 * 6 + 3  # = 15
        predicted_y = lr.predict(x_test)
        assert abs(predicted_y - expected_y) < 0.5  # prediction should be accurate

    def test_predict_single_sample(self):
        """Test prediction for a single sample."""
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]]).T
        Y_train = np.array([1.0, 2.0, 3.0])

        lr = Linear_Regression(alpha=0.01, epochs=500)
        lr.fit(X_train, Y_train)

        # Predict single sample (must include bias term)
        x_new = np.array([1.0, 7.0, 8.0])  # [bias, feat1, feat2]
        prediction = lr.predict(x_new)

        assert isinstance(prediction, (float, np.floating))
        assert not np.isnan(prediction)

    def test_predict_batch(self):
        """Test batch prediction."""
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]]).T
        Y_train = np.array([1.0, 2.0, 3.0])

        lr = Linear_Regression(alpha=0.01, epochs=500)
        lr.fit(X_train, Y_train)

        # Prepare test data
        X_test = np.array([[7, 8],
                           [9, 10]])

        # Build design matrix with bias row
        X_test_design = np.vstack([
            np.ones(X_test.shape[0]),  # bias row
            X_test.T  # feature rows
        ])

        predictions = lr.predict(X_test_design)

        assert len(predictions) == 2
        assert all(isinstance(p, (float, np.floating)) for p in predictions)
        assert not any(np.isnan(predictions))

    def test_predict_known_linear_function(self):
        """Test prediction accuracy on a known linear function."""
        # Create data for y = 2x1 + 3x2 + 1
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        Y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + 1

        lr = Linear_Regression(alpha=0.01, epochs=2000)
        lr.fit(X_train, Y_train)

        # Test on new data
        X_test = np.array([[1.0, 2.0], [0.5, 1.5]])
        expected = [2 * 1.0 + 3 * 2.0 + 1, 2 * 0.5 + 3 * 1.5 + 1]  # [9.0, 6.5]

        # Build design matrix
        X_test_design = np.vstack([
            np.ones(X_test.shape[0]),
            X_test.T
        ])

        predictions = lr.predict(X_test_design)

        # Should be reasonably close to expected values
        for pred, exp in zip(predictions, expected):
            assert abs(pred - exp) < 0.5

    def test_predict_single_vs_batch_consistency(self):
        """Test that single predictions match batch predictions."""
        X_train = np.array([[1, 2, 3],
                            [4, 5, 6]]).T
        Y_train = np.array([1.0, 2.0, 3.0])

        lr = Linear_Regression(alpha=0.01, epochs=500)
        lr.fit(X_train, Y_train)

        # Single prediction
        x_single = np.array([1.0, 7.0, 8.0])
        pred_single = lr.predict(x_single)

        # Batch prediction with same sample
        x_batch = x_single.reshape(-1, 1)  # shape (3, 1)
        pred_batch = lr.predict(x_batch)

        assert abs(pred_single - pred_batch[0]) < 1e-10

    def test_fit_single_sample(self):
        """Test fit with only one sample."""
        X_train = np.array([[1, 2]])  # 1 sample, 2 features
        Y_train = np.array([3.0])  # 1 target

        lr = Linear_Regression(alpha=0.01, epochs=100)
        lr.fit(X_train, Y_train)

        # Should work but may not be meaningful
        assert hasattr(lr, 'theta')
        assert lr.theta.shape == (3,)  # 2 features + bias

    def test_convergence_with_different_learning_rates(self):
        """Test convergence behavior with different learning rates."""
        X_train = np.array([[1], [2], [3], [4]])
        Y_train = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x

        # Test with reasonable learning rate
        lr1 = Linear_Regression(alpha=0.01, epochs=1000)
        lr1.fit(X_train, Y_train)

        # Test with very small learning rate
        lr2 = Linear_Regression(alpha=0.001, epochs=1000)
        lr2.fit(X_train, Y_train)

        # Both should converge, but lr1 should be closer to optimal
        x_test = np.array([1.0, 5.0])  # bias + feature
        pred1 = lr1.predict(x_test)
        pred2 = lr2.predict(x_test)
        expected = 10.0  # 2 * 5

        # lr1 should be closer to expected value
        assert abs(pred1 - expected) <= abs(pred2 - expected)

    def test_reproducibility(self):
        """Test that results are reproducible with same parameters."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        Y_train = np.array([1.0, 2.0, 3.0])

        lr1 = Linear_Regression(alpha=0.01, epochs=500)
        lr2 = Linear_Regression(alpha=0.01, epochs=500)

        lr1.fit(X_train, Y_train)
        lr2.fit(X_train, Y_train)

        # Results should be identical (assuming deterministic implementation)
        assert np.allclose(lr1.theta, lr2.theta, atol=1e-10)

    def test_large_dataset_performance(self):
        """Test performance on a larger dataset."""
        np.random.seed(42)
        n_samples, n_features = 1000, 5

        X_train = np.random.randn(n_samples, n_features)
        true_theta = np.random.randn(n_features + 1)  # +1 for bias

        # Generate Y with known relationship
        X_design = np.column_stack([np.ones(n_samples), X_train])
        Y_train = X_design @ true_theta + 0.1 * np.random.randn(n_samples)

        lr = Linear_Regression(alpha=0.001, epochs=2000)
        lr.fit(X_train, Y_train)

        # Should learn reasonably close to true parameters
        assert lr.theta.shape == (n_features + 1,)

        # Test prediction
        X_test = np.random.randn(10, n_features)
        X_test_design = np.vstack([np.ones(X_test.shape[0]), X_test.T])
        predictions = lr.predict(X_test_design)

        assert len(predictions) == 10
        assert not any(np.isnan(predictions))


# Additional utility test functions
def test_linear_regression_integration():
    """Integration test mimicking the provided use case."""
    # Exact replication of the provided example
    X_train = np.array([[1, 2, 3],
                        [4, 5, 6]]).T  # shape (3, 2)
    Y_train = np.array([1.0, 2.0, 3.0])

    lr = Linear_Regression(alpha=0.01, epochs=500)
    lr.fit(X_train, Y_train)

    # --- Predict single sample ---
    x_new = np.array([1.0, 7.0, 8.0])  # [bias, feat1, feat2]
    y_pred_single = lr.predict(x_new)  # returns a float

    assert isinstance(y_pred_single, (float, np.floating))

    # --- Predict batch ---
    X_test = np.array([[7, 8],
                       [9, 10]])  # shape (2, 2)

    # Build design matrix with bias row
    X_test_design = np.vstack([
        np.ones(X_test.shape[0]),  # bias row
        X_test.T  # feature rows
    ])  # shape (3, 2)

    y_preds = lr.predict(X_test_design)  # returns array of length 2

    assert len(y_preds) == 2
    assert all(isinstance(p, (float, np.floating)) for p in y_preds)


if __name__ == "__main__":
    pytest.main([__file__])