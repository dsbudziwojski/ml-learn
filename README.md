# ML Learn
This project explores the underlying structures of various machine learning models, by implementing each of the following without any specific machine learning library.

---
## Current Models
1. **Linear Regression**

   - **Methods:**
     - `__init__(alpha=0.025, epochs=100)`  
       – Set learning rate and number of gradient‐descent iterations.
     - `fit(X, Y)`  
       – Build design matrix (adds bias), initialize `theta`, and run batch gradient descent.
     - `_lms_gradient_descent()`  
       – Perform one full‐batch update of `theta`.
     - `_hypothesis(X_i)`  
       – Compute prediction for a single (bias‐augmented) sample.
     - `predict(X)`  
       – Return prediction(s) for a bias‐augmented sample or full design matrix.

   - **Use case:**

     ```python
     from models.linear_regression import Linear_Regression
     import numpy as np

     # --- Training ---
     # X_train: shape (n_samples, n_features)
     # Y_train: shape (n_samples,)
     X_train = np.array([[1, 2, 3],
                         [4, 5, 6]]).T      # shape (3, 2)
     Y_train = np.array([1.0, 2.0, 3.0])

     lr = Linear_Regression(alpha=0.01, epochs=500)
     lr.fit(X_train, Y_train)

     # --- Predict single sample ---
     # Must include bias term as first element (shape (n_features+1,))
     x_new = np.array([1.0, 7.0, 8.0])         # [bias, feat1, feat2]
     y_pred_single = lr.predict(x_new)         # returns a float

     # --- Predict batch ---
     # X_test: shape (n_test_samples, n_features)
     X_test = np.array([[7,  8],
                        [9, 10]])             # shape (2, 2)

     # Build design matrix with bias row:
     #   first row ones, then feature rows transposed
     X_test_design = np.vstack([
         np.ones(X_test.shape[0]),             # bias row
         X_test.T                              # feature rows
     ])                                         # shape (3, 2)

     y_preds = lr.predict(X_test_design)       # returns array of length 2


2. Decision Tree (Working on utility functions for DT)


3. K-Nearest-Neighbor...


