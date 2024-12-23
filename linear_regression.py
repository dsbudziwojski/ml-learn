from model import Model
import numpy as np
class linear_regression(Model):
    def __init__(self):
        self.X = None
        self.Y = None
        self.theta = None

    def hypothesis(self, j):
        return np.dot(self.theta.T,self.X[:,j])
    def fit(self, X, Y):
        self.X = np.ones(X.shape)
        self.X[1:np.shape(X)[0],1:np.shape(X)[1]] = X
        self.Y = np.array(Y)
        self.theta = np.zeros(X.shape[1]) # initial parameters are set to 0
        self.lms_gradient_descent()
    def lms_gradient_descent(self):
        # ToDo



