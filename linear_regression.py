from model import Model
import numpy as np
class Linear_Regression(Model):
    def __init__(self):
        self.X = None
        self.Y = None
        self.theta = None
        self.alpha = 0.05
        self.epochs = 100

    def hypothesis(self, j):
        return np.dot(self.theta.T,self.X[:,j])
    def fit(self, X, Y):
        self.X = np.ones((X.shape[0]+1,X.shape[1]))
        self.X[1:1+np.shape(X)[0],:np.shape(X)[1]] = X
        self.Y = Y
        self.theta = np.zeros(self.X.shape[0]) # initial parameters are set to 0
        for i in range(self.epochs):
            self.lms_gradient_descent()

    def lms_gradient_descent(self):
        for j in range(self.X.shape[0]):
            sum_delta = 0
            for i in range(self.X.shape[1]):
                sum_delta += (self.Y[i] - self.hypothesis(i)) * self.X[j,i]
            self.theta[j] += self.alpha * sum_delta



