from linear_regression import Linear_Regression
import matplotlib.pyplot as plt
import numpy as np

lr = Linear_Regression()

X = np.array([
    [0.0, 1.0],
    [0.1, 1.1],
    [0.2, 1.2],
    [0.3, 1.3],
    [0.4, 1.4],
    [0.5, 1.5],
    [0.6, 1.6],
    [0.7, 1.7],
    [0.8, 1.8],
    [0.9, 1.9],
    [1.0, 2.0],
    [1.1, 2.1],
    [1.2, 2.2],
    [1.3, 2.3],
    [1.4, 2.4],
    [1.5, 2.5],
    [1.6, 2.6],
    [1.7, 2.7],
    [1.8, 2.8],
    [1.9, 2.9]
])

Y = np.array([
    7.993428268211527, 7.02650180057981, 8.901437758788763, 10.955150526196617, 7.743814287597135,
    9.206624265204212, 11.592542163443731, 12.29691912177958, 11.640321229212129, 12.65971217778857,
    13.073295771764006, 13.732951022172657, 14.813272348639394, 14.129987054052614, 15.183055134417927,
    14.890320457198703, 16.13411853254695, 17.102826177407727, 16.43615911718744, 17.07434004844718
])

# fitting the data
lr.fit(X.T,Y) # X is transposed

# parameters
print(lr.theta)

# X, dot, hypothesis
h = []
for i in range(20):
    print(lr.X[:,i], np.dot(lr.theta,lr.X[:,i]),lr.hypothesis(lr.X[:,i]), Y[i])
    h.append(lr.hypothesis(lr.X[:,i]))

# create 3D Plot
x1, x2 = np.linspace(0, 2,20), np.linspace(1,3,20)
X1, X2 = np.meshgrid(x1,x2)
Z = lr.theta[0] + lr.theta[1] * X1 + lr.theta[2] * X2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], Y, c="b", label="Data")
ax.plot_surface(X1, X2, Z, label="Prediction", color="red", alpha=0.3)

ax.set_xlabel('X_0')
ax.set_ylabel('X_1')
ax.set_zlabel('hypothesis')
plt.title('Data and Predicted Plot')
ax.legend()
plt.show()

# create 2D Plot
plt.scatter(Y,h)
plt.xlabel('Y (Actual)')
plt.ylabel('h (Predicted)')
plt.title('Actual vs Predicted')
plt.plot([7,18], [7,18], color='r')

plt.show()