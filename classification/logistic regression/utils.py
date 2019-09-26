# Common imports
import numpy as np
import os

# For plot/visualizations
import matplotlib as mpl
import matplotlib.pyplot as plt

# Where to save the figures
EXERCISE_ROOT_DIR = "."
IMAGES_PATH = os.path.join(EXERCISE_ROOT_DIR, "images")

# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def hypothesis(X, theta):
    return np.dot(X,theta)

def feature_standardise(X):
    return (X-X.mean(axis=0))/(X.max(axis=0) - X.min(axis=0))

def cost_function(theta, X, y):
    h = hypothesis(X, theta)
    y_hat = sigmoid(h)
    cost = (-1/len(X)) * (np.dot(y.T,np.log(y_hat)) + np.dot((1-y).T,np.log(1-y_hat)))
    return cost 

def sigmoid(z):
    return (1/(1+ np.exp(-z)))

def predict(X,theta):
    probability = sigmoid(hypothesis(X,theta))
    return np.array([1 if x >= 0.5 else 0 for x in probability])

def lr_decision_boundary(X,theta):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(np.c_[np.ones((XX.shape[0], 1)), XX],theta)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    

def plot_decision_boundary(X,y,theta):
    fig, ax = plt.subplots()
    plot_dataset(X,y)
    x_vals = np.array(ax.get_xlim())
    y_vals = -1 * np.divide(((np.multiply(theta[1],x_vals)) + theta[0]),theta[2])
    plt.plot(x_vals, y_vals, '--', c="red", label='Decision Boundary')
    titlestr = 'Log. regression decision boundary(Model: h(x) = %.2f + %.2fx1 + %.2fx2)' % (theta[0], theta[1], theta[2])
    plt.title(titlestr)
    plt.grid(True)
    plt.legend()

def plot_dataset(X,y):
    plt.scatter(X[y == 0, 0],X[y == 0, 1],
                s=50, c='lightgreen',
                marker='o', edgecolor='black',label='class 1')
    plt.scatter(X[y == 1, 0],X[y == 1, 1],
                s=50, c='orange',
                marker='v', edgecolor='black',
                label='class 2')
    plt.tight_layout()
