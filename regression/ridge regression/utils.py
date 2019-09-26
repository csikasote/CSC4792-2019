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

def plot_dataset(X,y):
    plt.scatter(X, y, s=50, c='lightgreen', marker='o',
                edgecolor='black', label="datapoints")
    plt.grid(True)
    plt.xlabel('Age',fontsize=14)
    plt.ylabel('Income',fontsize=14)

def poly_features(X, degree):
    Xpoly = np.zeros((X.shape[0], degree))
    Xpoly[:,0] = X[:,0]
    for i in range(1,degree):
        Xpoly[:,i] = np.power(X[:,0],i+1); 
    return Xpoly

def normal_equation(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T),y)

def ridge_regression(X,y,Lambda):
    regularization_term = Lambda * np.identity(X.shape[1]) 
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X) + regularization_term), X.T),y)


