# Common imports
import numpy as np
import pandas as pd
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
    plt.scatter(X, y, s=50, c='lightgreen', marker='o', edgecolor='black', label="Training datapoints")
    plt.grid(True)
    plt.xlabel('Horsepower',fontsize=14)
    plt.ylabel('Miles per gallon',fontsize=14)

def plot_linear_reg(X,theta):
    Xval = np.arange(X.min()-15, X.max()+25, 0.05)
    Xval = np.reshape(Xval, (len(Xval), 1))
    Xtrain = np.insert(Xval,0,1,axis=1)
    # Plot the polynomial regression
    h = np.dot(Xtrain, theta)
    plt.plot(Xtrain, h, 'r--')

def normal_equation(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T),y)

def poly_features(X, degree):
    Xpoly = np.zeros((X.shape[0], degree))
    Xpoly[:,0] = X[:,0]
    for i in range(1,degree):
        Xpoly[:,i] = np.power(X[:,0],i+1); 
    return Xpoly

def plot_polynomial_reg(X,theta,degree):
    Xval = np.arange(X.min()-15, X.max()+25, 0.05)
    Xval = np.reshape(Xval, (len(Xval), 1))
    # Map the X values 
    Xpoly = poly_features(Xval, degree);
    Xpoly = np.insert(Xpoly,0,1,axis=1)
    # Plot the polynomial regression
    h = np.dot(Xpoly, theta)
    plt.title('Figure 3: Polynomial Regression(degree=%d)'%(degree))
    plt.plot(Xpoly, h, 'b--')
