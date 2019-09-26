# Common imports
import numpy as np
import pandas as pd
import os

# For plot/visualizations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ignoring Numpy error
np.seterr(divide = 'ignore')

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
    plt.scatter(X, y, s=50, c='lightgreen', marker='o', edgecolor='black', label="Training data")
    plt.grid(True)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('y',rotation=0,fontsize=14)

def hypothesis(X, theta):
    return np.dot(X,theta)

def mse_function(X, y, theta):
    y_hat = hypothesis(X, theta);
    errors      = np.subtract(y_hat,y);
    squared_errors    = np.square(errors);
    return (1/(2*len(y))) * np.sum(squared_errors)

def gradient_descent(X, y, theta, learning_rate, num_iters):
    for i in range(num_iters):
        y_hat = hypothesis(X, theta)
        errors = np.subtract(y_hat,y)
        gradient = (1/len(y))*np.dot(X.T,errors)
        theta = theta - learning_rate*gradient
    return theta

def normal_equation(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T),y)

def fit_linear_regression(X,theta):   
    # Fit Simple linear Regression
    Xval = np.arange(X.min()-25, X.max()+25, 0.05)
    Xval = np.reshape(Xval, (len(Xval), 1))
    Xval = np.insert(Xval,0,1,axis=1)
    plt.plot(Xval[:,1], np.dot(Xval, theta), 'r-', label = 'Linear Model: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
    plt.plot(Xval[:,1],3*Xval[:,1]+.5, 'b-', label = 'True function: y = %0.2f + %0.2fx'%(0.5,3))

def fit_normal_equation(X,y):
    # Compute theta with normal equations
    X = np.insert(X,0,1,axis=1)
    theta = normal_equation(X,y)
    
    # Fit Simple linear Regression
    Xval = np.arange(X.min()-25, X.max()+25, 0.05)
    Xval = np.reshape(Xval, (len(Xval), 1))
    Xval = np.insert(Xval,0,1,axis=1)
    
    # Plot linear regression
    plt.plot(Xval[:,1], np.dot(Xval, theta), 'b--',label = 'Normal Equations: h(x) = %0.2f + %0.2fX'%(theta[0],theta[1]))

