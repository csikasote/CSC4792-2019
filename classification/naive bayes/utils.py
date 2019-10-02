# Import common python libraries
import numpy as np
import pandas as pd
import os

# Matplotlib for plotting figures
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
    plt.scatter(X[y == 0, 0],X[y == 0, 1],
                s=50, c='red',
                marker='o', edgecolor='black',label='No')
    plt.scatter(X[y == 1, 0],X[y == 1, 1],
                s=50, c='green',
                marker='o', edgecolor='black',
                label='Yes')
    plt.legend(scatterpoints=1)
    plt.tight_layout()

def naive_bayes(X, y):
    parameters = []
    # Calculate the mean and variance of each feature for each class
    for c in np.unique(y):
        X_c = X[np.where(y == c)]
        mu = np.mean(X_c, axis= 0)
        prior = np.mean(y==c)
        var = np.var(X_c, axis= 0).T
        params = {"mean": mu, "var": var, "prior": prior}
        parameters.append(params)
    return parameters

def likelihood(X, mu, var):
    sigma = np.diag(var)
    sigma_sqrd_det = np.linalg.det(sigma)  
    sigma_sqrd_inv = np.linalg.pinv(sigma)  
    X_diff = X - mu
    eqn_part1 = 1/((2*np.pi)**(len(mu)/2.0) * sigma_sqrd_det**(0.5))
    eqn_part2 = np.exp(-0.5 * (X_diff.dot(sigma_sqrd_inv) * X_diff).sum(axis=1))
    return eqn_part1 * eqn_part2

def density(parameters, n_classes):
    density_contour_params = []
    for i in n_classes:
        linespace = np.arange(0, 10, 0.05)
        xx, yy = np.meshgrid(linespace, linespace)
        Z = likelihood(np.c_[xx.ravel(), yy.ravel()],
                       parameters[i]['mean'],
                       parameters[i]['var']);
        Z = Z.reshape(xx.shape)
        dimension = {'x':xx, 'y':yy, 'z':Z}
        density_contour_params.append(dimension)
    return density_contour_params


def posterior(sample,parameters,n_classes):
    posteriors = []
    for i in n_classes:
        likelihood_prob = likelihood(sample, parameters[i]["mean"], parameters[i]["var"])
        posterior = likelihood_prob * parameters[i]['prior']
        posteriors.append(posterior)
    return posteriors

def classify(n_classes,posteriors):
    y_hat = n_classes[np.argmax(posteriors)]
    if y_hat == 0:
        return 'Don`t Contact Customer!'
    elif y_hat == 1:
        return 'Contact Customer!'

