# Common imports
import numpy as np
import pandas as pd
import os

# For plot/visualizations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ignoring Numpy error
np.seterr(all = 'ignore')

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

def feature_standardise(X):
    return (X- X.mean())/(X.max()-X.min())

def mean_squared_error(X,y):
    error_sqrd = np.square(X-y)
    sum_error_sqrd = np.sum(error_sqrd)
    return (1/(len(y))) * sum_error_sqrd
