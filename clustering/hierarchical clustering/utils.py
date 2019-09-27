import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

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

def plot_dataset(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], 
                c='white', marker='o', 
                edgecolor='black', s=50,label='Training data points')
    plt.xlabel("sepal length", fontsize=14)
    plt.ylabel("sepal width", fontsize=14)
 

def plot_clusters(X,y):
    plt.scatter(X[y == 0, 0],X[y == 0, 1],
                s=50, c='lightgreen',
                marker='o', edgecolor='black',label='class 1')
    plt.scatter(X[y == 1, 0],X[y == 1, 1],
                s=50, c='orange',
                marker='v', edgecolor='black',
                label='class 2')
    plt.xlabel("sepal length", fontsize=14)
    plt.ylabel("sepal width", fontsize=14)


