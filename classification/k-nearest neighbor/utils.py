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


class KNearestNeighbor(object):
    def __init__(self, k=None):
        self.k = k

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        self.X = X
        self.y = y
    def distance(self,Xtest):
        """ Returns the L2 distance between the ith test example and jth training example"""
        dists = np.zeros((Xtest.shape[0],self.X.shape[0]))
        for i in range(Xtest.shape[0]):
            for j in range(self.X.shape[0]):
                dists[[i],[j]] = np.sqrt(np.sum(np.square(self.X[[j],:] - Xtest[[i],:]), axis=1))
        return dists

    def majority_vote(self, labels):
        """ Return the most common class among the k nearest neighbors """
        return np.bincount(labels[0]).argmax()
                      
    def predict(self, Xtest):
        """ Predicts the class to which the test example, 'Xtest' """
        y_hat = np.zeros((Xtest.shape[0]))
        for i in range(Xtest.shape[0]):
            k_nearest_neighbors = []
            dists = self.distance(Xtest)
            labels = self.y[np.argsort(dists[i])][:self.k]
            k_nearest_neighbors.append(labels.tolist())
            y_hat[i] = self.majority_vote(k_nearest_neighbors)
        return y_hat[0]

def knn_decision(idx):
    if idx == 0:
        return "Do Not Contact Customer!"
    elif idx == 1:
        return "Contact Customer!"
    
    
