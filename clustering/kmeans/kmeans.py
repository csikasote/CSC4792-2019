# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

import scipy.io as sio
import pandas as pd
import numpy as np
import os

#Import helper functions from utils.py
from utils import save_fig
from utils import plot_dataset
from utils import plot_clusters
from utils import find_closest_centroids
from utils import compute_centroids
from utils import initialize_centroids
from utils import kmeans_model
from utils import L2

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt


# load the preprocessed iris dataset
dataset = pd.read_csv('datasets/iris_process.data',
                      delimiter=',',
                      header=None, 
                      names=['sepal length', 'sepal width', 'petal length', 'petal width','class'])

print('\nNumber of records:',len(dataset))
print('\nclass domain:', dataset['class'].unique())
print('\n\n',dataset.head())


X = pd.DataFrame(dataset,columns=['sepal length','sepal width']).values
print('The shape of X:',X.shape)

plt.figure(1)
plt.title('Iris data: sepal length vs. sepal width')
plot_dataset(X, y=None)
plt.axis([4.0,8.0,1.5,4.5])
plt.legend()
save_fig("kmeans_datapoints")
plt.show(block=False)
input("Press <ENTER> key to continue ...")

# Settings for running K-Means
K = 2; # Number of clusters
max_iters = 10;
initial_centroids = initialize_centroids(X, K)
centroids, idx = kmeans_model(X, initial_centroids, max_iters,plot_progress=True)

plt.figure(2)
plt.title('Iris data: sepal length vs. sepal width')
plot_clusters(X,idx)
plt.plot(centroids[:,0], centroids[:,1],'s',color='black',markersize=8, label='centroids')
plt.axis([4.0,8.0,1.5,4.5])
plt.legend()
save_fig("kmeans_clusters")
plt.show(block=False)
input("Press <ENTER> key to continue ...")
