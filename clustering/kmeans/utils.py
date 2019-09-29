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
    plt.grid(True)
    plt.tight_layout()
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
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel("sepal length", fontsize=14)
    plt.ylabel("sepal width", fontsize=14)

def L2(X,centroids):
    """ Returns the L2 distance between the ith training example and jth centroid"""
    dists = np.zeros((X.shape[0],centroids.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            dists[[i],[j]] = np.sqrt(np.sum(np.square(X[[i],:] - centroids[[j],:]), axis=1))
    return dists

def find_closest_centroids(X,centroids):
    return np.argmin(L2(X,centroids), axis=1)

def compute_centroids(X, idx, K):
    (m, n) = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        centroids[[k],:] = np.mean(X[idx == k,:],axis=0)
    return centroids

def initialize_centroids(X, K):
    rand_indices = np.random.permutation(X.shape[0])
    centroids = X[rand_indices[:K], :]
    return centroids

def plot_current_centroid(X, centroids, idx, K, i):
    # Plot the training datapoints 
    plot_clusters(X,idx)
    # Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1], 's',color='black',markersize=8)
    plt.title('Iteration number %d'% (i+1))
    

def kmeans_model(X, initial_centroids, max_iters, plot_progress = False):   
    # Initialize values
    (m,n) = X.shape
    K = initial_centroids.shape[0];
    centroids = initial_centroids;
    
    for i in range(max_iters):
        print('K-Means iteration %d/%d...\n'% (i+1, max_iters))
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        # Optionally, plot progress here
        if plot_progress:
            plt.figure(2)
            plot_current_centroid(X, centroids, idx, K, i);
            plt.axis([4.0,8.0,1.5,4.5])
            plt.show(block=False)
            input('Press enter to continue ...')
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K);
    print('\nK-Means Done.')
    return centroids, idx

def compute_distortion(X,idx,centroids):
    dists = L2(X,centroids)
    return np.sum(dists[np.arange(len(dists)), idx]**2)
