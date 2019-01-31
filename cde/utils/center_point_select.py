from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import pandas as pd
import numpy as np


def sample_center_points(Y, method='all', k=100, keep_edges=False, parallelize=False, random_state=None):
    """ function to define kernel centers with various downsampling alternatives

    Args:
      Y: numpy array from which kernel centers shall be selected - shape (n_samples,) or (n_samples, n_dim)
      method: kernel center selection method - choices: [all, random, distance, k_means, agglomerative]
      k: number of centers to be returned (not relevant for 'all' method)
      random_state: numpy.RandomState object

    Returns: selected center points - numpy array of shape (k, n_dim). In case method is 'all' k is equal to n_samples
    """
    assert k <= Y.shape[0], "k must not exceed the number of samples in Y"

    if random_state is None:
        random_state = np.random.RandomState()

    n_jobs = 1
    if parallelize:
        n_jobs = -2  # use all cpu's but one

    # make sure Y is 2d array of shape (
    if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=1)
    assert Y.ndim == 2

    # keep all points as kernel centers
    if method == 'all':
        return Y

    # retain outer points to ensure expressiveness at the target borders
    if keep_edges:
        ndim_y = Y.shape[1]
        n_edge_points = min(2 * ndim_y, k//2)

        # select 2*n_edge_points that are the farthest away from mean
        fathest_points_idx = np.argsort(np.linalg.norm(Y - Y.mean(axis=0), axis=1))[-2 * n_edge_points:]
        Y_farthest = Y[np.ix_(fathest_points_idx)]

        # choose points among Y farthest so that pairwise cosine similarity maximized
        dists = cosine_distances(Y_farthest)
        selected_indices = [0]
        for _ in range(1, n_edge_points):
            idx_greatest_distance = \
            np.argsort(np.min(dists[np.ix_(range(Y_farthest.shape[0]), selected_indices)], axis=1), axis=0)[-1]
            selected_indices.append(idx_greatest_distance)
        centers_at_edges = Y_farthest[np.ix_(selected_indices)]

        # remove selected centers from Y
        indices_to_remove = fathest_points_idx[np.ix_(selected_indices)]
        Y = np.delete(Y, indices_to_remove, axis=0)

        # adjust k such that the final output has size k
        k -= n_edge_points

    if method == 'random':
        cluster_centers = Y[random_state.choice(range(Y.shape[0]), k, replace=False)]

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method == 'distance':
        dists = euclidean_distances(Y)
        selected_indices = [0]
        for _ in range(1, k):
            idx_greatest_distance = np.argsort(np.min(dists[np.ix_(range(Y.shape[0]), selected_indices)], axis=1), axis=0)[-1]
            selected_indices.append(idx_greatest_distance)
        cluster_centers = Y[np.ix_(selected_indices)]


    # use 1-D k-means clustering
    elif method == 'k_means':
        model = KMeans(n_clusters=k, n_jobs=n_jobs, random_state=random_state)
        model.fit(Y)
        cluster_centers = model.cluster_centers_

    # use agglomerative clustering
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(Y)
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.DataFrame(Y)
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')[np.arange(Y.shape[1])].mean().values

    else:
        raise ValueError("unknown method '{}'".format(method))

    if keep_edges:
        return np.concatenate([centers_at_edges, cluster_centers], axis=0)
    else:
        return cluster_centers


