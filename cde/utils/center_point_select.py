from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
import numpy as np


def sample_center_points(Y, method='all', k=100, keep_edges=False, parallelize=False):
    """ function to define kernel centers with various downsampling alternatives

    Args:
      Y: numpy array from which kernel centers shall be selected - shape (n_samples,) or (n_samples, n_dim)
      method: kernel center selection method - choices: [all, random, distance, k_means, agglomerative]
      k: number of centers to be returned (not relevant for 'all' method)

    Returns: selected center points - numpy array of shape (k, n_dim). In case method is 'all' k is equal to n_samples
    """
    assert k <= Y.shape[0], "k must not exceed the number of samples in Y"

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
        y_s = pd.DataFrame(Y)
        # calculate distance of datapoint from "center of mass"
        dist = pd.Series(np.linalg.norm(Y - Y.mean(axis=0), axis=1), name='distance')
        df = pd.concat([dist, y_s], axis=1).sort_values('distance')

        # Y sorted by their distance to the
        Y = df[np.arange(Y.shape[1])].values
        centers = np.array([Y[0], Y[-1]])
        Y = Y[1:-1]
        # adjust k such that the final output has size k
        k -= 2
    else:
        centers = np.empty([0, 0])

    if method == 'random':
        cluster_centers = Y[np.random.choice(range(Y.shape[0]), k, replace=False)]

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method == 'distance':
        raise NotImplementedError  # TODO

    # use 1-D k-means clustering
    elif method == 'k_means':
        model = KMeans(n_clusters=k, n_jobs=n_jobs)
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
        return np.concatenate([centers, cluster_centers], axis=0)
    else:
        return cluster_centers


