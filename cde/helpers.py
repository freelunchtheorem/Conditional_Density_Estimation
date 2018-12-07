from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
import numpy as np
import logging
import scipy.stats as stats



def norm_along_axis_1(A, B, squared=False):
  """ calculates the (squared) euclidean distance along the axis 1 of both 2d arrays

  Args:
    A: numpy array of shape (n, k)
    B: numpy array of shape (m, k)
    squared: boolean that indicates whether the squared euclidean distance shall be returned, \
             otherwise the euclidean distance is returned

    Returns:
       euclidean distance along the axis 1 of both 2d arrays - numpy array of shape (n, m)
  """
  assert A.shape[1] == B.shape[1]
  result = np.zeros(shape=(A.shape[0], B.shape[0]))

  if squared:
    for i in range(B.shape[0]):
      result[:, i] = np.sum(np.square(A - B[i, :]), axis=1)
  else:
    for i in range(B.shape[0]):
      result[:, i] = np.linalg.norm(A - B[i, :], axis=1)
  return result

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
    n_jobs = -2 # use all cpu's but one

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
    centers = np.empty([0,0])

  if method == 'random':
    cluster_centers = Y[np.random.choice(range(Y.shape[0]), k, replace=False)]

  # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
  elif method == 'distance':
    raise NotImplementedError #TODO

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

def check_for_noise(x_after, x_before, name='array'):
  """checks if noise has been added to x_after. The difference between both arrays is determined with a relative tolerance of 1e-10

    Args:
      x_after: an ndarray containing numericals
      x_before: an ndarray containing numericals

    Returns:
      true if all elements are equal or almost equal (relative tolerance 1e-10)

  """
  all_close = np.allclose(x_after, x_before, rtol=1e-10)
  if not all_close:
    logging.warning("%s is changed by the model before evaluation_runs. Possibly noise in test mode used.", name)
  return all_close

def mc_integration_cauchy(func, ndim, n_samples=10**7, batch_size=None):
  """ Monte carlo integration using importance sampling with a cauchy distribution

  Args:
    func: function to integrate over - must take numpy arrays of shape (n_samples, ndim) as first argument
          and return a numpy array of shape (n_samples, ndim_out)
    ndim: (int) number of dimensions to integrate over
    n_samples: (int) number of samples
    batch_size: (int) batch_size for junking the n_samples in batches (optional)

  Returns:
    approximated integral - numpy array of shape (ndim_out,)

  """
  if batch_size is None:
    n_batches = 1
    batch_size = n_samples
  else:
    n_batches = n_samples // batch_size + int(n_samples % batch_size > 0)

  batch_results = []
  for j in range(n_batches):
    samples = stats.cauchy.rvs(loc=0, scale=2, size=(batch_size,ndim))
    f = np.expand_dims(_multidim_cauchy_pdf(samples, loc=0, scale=2), axis=1)
    r = func(samples)
    assert r.ndim == 2, 'func must return a 2-dimensional numpy array'
    f = np.tile(f, (1, r.shape[1])) # bring f into same shape like r
    assert(f.shape == r.shape)
    batch_results.append(np.mean(r / f, axis=0))

  result = np.mean(np.stack(batch_results, axis=0), axis=0)
  return result

def _multidim_cauchy_pdf(x, loc=0, scale=1):
  """ multidimensional cauchy pdf """

  p = stats.cauchy.pdf(x, loc=loc, scale=scale)
  p = np.prod(p, axis=1).flatten()
  assert p.ndim == 1
  return p


def is_pos_def(M):
  """ checks whether x^T * M * x > 0, M being the matrix to be checked
  :param M: the matrix to be checked
  :return: True if positive definite, False otherwise
  """
  return np.all(np.linalg.eigvals(M) > 0)


def _project_to_pos_semi_def(M):
  return M.T.dot(M)


def project_to_pos_semi_def(M):
  """
  Projects a symmetric matrix M (norm) or a stack of symmetric matrices M onto the cone of pos. (semi) def. matrices
  :param M: Either M is a symmetric matrix of the form (m,m) or stack of k such matrices -> shape (k,m,m)
  :return: M, the projection of M or all projections of matrices in M on the cone pos. semi-def. matrices
  """
  assert M.ndim <= 3

  if M.ndim == 3:
    assert M.shape[1] == M.shape[2]
    for i in range(M.shape[0]):
      M[i] = _project_to_pos_semi_def(M[i])
  else:
    assert M.shape[0] == M.shape[1]
    M = _project_to_pos_semi_def(M)

  return M


def take(n, mydict):
  "Return first n items of the iterable as a list"
  return {k: mydict[k] for k in list(mydict)[:n]}


def take_of_type(n, type, mydict):
  d = {k: mydict[k] for k, v in mydict.items() if v.task_name.split('_')[0] == type}
  return take(n, d)