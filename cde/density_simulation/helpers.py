import numpy as np

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