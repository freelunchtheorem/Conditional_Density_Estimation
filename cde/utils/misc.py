import numpy as np

def norm_along_axis_1(A, B, squared=False, norm_dim=False):
    """ calculates the (squared) euclidean distance along the axis 1 of both 2d arrays

    Args:
      A: numpy array of shape (n, k)
      B: numpy array of shape (m, k)
      squared: boolean that indicates whether the squared euclidean distance shall be returned, \
               otherwise the euclidean distance is returned
      norm_dim: (boolean) normalized the distance by the dimensionality k -> divides result by sqrt(k)

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

    if norm_dim:
        result = result / np.sqrt(A.shape[1])
    return result


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