import numpy as np
import warnings

class AdamOptimizer:
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):

        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_update(self, params, grads):
        """ params and grads are list of numpy arrays
        """
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]

        """ #TODO: implement clipping
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        """

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t

        self.iterations += 1

        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])


        return ret

def find_root_newton_method(fun, grad, x0, eps=1e-6, learning_rate=2e-3, max_iter=1e5):
    """
    Newton's root finding method in conjunction with the adam optimizer

    Args:
        fun (callable): function f for which f(x) = 0 shall be solved
        grad (callable): gradient of f
        x0 (np.ndarray): initial value
        eps (float): tolerance
        learning_rate (float): learning rate of the optimizer
        max_iter (int): maximum iterations

    Returns:
        numpy array (result of the root finding) - if not successful, returns np.NaN
    """
    assert callable(fun)
    assert callable(grad)

    optimizer = AdamOptimizer(lr=learning_rate)

    x = x0
    n_iter = 0
    percentage_roots_found = 0.0

    while percentage_roots_found < 0.999:
        f = fun(x)
        g = grad(x)
        newton_step = (f + 1e-10)/(g + 1e-10) #
        newton_step = np.clip(newton_step, -1000, 1000)  # gradient clipping
        x = optimizer.get_update([x], [newton_step])[0]

        approx_error = np.abs(f)
        percentage_roots_found = np.mean(approx_error < eps)
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping newton method for determining quantiles")
            return np.NaN

    return x


def find_root_by_bounding(fun, left, right, eps=1e-8, max_iter=1e4):
    """
    Root finding method that uses selective shrinking of a target interval bounded by left and right
    --> other than the newton method, this method only works for for vectorized univariate functions
    Args:
        fun (callable): function f for which f(x) = 0 shall be solved
        left: (np.ndarray): initial left bound
        right (np.ndarray): initial right bound
        eps (float): tolerance
        max_iter (int): maximum iterations
    """

    assert callable(fun)

    n_iter = 0
    approx_error = 1e10
    while approx_error > eps:
        middle = (right + left)/2
        f = fun(middle)

        left_of_zero = (f < 0).flatten()
        left[left_of_zero] = middle[left_of_zero]
        right[np.logical_not(left_of_zero)] = middle[np.logical_not(left_of_zero)]

        assert np.all(left <= right)

        approx_error = np.mean(np.abs(right-left))/2
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping newton method for determining quantiles")
            return np.NaN

    return middle