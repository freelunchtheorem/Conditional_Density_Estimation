import numpy as np
import warnings


__all__ = ["find_root_newton_method", "find_root_by_bounding"]


class _AdamRootOptimizer:
    """Lightweight Adam-style updater used internally by the root finder."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0
        self.m = None
        self.v = None

    def step(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)

        lr = self.lr
        if self.decay > 0:
            lr /= (1.0 + self.decay * self.iterations)

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1.0 - self.beta2 ** t) / (1.0 - self.beta1 ** t))

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.square(grad)

        delta = lr_t * self.m / (np.sqrt(self.v) + self.epsilon)

        self.iterations = t
        return param - delta


def find_root_newton_method(fun, grad, x0, eps=1e-6, learning_rate=2e-3, max_iter=1e5):
    """Solve f(x)=0 via Newton steps with Adam-style step sizes."""

    assert callable(fun)
    assert callable(grad)

    optimizer = _AdamRootOptimizer(lr=learning_rate)
    x = x0
    percentage_roots_found = 0.0
    n_iter = 0

    while percentage_roots_found < 0.999:
        f = fun(x)
        g = grad(x)
        newton_step = (f + 1e-10) / (g + 1e-10)
        newton_step = np.clip(newton_step, -1000, 1000)
        x = optimizer.step(x, newton_step)

        approx_error = np.abs(f)
        percentage_roots_found = np.mean(approx_error < eps)
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping newton method for determining quantiles")
            return np.NaN

    return x


def find_root_by_bounding(fun, left, right, eps=1e-8, max_iter=1e4):
    """Find a root by shrinking the bounding interval until it collapses."""

    assert callable(fun)

    n_iter = 0
    approx_error = 1e10

    while approx_error > eps:
        middle = (right + left) / 2
        f = fun(middle)

        left_of_zero = (f < 0).flatten()
        left[left_of_zero] = middle[left_of_zero]
        right[np.logical_not(left_of_zero)] = middle[np.logical_not(left_of_zero)]

        assert np.all(left <= right)

        approx_error = np.mean(np.abs(right - left)) / 2
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping Newton method for determining quantiles")
            return np.NaN

    return middle

