import unittest
from scipy.stats import norm
import warnings
import pickle
import sys
import os
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.density_estimator import MixtureDensityNetwork, KernelMixtureNetwork, \
  ConditionalKernelDensityEstimation, LSConditionalDensityEstimation, NeighborKernelDensityEstimation, NormalizingFlowEstimator

class TestConditionalDensityEstimators_2d_gaussian(unittest.TestCase):

  def get_samples(self, mu=2, std=1.0):
    np.random.seed(22)
    data = np.random.normal([mu, mu], std, size=(2000, 2))
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_NKDE_with_4d_gaussian(self):
    mu = 5
    std = 2.0
    X = np.random.normal(loc=mu, scale=std, size=(4000, 2))
    Y = np.random.normal(loc=mu, scale=std, size=(4000, 2))

    model = NeighborKernelDensityEstimation(epsilon=0.3)
    model.fit(X, Y)

    y = np.random.uniform(low=[1.0, 1.0], high=[9.0, 9.0], size=(500, 2))
    x = np.ones(shape=(500, 2)) * mu

    p_est = model.pdf(x, y)
    p_true = stats.multivariate_normal.pdf(y, mean=np.ones(2)*mu, cov=std**2)

    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_NKDE_with_2d_gaussian(self):
    X = np.random.uniform(-1, 1, size=4000)
    Y = (2 + X) * np.random.normal(size=4000) + 2*X

    for weighted in [True, False]:
      model = NeighborKernelDensityEstimation(epsilon=0.3, weighted=weighted)
      model.fit(X, Y)

      y = np.linspace(-5, 5, num=100)
      x = np.ones(100) * 0

      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=0, scale=2)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

      y = np.linspace(-5, 5, num=100)
      x = - np.ones(100) * 0.5

      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=-1, scale=1.5)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_NKDE_loo_log_likelihood(self):
    mu = 1
    std = 1
    X = np.random.normal(loc=mu, scale=std, size=(500, 2))
    Y = np.random.normal(loc=mu, scale=std, size=(500, 2))

    model = NeighborKernelDensityEstimation(epsilon=0.3)
    model.fit(X, Y)
    bw = np.array([0.5])
    epsilon = 0.3
    ll1 = model.loo_likelihood(bandwidth=bw, epsilon=epsilon)

    bw = np.array([0.05])
    epsilon = 0.1
    ll2 = model.loo_likelihood(bandwidth=bw, epsilon=epsilon)
    self.assertGreater(ll1, ll2)

  def test_NKDE_param_selection(self):
    mu = 5
    std = 2
    X = np.random.normal(loc=mu, scale=std, size=(500, 2))
    Y = np.random.normal(loc=mu, scale=std, size=(500, 2))

    model1 = NeighborKernelDensityEstimation('NKDE', 2, 2, epsilon=0.1, bandwidth=0.3, param_selection=None)
    model1.fit(X, Y)
    self.assertAlmostEqual(model1.epsilon, 0.1)

    model2 = NeighborKernelDensityEstimation('NKDE', 2, 2, epsilon=0.1, bandwidth=0.3, param_selection='normal_reference')
    model2.fit(X, Y)
    self.assertAlmostEqual(model1.epsilon, 0.1)
    self.assertGreaterEqual(np.mean(model2.bandwidth - model1.bandwidth), 0.0)

    model3 = NeighborKernelDensityEstimation('NKDE', 2, 2, epsilon=0.1, bandwidth=0.3, param_selection='cv_ml')
    model3.fit(X, Y)
    self.assertNotEqual(model3.epsilon, 0.1)

    X_test, Y_test = np.random.normal(loc=mu, scale=std, size=(2000, 2)), np.random.normal(loc=mu, scale=std, size=(2000, 2))
    score1 = model1.score(X_test, Y_test)
    score2 = model2.score(X_test, Y_test)
    score3 = model3.score(X_test, Y_test)
    print(score1, score2, score3)

    self.assertGreaterEqual(score2, score1)
    self.assertGreaterEqual(score3, score2 - 0.1)

  def test_LSCD_with_4d_gaussian(self):
    mu = 5
    std = 2.0
    X = np.random.normal(loc=mu, scale=std, size=(4000, 2))
    Y = np.random.normal(loc=mu, scale=std, size=(4000, 2))

    for method in ["all", "k_means"]:
      model = LSConditionalDensityEstimation(center_sampling_method=method)
      model.fit(X, Y)

      y = np.random.uniform(low=[1.0, 1.0], high=[9.0, 9.0], size=(500, 2))
      x = np.ones(shape=(500, 2)) * mu

      p_est = model.pdf(x, y)
      p_true = stats.multivariate_normal.pdf(y, mean=np.ones(2)*mu, cov=std**2)

      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_LSCD_with_2d_gaussian(self):
    X = np.random.uniform(-1, 1, size=4000)
    Y = (2 + X) * np.random.normal(size=4000) + 2*X

    model = LSConditionalDensityEstimation()
    model.fit(X, Y)

    y = np.linspace(-5, 5, num=100)
    x = np.ones(100) * 0

    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=0, scale=2)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    y = np.linspace(-5, 5, num=100)
    x = - np.ones(100) * 0.5

    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=-1, scale=1.5)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian(self):
    mu = -2.0
    std = 2.0
    X, Y = self.get_samples(mu=mu, std=std)

    for method in ["agglomerative"]:
      model = KernelMixtureNetwork("kmn_"+method, 1, 1, center_sampling_method=method, n_centers=20,
                                   hidden_sizes=(16, 16), init_scales=np.array([0.5]), train_scales=True,
                                   data_normalization=False)
      model.fit(X, Y)

      y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
      x = np.asarray([mu for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=mu, scale=std)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

      p_est = model.cdf(x, y)
      p_true = norm.cdf(y, loc=mu, scale=std)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian_2(self):
    mu = 200
    std = 23
    X, Y = self.get_samples(mu=mu, std=std)

    for method in ["agglomerative"]:
      model = KernelMixtureNetwork("kmn2_" + method, 1, 1, center_sampling_method=method, n_centers=10,
                                   hidden_sizes=(16, 16), init_scales=np.array([1.0]), train_scales=True,
                                   data_normalization=True)
      model.fit(X, Y)

      y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
      x = np.asarray([mu for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=mu, scale=std)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

      p_est = model.cdf(x, y)
      p_true = norm.cdf(y, loc=mu, scale=std)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian_sampling(self):
    np.random.seed(22)
    X, Y = self.get_samples(mu=5)

    import time
    t = time.time()
    model = KernelMixtureNetwork("kmn_sampling", 1, 1, center_sampling_method='k_means', n_centers=5,
                                 n_training_epochs=1000, data_normalization=True)
    print("time to build model:", time.time() - t)
    t = time.time()

    model.fit(X, Y)
    print("time to fit model:", time.time() - t)

    x_cond = 5 * np.ones(shape=(2000000,1))
    _, y_sample = model.sample(x_cond)
    print(np.mean(y_sample), np.std(y_sample))
    self.assertAlmostEqual(np.mean(y_sample), model.mean_(x_cond[1]).item(), places=1)
    self.assertAlmostEqual(np.std(y_sample), model.covariance(x_cond[1]).item(), places=1)

    x_cond = np.ones(shape=(400000, 1))
    x_cond[0,0] = 5.0
    _, y_sample = model.sample(x_cond)
    self.assertAlmostEqual(np.mean(y_sample), model.mean_(x_cond[1]).item(), places=1)
    self.assertAlmostEqual(np.std(y_sample), np.sqrt(model.covariance(x_cond[1])).item(), places=1)

  def test_MDN_with_2d_gaussian_sampling(self):
    X, Y = self.get_samples()

    model = MixtureDensityNetwork("mdn_gaussian_sampling", 1, 1, n_centers=5, n_training_epochs=200)
    model.fit(X, Y)

    x_cond = np.ones(shape=(10**6,1))
    _, y_sample = model.sample(x_cond)
    self.assertAlmostEqual(np.mean(y_sample), model.mean_(y_sample[1]).item(), places=0)
    self.assertAlmostEqual(np.std(y_sample), model.covariance(y_sample[1]).item(), places=0)

  def test_MDN_with_2d_gaussian(self):
    mu = 200
    std = 23
    X, Y = self.get_samples(mu=mu, std=std)

    model = MixtureDensityNetwork("mdn", 1, 1, n_centers=10, data_normalization=True)
    model.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = model.cdf(x, y)
    p_true = norm.cdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


  def test_MDN_with_2d_gaussian2(self):
    mu = -5
    std = 2.5
    X, Y = self.get_samples(mu=mu, std=std)

    model = MixtureDensityNetwork("mdn2", 1, 1, n_centers=5, weight_normalization=True)
    model.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = model.cdf(x, y)
    p_true = norm.cdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_CDE_with_2d_gaussian(self):
    X, Y = self.get_samples()

    model = ConditionalKernelDensityEstimation('cde', 1, 1)
    model.fit(X, Y)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = model.cdf(x, y)
    p_true = norm.cdf(y, loc=2, scale=1)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_MDN_KMN_eval_set(self):
    mu = 200
    std = 23
    X_train, Y_train = self.get_samples(mu=mu, std=std)
    X_test, Y_test = self.get_samples(mu=mu, std=std)
    X_test = X_test

    model = MixtureDensityNetwork("mdn_eval_set", 1, 1, n_centers=10, data_normalization=True, n_training_epochs=100)
    model.fit(X_train, Y_train, eval_set=(X_test, Y_test))

    model = KernelMixtureNetwork("kmn_eval_set", 1, 1, n_centers=10, data_normalization=True, n_training_epochs=100)
    model.fit(X_train, Y_train, eval_set=(X_test, Y_test))

class TestSerializationDensityEstimators(unittest.TestCase):

  def get_samples(self, std=1.0):
    np.random.seed(22)
    data = np.random.normal([2, 2, 2, 2], std, size=(2000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]
    return X, Y

  def testPickleUnpickleMDN(self):
    X, Y = self.get_samples()
    model = MixtureDensityNetwork("mdn_pickle", 2, 2, n_training_epochs=10, data_normalization=True,  weight_normalization=False)
    model.fit(X, Y)
    pdf_before = model.pdf(X, Y)

    # pickle and unpickle model
    dump_string = pickle.dumps(model)
    model_loaded = pickle.loads(dump_string)
    pdf_after = model_loaded.pdf(X, Y)

    diff = np.sum(np.abs(pdf_after - pdf_before))
    self.assertAlmostEqual(diff, 0, places=2)

  def testPickleUnpickleKDN(self):
    X, Y = self.get_samples()
    model = KernelMixtureNetwork("kde", 2, 2, n_centers=10, n_training_epochs=10, data_normalization=True,  weight_normalization=True)
    model.fit(X, Y)
    pdf_before = model.pdf(X, Y)

    # pickle and unpickle model
    dump_string = pickle.dumps(model)
    model_loaded = pickle.loads(dump_string)
    pdf_after = model_loaded.pdf(X, Y)

    diff = np.sum(np.abs(pdf_after - pdf_before))
    self.assertAlmostEqual(diff, 0, places=2)

class TestRegularization(unittest.TestCase):

  def get_samples(self, std=1.0, mu=2, n_samples=2000):
    np.random.seed(22)
    data = np.random.normal([mu, mu], std, size=(n_samples, 2))
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test1_KMN_with_2d_gaussian_noise_y(self):
    X, Y = self.get_samples(std=0.5)

    model_no_noise = KernelMixtureNetwork("kmn_no_noise_y", 1, 1, n_centers=5, x_noise_std=None, y_noise_std=None)
    model_no_noise.fit(X, Y)
    var_no_noise = model_no_noise.covariance(x_cond=np.array([[2]]))[0][0][0]

    model_noise = KernelMixtureNetwork("kmn_noise_y", 1, 1, n_centers=5, x_noise_std=None, y_noise_std=1)
    model_noise.fit(X, Y)
    var_noise = model_noise.covariance(x_cond=np.array([[2]]))[0][0][0]

    print("Training w/o noise:", var_no_noise)
    print("Training w/ noise:", var_noise)

    self.assertGreaterEqual(var_noise - var_no_noise, 0.1)

  def test2_KMN_with_2d_gaussian_noise_x(self):
    np.random.seed(22)
    X = np.random.uniform(0, 6, size=4000)
    Y = X + np.random.normal(0, 1, size=4000)

    x_test_2 = np.ones(100) * 2
    x_test_4 = np.ones(100) * 4
    y_test = np.linspace(1, 5, num=100)

    model_no_noise = KernelMixtureNetwork("kmn_no_noise_x", 1, 1, n_centers=5, x_noise_std=None, y_noise_std=None)
    model_no_noise.fit(X, Y)
    pdf_distance_no_noise = np.mean(np.abs(model_no_noise.pdf(x_test_2, y_test) - model_no_noise.pdf(x_test_4, y_test)))

    model_noise = KernelMixtureNetwork("kmn_noise_x", 1, 1, n_centers=5, x_noise_std=2, y_noise_std=None)
    model_noise.fit(X, Y)
    pdf_distance_noise = np.mean(np.abs(model_noise.pdf(x_test_2, y_test) - model_noise.pdf(x_test_4, y_test)))

    print("Training w/o noise - pdf distance:", pdf_distance_no_noise)
    print("Training w/ noise - pdf distance", pdf_distance_noise)

    self.assertGreaterEqual(pdf_distance_no_noise / pdf_distance_noise, 2.0)

  def test3_MDN_with_2d_gaussian_noise_y(self):
    X, Y = self.get_samples(std=0.5)

    model_no_noise = MixtureDensityNetwork("mdn_no_noise_y", 1, 1, n_centers=1, x_noise_std=None, y_noise_std=None)
    model_no_noise.fit(X, Y)
    var_no_noise = model_no_noise.covariance(x_cond=np.array([[2]]))[0][0][0]

    model_noise = MixtureDensityNetwork("mdn_noise_y", 1, 1, n_centers=1, x_noise_std=None, y_noise_std=1)
    model_noise.fit(X, Y)
    var_noise = model_noise.covariance(x_cond=np.array([[2]]))[0][0][0]

    print("Training w/o noise:", var_no_noise)
    print("Training w/ noise:", var_noise)

    self.assertGreaterEqual(var_noise - var_no_noise, 0.1)

  def test4_MDN_with_2d_gaussian_noise_x(self):
    np.random.seed(22)
    X = np.random.uniform(0, 6, size=4000)
    Y = X + np.random.normal(0,1, size=4000)

    x_test_2 = np.ones(100) * 2
    x_test_4 = np.ones(100) * 4
    y_test = np.linspace(1,5,num=100)

    model_no_noise = MixtureDensityNetwork("mdn_no_noise_x", 1, 1, n_centers=1, x_noise_std=None, y_noise_std=None)
    model_no_noise.fit(X, Y)
    pdf_distance_no_noise = np.mean(np.abs(model_no_noise.pdf(x_test_2, y_test) - model_no_noise.pdf(x_test_4, y_test)))

    model_noise = MixtureDensityNetwork("mdn_noise_x", 1, 1, n_centers=1, x_noise_std=2, y_noise_std=None)
    model_noise.fit(X, Y)
    pdf_distance_noise = np.mean(np.abs(model_noise.pdf(x_test_2, y_test) - model_noise.pdf(x_test_4, y_test)))

    print("Training w/o noise - pdf distance:", pdf_distance_no_noise)
    print("Training w/ noise - pdf distance", pdf_distance_noise)

    self.assertGreaterEqual(pdf_distance_no_noise/pdf_distance_noise, 2.0)

  def test7_data_normalization(self):
    X, Y = self.get_samples(std=2, mu=20)
    model = KernelMixtureNetwork("kmn_data_normalization", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                        data_normalization=True, n_training_epochs=100)
    model.fit(X, Y)

    # test if data statistics were properly assigned to tf graph
    x_mean, x_std = model.x_mean, model.x_std
    print(x_mean, x_std)
    mean_diff = np.abs(x_mean - 20).item()
    std_diff = np.abs(x_std - 2).item()
    self.assertLessEqual(mean_diff, 0.5)
    self.assertLessEqual(std_diff, 0.5)

  def test8_data_normalization(self):
    np.random.seed(24)
    mean = 10
    std = 2
    data = np.random.normal([mean, mean, mean, mean], std, size=(2000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = MixtureDensityNetwork("mdn_data_normalization", 2, 2, n_centers=2, x_noise_std=None, y_noise_std=None,
                                        data_normalization=True, n_training_epochs=2000, random_seed=22)
    model.fit(X, Y)
    y_mean, y_std = model.y_mean, model.y_std
    print(y_mean, y_std)
    cond_mean = model.mean_(Y)
    mean_diff = np.abs(mean-np.mean(cond_mean))
    self.assertLessEqual(mean_diff, 0.5)

    cond_cov = np.mean(model.covariance(Y), axis=0)
    print(cond_cov)
    self.assertGreaterEqual(cond_cov[0][0], std**2 * 0.7)
    self.assertLessEqual(cond_cov[0][0], std**2 * 1.3)
    self.assertGreaterEqual(cond_cov[1][1], std**2 * 0.7)
    self.assertLessEqual(cond_cov[1][1], std**2 * 1.3)

  def test9_data_normalization(self):
    np.random.seed(24)
    mean = -80
    std = 7
    data = np.random.normal([mean, mean, mean, mean], std, size=(4000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    model = KernelMixtureNetwork("kmn_data_normalization_2", 2, 2, n_centers=5, x_noise_std=None, y_noise_std=None,
                                  data_normalization=True, n_training_epochs=2000, random_seed=22, keep_edges=False,
                                 train_scales=True, weight_normalization=True, init_scales=np.array([1.0]))

    model.fit(X, Y)
    cond_mean = model.mean_(Y)
    print(np.mean(cond_mean))
    mean_diff = np.abs(mean - np.mean(cond_mean))
    self.assertLessEqual(mean_diff, np.abs(mean) * 0.1)

    cond_cov = np.mean(model.covariance(Y), axis=0)
    print(cond_cov)
    self.assertGreaterEqual(cond_cov[0][0], std**2 * 0.7)
    self.assertLessEqual(cond_cov[0][0], std**2 * 1.3)
    self.assertGreaterEqual(cond_cov[1][1], std**2 * 0.7)
    self.assertLessEqual(cond_cov[1][1], std**2 * 1.3)

  def test_MDN_adaptive_noise(self):
    adaptive_noise_fn = lambda n, d: 0.0 if n < 1000 else 5.0

    X, Y = self.get_samples(mu=0, std=1, n_samples=999)
    est = MixtureDensityNetwork("mdn_adaptive_noise_999", 1, 1, n_centers=1, y_noise_std=0.0, x_noise_std=0.0,
                                hidden_sizes=(8, 8),
                                adaptive_noise_fn=adaptive_noise_fn, n_training_epochs=500)
    est.fit(X, Y)
    std_999 = est.std_(x_cond=np.array([[0.0]]))[0].item()

    X, Y = self.get_samples(mu=0, std=1, n_samples=1002)
    est = MixtureDensityNetwork("mdn_adaptive_noise_1002", 1, 1, n_centers=1, y_noise_std=0.0, x_noise_std=0.0,
                                hidden_sizes=(8, 8),
                                adaptive_noise_fn=adaptive_noise_fn, n_training_epochs=500)
    est.fit(X, Y)
    std_1002 = est.std_(x_cond=np.array([[0.0]]))[0].item()

    self.assertLess(std_999, std_1002)
    self.assertGreater(std_1002, 2)

  def test_NF_adaptive_noise(self):
    adaptive_noise_fn = lambda n, d: 0.0 if n < 1000 else 5.0

    X, Y = self.get_samples(mu=0, std=1, n_samples=999)
    est = NormalizingFlowEstimator("nf_999", 1, 1, y_noise_std=0.0, n_flows=2, hidden_sizes=(8,8),
                                   x_noise_std=0.0, adaptive_noise_fn=adaptive_noise_fn,
                                   n_training_epochs=500)
    est.fit(X, Y)
    std_999 = est.std_(x_cond=np.array([[0.0]]))[0].item()

    X, Y = self.get_samples(mu=0, std=1, n_samples=1002)
    est = NormalizingFlowEstimator("nf_1002", 1, 1, y_noise_std=0.0, n_flows=2,  hidden_sizes=(8,8),
                                   x_noise_std=0.0, adaptive_noise_fn=adaptive_noise_fn,
                                   n_training_epochs=500)
    est.fit(X, Y)
    std_1002 = est.std_(x_cond=np.array([[0.0]]))[0].item()

    self.assertLess(std_999, std_1002)
    self.assertGreater(std_1002, 2)

  def test_MDN_weight_decay(self):
    mu = 5
    std = 5
    X, Y = self.get_samples(mu=mu, std=std)

    no_decay = MixtureDensityNetwork("mdn_no_weight_decay", 1, 1, hidden_sizes=(32, 32), n_centers=10,
                                     n_training_epochs=2000, weight_decay=0.0, weight_normalization=False)
    decay = MixtureDensityNetwork("mdn_weight_decay", 1, 1, n_centers=10,  hidden_sizes=(32, 32),
                                  n_training_epochs=2000, weight_decay=1e-3, weight_normalization=False)
    no_decay.fit(X, Y)
    decay.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_true = norm.pdf(y, loc=mu, scale=std)
    l1_err_no_dec = np.mean(np.abs(no_decay.pdf(x, y) - p_true))
    l1_err_dec = np.mean(np.abs(decay.pdf(x, y) - p_true))

    self.assertLessEqual(l1_err_dec, 0.1)
    self.assertLessEqual(l1_err_dec, l1_err_no_dec)

  def test_KMN_l2_regularization(self):
    mu = 5
    std = 5
    X, Y = self.get_samples(mu=mu, std=std, n_samples=500)

    kmn_no_reg = KernelMixtureNetwork("kmn_no_reg", 1, 1, n_centers=10,
                                     n_training_epochs=200, l2_reg=0.0, weight_normalization=False)
    kmn_reg_l2 = KernelMixtureNetwork("kmn_reg_l2", 1, 1, n_centers=10, hidden_sizes=(16, 16),
                                  n_training_epochs=200, l2_reg=1.0, weight_normalization=False)
    kmn_no_reg.fit(X, Y)
    kmn_reg_l2.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_true = norm.pdf(y, loc=mu, scale=std)
    err_no_reg = np.mean(np.abs(kmn_no_reg.pdf(x, y) - p_true))
    err_reg_l2 = np.mean(np.abs(kmn_reg_l2.pdf(x, y) - p_true))

    self.assertLessEqual(err_reg_l2, err_no_reg + 1e-2)

  def test_NF_l1_regularization(self):
    mu = 5
    std = 5
    X, Y = self.get_samples(mu=mu, std=std, n_samples=500)

    nf_no_reg = NormalizingFlowEstimator("nf_no_reg", 1, 1, hidden_sizes=(16, 16), n_flows=10,
                                         weight_normalization=False, l1_reg=0.0, l2_reg=0.0, n_training_epochs=500)
    nf_reg_l1 = NormalizingFlowEstimator("nf_reg_l1", 1, 1, hidden_sizes=(16, 16), n_flows=10,
                                         weight_normalization=False, l1_reg=10.0, l2_reg=0.0, n_training_epochs=500)
    nf_no_reg.fit(X, Y)
    nf_reg_l1.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_true = norm.pdf(y, loc=mu, scale=std)
    err_no_reg = np.mean(np.abs(nf_no_reg.pdf(x, y) - p_true))
    err_reg_l1 = np.mean(np.abs(nf_reg_l1.pdf(x, y) - p_true))

    print(err_no_reg, err_reg_l1)

    self.assertLessEqual(err_reg_l1, err_no_reg)

  def test_MDN_dropout(self):
    mu = -8
    std = 2.5
    X, Y = self.get_samples(mu=mu, std=std)

    dropout_model = MixtureDensityNetwork("mdn_dropout_reasonable", 1, 1, n_centers=5, weight_normalization=True,
                                          dropout=0.5, n_training_epochs=400)
    dropout_model.fit(X, Y)

    y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
    x = np.asarray([mu for i in range(y.shape[0])])
    p_est = dropout_model.pdf(x, y)
    p_true = norm.pdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    p_est = dropout_model.cdf(x, y)
    p_true = norm.cdf(y, loc=mu, scale=std)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

class TestLogProbability(unittest.TestCase):

  def test_KMN_log_pdf(self):
    X, Y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 2))

    for data_norm in [True, False]:
      model = KernelMixtureNetwork("kmn_logprob"+str(data_norm), 3, 2, n_centers=5,
                                   hidden_sizes=(8, 8), init_scales=np.array([0.5]), n_training_epochs=10, data_normalization=data_norm)
      model.fit(X, Y)

      x, y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 2))
      prob = model.pdf(x,y)
      log_prob = model.log_pdf(x,y)
      self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)

  def test_MDN_log_pdf(self):
    X, Y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 2))

    for data_norm in [True, False]:
      model = MixtureDensityNetwork("mdn_logprob"+str(data_norm), 3, 2, n_centers=1,
                                   hidden_sizes=(8, 8), n_training_epochs=10, data_normalization=data_norm)
      model.fit(X, Y)

      x, y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 2))
      prob = model.pdf(x,y)
      log_prob = model.log_pdf(x,y)
      self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)

  def test_CKDE_log_pdf(self):
    X, Y = np.random.normal(size=(500, 2)), np.random.normal(size=(500, 2))

    model = ConditionalKernelDensityEstimation(bandwidth='normal_reference')
    model.fit(X, Y)

    x, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2))
    prob = model.pdf(x, y)
    log_prob = model.log_pdf(x, y)
    self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)

  def test_NKDE_log_pdf(self):
    X, Y = np.random.normal(size=(500, 2)), np.random.normal(size=(500, 2))

    model = NeighborKernelDensityEstimation()
    model.fit(X, Y)

    x, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2))
    prob = model.pdf(x, y)
    log_prob = model.log_pdf(x, y)
    self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)

  def test_LSCDE_log_pdf(self):
    X, Y = np.random.normal(size=(500, 2)), np.random.normal(size=(500, 2))

    model = LSConditionalDensityEstimation()
    model.fit(X, Y)

    x, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2))
    prob = model.pdf(x, y)
    log_prob = model.log_pdf(x, y)
    self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)

class TestConditionalDensityEstimators_fit_by_crossval(unittest.TestCase):
  def get_samples(self):
    np.random.seed(22)
    data = np.concatenate([np.random.normal([i, -i], 1, size=(500, 2)) for i in range(-20, 20, 4)], axis=0)
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_1_KMN_with_2d_gaussian_fit_by_crossval(self):
    X, Y = self.get_samples()

    param_grid = {
      "n_centers": [3, 10],
      "center_sampling_method": ["k_means"],
      "keep_edges": [True]
    }

    model = KernelMixtureNetwork(center_sampling_method="k_means", n_centers=20)
    model.fit_by_cv(X, Y, param_grid=param_grid)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertEqual(model.get_params()["n_centers"], 3)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)


  def test_2_MDN_with_2d_gaussian_fit_by_crossval(self):
    X, Y = self.get_samples()

    param_grid = {
      "n_centers": [2, 10, 50]
    }

    model = MixtureDensityNetwork()
    model.fit_by_cv(X, Y, param_grid=param_grid)

    y = np.arange(-1, 5, 0.5)
    x = np.asarray([2 for i in range(y.shape[0])])
    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=2, scale=1)
    self.assertEqual(model.get_params()["n_centers"], 50)
    self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)

if __name__ == '__main__':
  warnings.filterwarnings("ignore")

  testmodules = [
   'unittests_estimators.TestConditionalDensityEstimators_2d_gaussian',
   'unittests_estimators.TestRegularization',
   'unittests_estimators.TestSerializationDensityEstimators',
   'unittests_estimators.TestLogProbability'
   ]
  suite = unittest.TestSuite()
  for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

  unittest.TextTestRunner().run(suite)
