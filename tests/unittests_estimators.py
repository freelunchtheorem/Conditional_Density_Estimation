import unittest
from scipy.stats import norm
import warnings
import pickle
import tensorflow as tf
import sys
import os
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.helpers import sample_center_points, mc_integration_cauchy, norm_along_axis_1
from cde.density_estimator import MixtureDensityNetwork, KernelMixtureNetwork, \
  ConditionalKernelDensityEstimation, LSConditionalDensityEstimation, NeighborKernelDensityEstimation

class TestHelpers(unittest.TestCase):

  """ sample center points """

  def test_1_shape_center_point(self):
    methods = ["all", "random", "k_means" , "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(120,2))
      centers = sample_center_points(Y, method=m, k=50)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  def test_1_shape_center_point_k_means(self):
    Y = np.asarray([1.0, 2.0])
    centers = sample_center_points(Y, method="k_means", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_1_shape_center_point_agglomerative(self):
    Y = np.random.uniform(size=[20,3])
    centers = sample_center_points(Y, method="agglomerative", k=1)
    self.assertAlmostEqual(Y.mean(), centers.mean())

  def test_1_shape_center_point_keep_edges(self):
    methods = ["random", "k_means", "agglomerative"]
    for m in methods:
      Y = np.random.uniform(size=(100,2))
      centers = sample_center_points(Y, method=m, k=5, keep_edges=True)
      self.assertEqual(centers.ndim, Y.ndim)
      self.assertEqual(centers.shape[1], Y.shape[1])

  """ norm along axis """

  def test_2_norm_along_axis_1(self):
    A = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    B = np.asarray([[0.0,0.0], [0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_2_norm_along_axis_2(self):
    A = np.asarray([[1.0, 0.0]])
    B = np.asarray([[0.0,0.0]])
    dist1 = norm_along_axis_1(A, B, squared=True)
    dist2 = norm_along_axis_1(A, B, squared=False)
    self.assertEqual(np.mean(dist1), 1.0)
    self.assertEqual(np.mean(dist2), 1.0)

  def test_2_norm_along_axis_3(self):
    A = np.random.uniform(size=[20, 3])
    B = np.random.uniform(size=[10, 3])
    dist = norm_along_axis_1(A, B, squared=True)
    self.assertEqual(dist.shape, (20,10))

  """ monte carlo integration """

  def test_mc_integration_chauchy_1(self):
    func = lambda y: np.expand_dims(stats.multivariate_normal.pdf(y, mean=[0, 0], cov=np.diag([2, 2])), axis=1)
    integral = mc_integration_cauchy(func, ndim=2, n_samples=10 ** 7, batch_size=10**6)
    self.assertAlmostEqual(1.0, integral[0], places=2)

  def test_mc_integration_chauchy_2(self):
    func = lambda y: y * np.tile(np.expand_dims(stats.multivariate_normal.pdf(y, mean=[1, 2], cov=np.diag([2, 2])), axis=1), (1,2))
    integral = mc_integration_cauchy(func, ndim=2, n_samples=10 ** 7, batch_size=10**6)
    self.assertAlmostEqual(1, integral[0], places=2)
    self.assertAlmostEqual(2, integral[1], places=2)

class TestConditionalDensityEstimators_2d_gaussian(unittest.TestCase):

  def get_samples(self, mu=2, std=1.0):
    np.random.seed(22)
    data = np.random.normal([mu, mu], std, size=(2000, 2))
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y

  def test_NKDE_with_2d_gaussian(self):
    mu = 5
    std = 2.0
    X = np.random.normal(loc=mu, scale=std, size=(4000, 2))
    Y = np.random.normal(loc=mu, scale=std, size=(4000, 2))

    model = NeighborKernelDensityEstimation(epsilon=0.3)
    model.fit(X, Y)

    y = np.random.uniform(low=[1.0, 1.0], high=[9.0, 9.0], size=(500, 2))
    x = np.ones(shape=(500,2)) * mu

    p_est = model.pdf(x, y)
    p_true = norm.pdf(y, loc=mu, scale=std)

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

  def test_LSCD_with_2d_gaussian(self):
    X, Y = self.get_samples()

    for method in ["all", "k_means"]:
      model = LSConditionalDensityEstimation(center_sampling_method=method, n_centers=400, bandwidth=0.5)
      model.fit(X, Y)

      y = np.arange(-1, 5, 0.5)
      x = np.asarray([2 for i in range(y.shape[0])])
      p_est = model.pdf(x, y)
      p_true = norm.pdf(y, loc=2, scale=1)
      self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

  def test_KMN_with_2d_gaussian(self):
    mu = -2.0
    std = 2.0
    X, Y = self.get_samples(mu=mu, std=std)

    for method in ["agglomerative"]:
      with tf.Session() as sess:
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
      with tf.Session() as sess:
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
                                 n_training_epochs=500, data_normalization=False)
    print("time to build model:", time.time() - t)
    t = time.time()

    model.fit(X, Y)
    print("time to fit model:", time.time() - t)

    x_cond = 5 * np.ones(shape=(2000000,1))
    _, y_sample = model.sample(x_cond)
    print(np.mean(y_sample), np.std(y_sample))
    self.assertAlmostEqual(np.mean(y_sample), float(model.mean_(x_cond[1])), places=1)
    self.assertAlmostEqual(np.std(y_sample), float(model.covariance(x_cond[1])), places=1)

    x_cond = np.ones(shape=(400000, 1))
    x_cond[0,0] = 5.0
    _, y_sample = model.sample(x_cond)
    self.assertAlmostEqual(np.mean(y_sample), float(model.mean_(x_cond[1])), places=1)
    self.assertAlmostEqual(np.std(y_sample), float(np.sqrt(model.covariance(x_cond[1]))), places=1)

  def test_MDN_with_2d_gaussian_sampling(self):
    X, Y = self.get_samples()

    model = MixtureDensityNetwork("mdn_gaussian_sampling", 1, 1, n_centers=5, n_training_epochs=200)
    model.fit(X, Y)

    x_cond = np.ones(shape=(10**6,1))
    _, y_sample = model.sample(x_cond)
    self.assertAlmostEqual(np.mean(y_sample), float(model.mean_(y_sample[1])), places=0)
    self.assertAlmostEqual(np.std(y_sample), float(model.covariance(y_sample[1])), places=0)

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

class TestSerializationDensityEstimators(unittest.TestCase):

  def get_samples(self, std=1.0):
    np.random.seed(22)
    data = np.random.normal([2, 2, 2, 2], std, size=(2000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]
    return X, Y

  def testPickleUnpickleMDN(self):
    X, Y = self.get_samples()
    with tf.Session() as sess:
      model = MixtureDensityNetwork("mdn_pickle", 2, 2, n_training_epochs=10, data_normalization=True,  weight_normalization=False)
      model.fit(X, Y)
      pdf_before = model.pdf(X, Y)

      # pickle and unpickle model
      dump_string = pickle.dumps(model)
    tf.reset_default_graph()
    with tf.Session() as sess:
      model_loaded = pickle.loads(dump_string)
      pdf_after = model_loaded.pdf(X, Y)

    diff = np.sum(np.abs(pdf_after - pdf_before))
    self.assertAlmostEqual(diff, 0, places=2)

  def testPickleUnpickleKDN(self):
    X, Y = self.get_samples()
    with tf.Session() as sess:
      model = KernelMixtureNetwork("kde", 2, 2, n_centers=10, n_training_epochs=10, data_normalization=True,  weight_normalization=True)
      model.fit(X, Y)
      pdf_before = model.pdf(X, Y)

      # pickle and unpickle model
      dump_string = pickle.dumps(model)
    tf.reset_default_graph()
    with tf.Session() as sess:
      model_loaded = pickle.loads(dump_string)
      pdf_after = model_loaded.pdf(X, Y)

    diff = np.sum(np.abs(pdf_after - pdf_before))
    self.assertAlmostEqual(diff, 0, places=2)

class TestRegularization(unittest.TestCase):

  def get_samples(self, std=1.0, mean=2):
    np.random.seed(22)
    data = np.random.normal([mean, mean], std, size=(2000, 2))
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

  def test5_MDN_entropy_regularization(self):
    X1, Y1 = self.get_samples(std=1, mean=2)
    X2, Y2 = self.get_samples(std=1, mean=-2)

    # DATA for GMM with two modes
    X = np.expand_dims(np.concatenate([X1, X2], axis=0), axis=1)
    Y = np.expand_dims(np.concatenate([Y1, Y2], axis=0), axis=1)

    with tf.Session() as sess:
      model_no_reg = MixtureDensityNetwork("mdn_no_entropy_reg", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                          entropy_reg_coef=0.0)
      model_no_reg.fit(X, Y)

      entropy1 = np.mean(
        sess.run(model_no_reg.softmax_entropy, feed_dict={model_no_reg.X_ph: X, model_no_reg.Y_ph: Y}))

      model_reg = MixtureDensityNetwork("mdn_entropy_reg", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                        entropy_reg_coef=10.0)
      model_reg.fit(X, Y)
      entropy2 = np.mean(sess.run(model_reg.softmax_entropy, feed_dict={model_reg.X_ph: X, model_reg.Y_ph: Y}))

      print(entropy1)
      print(entropy2)
      self.assertGreaterEqual(entropy1 / entropy2, 10)

  def test6_KMN_entropy_regularization(self):
    X1, Y1 = self.get_samples(std=1, mean=2)
    X2, Y2 = self.get_samples(std=1, mean=-2)

    # DATA for GMM with two modes
    X = np.expand_dims(np.concatenate([X1, X2], axis=0), axis=1)
    Y = np.expand_dims(np.concatenate([Y1, Y2], axis=0), axis=1)

    with tf.Session() as sess:
      model_no_reg = KernelMixtureNetwork("kmn_no_entropy_reg", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                          entropy_reg_coef=0.0)
      model_no_reg.fit(X, Y)

      entropy1 = np.mean(
        sess.run(model_no_reg.softmax_entropy, feed_dict={model_no_reg.X_ph: X, model_no_reg.Y_ph: Y}))

      model_reg = KernelMixtureNetwork("kmn_entropy_reg", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                        entropy_reg_coef=10.0)
      model_reg.fit(X, Y)
      entropy2 = np.mean(sess.run(model_reg.softmax_entropy, feed_dict={model_reg.X_ph: X, model_reg.Y_ph: Y}))

      print(entropy1)
      print(entropy2)
      self.assertGreaterEqual(entropy1 / entropy2, 10)

  def test7_data_normalization(self):
    X, Y = self.get_samples(std=2, mean=20)
    with tf.Session() as sess:
      model = KernelMixtureNetwork("kmn_data_normalization", 1, 1, n_centers=2, x_noise_std=None, y_noise_std=None,
                                          data_normalization=True, n_training_epochs=100)
      model.fit(X, Y)

      # test if data statistics were properly assigned to tf graph
      x_mean, x_std = sess.run([model.mean_x_sym, model.std_x_sym])
      print(x_mean, x_std)
      mean_diff = float(np.abs(x_mean-20))
      std_diff = float(np.abs(x_std-2))
      self.assertLessEqual(mean_diff, 0.5)
      self.assertLessEqual(std_diff, 0.5)

  def test8_data_normalization(self):
    np.random.seed(24)
    mean = 10
    std = 2
    data = np.random.normal([mean, mean, mean, mean], std, size=(2000, 4))
    X = data[:, 0:2]
    Y = data[:, 2:4]

    with tf.Session() as sess:
      model = MixtureDensityNetwork("mdn_data_normalization", 2, 2, n_centers=2, x_noise_std=None, y_noise_std=None,
                                          data_normalization=True, n_training_epochs=2000, random_seed=22)
      model.fit(X, Y)
      y_mean, y_std = sess.run([model.mean_y_sym, model.std_y_sym])
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

    with tf.Session():
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


#
# class TestConditionalDensityEstimators_fit_by_crossval(unittest.TestCase):
#   def get_samples(self):
#     np.random.seed(22)
#     data = np.concatenate([np.random.normal([i, -i], 1, size=(500, 2)) for i in range(-20, 20, 4)], axis=0)
#     X = data[:, 0]
#     Y = data[:, 1]
#     return X, Y
#
#   def test_1_KMN_with_2d_gaussian_fit_by_crossval(self):
#     X, Y = self.get_samples()
#
#     param_grid = {
#       "n_centers": [3, 10],
#       "center_sampling_method": ["k_means"],
#       "keep_edges": [True]
#     }
#
#     model = KernelMixtureNetwork(center_sampling_method="k_means", n_centers=20)
#     model.fit_by_cv(X, Y, param_grid=param_grid)
#
#     y = np.arange(-1, 5, 0.5)
#     x = np.asarray([2 for i in range(y.shape[0])])
#     p_est = model.pdf(x, y)
#     p_true = norm.pdf(y, loc=2, scale=1)
#     self.assertEqual(model.get_params()["n_centers"], 10)
#     self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)
#
#
#   def test_2_MDN_with_2d_gaussian_fit_by_crossval(self):
#     X, Y = self.get_samples()
#
#     param_grid = {
#       "n_centers": [2, 10, 50]
#     }
#
#     model = MixtureDensityNetwork()
#     model.fit_by_cv(X, Y, param_grid=param_grid)
#
#     y = np.arange(-1, 5, 0.5)
#     x = np.asarray([2 for i in range(y.shape[0])])
#     p_est = model.pdf(x, y)
#     p_true = norm.pdf(y, loc=2, scale=1)
#     self.assertEqual(model.get_params()["n_centers"], 10)
#     self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.2)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(TestConditionalDensityEstimators_2d_gaussian())
  #suite.addTest(TestConditionalDensityEstimators_fit_by_crossval())
  suite.addTest(TestHelpers())
  return suite

if __name__ == '__main__':
  warnings.filterwarnings("ignore")

  testmodules = [
   'unittests_estimators.TestHelpers',
   'unittests_estimators.TestRiskMeasures',
   'unittests_estimators.TestConditionalDensityEstimators_2d_gaussian',
   'unittests_estimators.TestRegularization',
   #'unittests_estimators.TestConditionalDensityEstimators_fit_by_crossval'
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
