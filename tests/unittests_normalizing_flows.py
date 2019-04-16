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
from cde.density_estimator import NormalizingFlowEstimator
from cde.density_estimator.normalizing_flows import InvertedPlanarFlow


class TestFlows(unittest.TestCase):
    def test_planar_invertibel(self):
        with tf.Session() as sess:
            u = tf.constant([[-2.], [1.], [10.], [2.]])
            w = tf.constant([[80.], [-1.], [1.], [1.]])
            # Compute w * û
            inv = sess.run(w*InvertedPlanarFlow._u_circ(u, w))
            for i in inv:
                self.assertGreater(i, -1.)


class Test_NF_2d_gaussian(unittest.TestCase):

    def get_samples(self, mu=2, std=1.0):
        np.random.seed(22)
        data = np.random.normal([mu, mu], std, size=(2000, 2))
        X = data[:, 0]
        Y = data[:, 1]
        return X, Y

    def test_NF_radial_with_2d_gaussian(self):
        mu = 200
        std = 23
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_radial", 1, 1, flows_type=('radial',),
                                         n_training_epochs=500, random_seed=22)
        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_NF_planar_with_2d_gaussian(self):
        mu = 200
        std = 23
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_planar", 1, 1, flows_type=('planar',),
                                         n_training_epochs=500, random_seed=22)
        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_NF_identitiy_with_2d_gaussian(self):
        mu = 200
        std = 23
        X, Y = self.get_samples(mu=mu, std=std)

        model1 = NormalizingFlowEstimator("nf_estimator_2d_planar_no_id", 1, 1, flows_type=('planar', ),
                                         n_training_epochs=50, random_seed=22)
        model2 = NormalizingFlowEstimator("nf_estimator_2d_planar_id", 1, 1, flows_type=('planar', 'identity'),
                                         n_training_epochs=50, random_seed=22)
        model1.fit(X, Y)
        model2.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p = model1.pdf(x, y)
        p_id = model2.pdf(x, y)
        self.assertLessEqual(np.mean(np.abs(p - p_id)), 0.01)

    def test_NF_chain_with_2d_gaussian(self):
        mu = 200
        std = 23
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_chain", 1, 1, flows_type=('planar', 'radial'),
                                         n_training_epochs=500, random_seed=22)
        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_NF_radial_with_2d_gaussian2(self):
        mu = -5
        std = 2.5
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_radial_2", 1, 1, flows_type=('radial',),
                                         n_training_epochs=500, random_seed=22)
        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

        p_est = model.cdf(x, y)
        p_true = norm.cdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_NF_chain_with_2d_gaussian2(self):
        mu = -5
        std = 2.5
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_chain_2", 1, 1, flows_type=('planar', 'planar', 'planar'),
                                         n_training_epochs=500, random_seed=22)

        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

        p_est = model.cdf(x, y)
        p_true = norm.cdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_NF_chain2_with_2d_gaussian2(self):
        mu = -5
        std = 2.5
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_chain2_2", 1, 1, flows_type=('radial', 'planar', 'radial'),
                                         n_training_epochs=500, random_seed=22)
        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

        p_est = model.cdf(x, y)
        p_true = norm.cdf(y, loc=mu, scale=std)
        self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


class TestMultiModal(unittest.TestCase):
    """
    This tests whether the flows can model multimodal distributions
    The distributions used aren't actually conditional distributions
    """
    def test_bi_modal_planar_chain(self):
        with tf.Session() as sess:
            bimix_gauss = tf.contrib.distributions.Mixture(
                cat=tf.distributions.Categorical(probs=[0.5, 0.5]),
                components=[
                    tf.distributions.Normal(loc=-1., scale=0.5),
                    tf.distributions.Normal(loc=+1., scale=0.5),
                ])
            x = np.ones(5000)
            y = sess.run(bimix_gauss.sample([5000]))

            model = NormalizingFlowEstimator("nf_estimator_bimodal_planar", 1, 1, flows_type=('planar', 'planar', 'planar'),
                                             n_training_epochs=1000, random_seed=22)
            model.fit(x, y)

            p_est = model.pdf(x, y)
            p_true = sess.run(bimix_gauss.prob(y))
            self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_bi_modal_mixed_chain(self):
        with tf.Session() as sess:
            bimix_gauss = tf.contrib.distributions.Mixture(
                cat=tf.distributions.Categorical(probs=[0.5, 0.5]),
                components=[
                    tf.distributions.Normal(loc=-1., scale=0.4),
                    tf.distributions.Normal(loc=+1., scale=0.4),
                ])
            x = np.ones(3000)
            y = sess.run(bimix_gauss.sample([3000]))

            model = NormalizingFlowEstimator("nf_estimator_trimodal_chain", 1, 1, flows_type=('radial', 'planar', 'radial'),
                                             n_training_epochs=1000, random_seed=22)
            model.fit(x, y)

            p_est = model.pdf(x, y)
            p_true = sess.run(bimix_gauss.prob(y))
            self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)

    def test_tri_modal_radial_chain(self):
        with tf.Session() as sess:
            bimix_gauss = tf.contrib.distributions.Mixture(
                cat=tf.distributions.Categorical(probs=[0.3, 0.4, 0.3]),
                components=[
                    tf.distributions.Normal(loc=-1., scale=0.4),
                    tf.distributions.Normal(loc=0., scale=0.4),
                    tf.distributions.Normal(loc=+1., scale=0.4),
                ])
            x = np.ones(5000)
            y = sess.run(bimix_gauss.sample([5000]))

            model = NormalizingFlowEstimator("nf_estimator_bimodal_radial", 1, 1, flows_type=('radial', 'radial', 'radial'),
                                             n_training_epochs=1000, random_seed=22)
            model.fit(x, y)

            p_est = model.pdf(x, y)
            p_true = sess.run(bimix_gauss.prob(y))
            self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


class TestLogProbability(unittest.TestCase):

    def test_NF_log_pdf(self):
        X, Y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 1))

        with tf.Session() as sess:
            model = NormalizingFlowEstimator("nf_logprob", 3, 1, flows_type=('radial', 'radial'),
                                             n_training_epochs=10, random_seed=22)
            model.fit(X, Y)

            x, y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 1))
            prob = model.pdf(x,y)
            log_prob = model.log_pdf(x,y)
            self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    testmodules = [
        'unittests_normalizing_flows.Test_NF_2d_gaussian',
        'unittests_normalizing_flows.TestLogProbability',
        'unittests_normalizing_flows.TestFlows',
        'unittests_normalizing_flows.TestMultiModal',
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
