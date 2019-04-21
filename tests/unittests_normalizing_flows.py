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
from cde.density_estimator.normalizing_flows import InvertedPlanarFlow, AffineFlow, IdentityFlow, InvertedRadialFlow


class TestFlows(unittest.TestCase):
    def test_planar_invertibel(self):
        with tf.Session() as sess:
            u = tf.constant([[-2.], [1.], [10.], [2.]])
            w = tf.constant([[80.], [-1.], [1.], [1.]])
            # Compute w * û
            inv = sess.run(w*InvertedPlanarFlow._u_circ(u, w))
            for i in inv:
                self.assertGreater(i, -1.)

    def test_affine_shift_and_scale(self):
        with tf.Session() as sess:
            base_dist = tf.distributions.Normal(loc=0., scale=1.)
            # shift the distribution three to the right
            transf_dist = tf.distributions.Normal(loc=3., scale=1.)

            flow = AffineFlow(tf.constant([[0., 3.]]), 1)
            flow_dist = tf.contrib.distributions.TransformedDistribution(distribution=base_dist, bijector=flow)

            # eval the samples so they stay constant
            samples = sess.run(base_dist.sample([1000]))

            # the output is of shape (?, 1) so it needs to be squeezed
            pdf_estimate = tf.squeeze(flow_dist.prob(samples))
            pdf_actual = transf_dist.prob(samples)
            pdf_estimate, pdf_actual = sess.run([pdf_estimate, pdf_actual])
            self.assertLessEqual(np.mean(np.abs(pdf_actual - pdf_estimate)), 0.1)

    def _test_flow_correct_dims_NN(self, flow_name):
        """
        General structure:
        flow_params = MLP(x)
        pdf(y|x) = flow(y, flow_params)

        The tensor being transformed (=y) are of shape (batch_size, event_dims)
        - batch_size = len(x) == len(y)
        - event_dims = rank(y)

        For each element of x, the MLP outputs one parametrization for the flows
        for each of these parameters, the flow transforms one element of y
        therefore len(x) == len(y)
        the event dimension describes the rank of the base probability distribution that's being transformed

        Tensorflow's MultivariateNormal doesn't implement a CDF. Therefore we switch to a Normal for 1-D Problems
        Caveat:
          MultivariateNormal PDF output shape: (batch_size, )
          UnivariateNormal PDF output shape: (batch_size, 1)
        Therefore we adapt the output shape of the ildj to be (batch_size, 1) for 1-D, (batch_size, ) for N-D

        The flows are transforming tensors (batch_size, event_size)
        Forward: (batch_size, event_size) -> (batch_size, event_size)
        Inverse: (batch_size, event_size) -> (batch_size, event_size)
        ILDJ: (batch_size, event_size) -> (batch_size, 1) [1-D] or (batch_size, ) [N-D]

        This forms a transformed distribution:
        Sample:  -> (batch_size, event_size)
        PDF: (batch_size, event_size) -> (batch_size, 1) [1-D] or (batch_size, ) [N-D]
        CDF: (batch_size, event_size) -> (batch_size, 1) [EXISTS ONLY FOR 1-D!]
        """
        tests = [
            {
                'x': [[1.], [0.], [2.], [4.], [1.]],
                'y': [[1.], [0.], [2.], [3.], [1.]],
                'ndim_x': 1,
                'ndim_y': 1
            },
            {
                'x': [[1., 1.], [0., 0.], [2., 2.], [4., 4.], [1., 1.]],
                'y': [[1., 1.], [0., 0.], [2., 2.], [3., 3.], [1., 1.]],
                'ndim_x': 2,
                'ndim_y': 2
            }
        ]
        with tf.Session() as sess:
            for test in tests:
                model = NormalizingFlowEstimator('nf_dimtest_' + flow_name + str(tests.index(test)),
                                                 test['ndim_x'], test['ndim_y'],
                                                 random_seed=22, n_training_epochs=2,
                                                 flows_type=(flow_name,))
                x, y = np.array(test['x']), np.array(test['y'])
                model.fit(x, y)
                p = model.pdf(x, y)
                self.assertEqual(p.shape, (len(y), ))
                # every test has equal first and last elements, theses are basic sanity tests
                self.assertAlmostEqual(p[0], p[-1])
                self.assertNotAlmostEqual(p[0], p[1])

    def _test_flow_correct_dims(self, flow_class):
        tests = [
            ([[1.], [2.], [1.]], 1),
            ([[1., 1.], [2., 2.], [1., 1.]], 2),
        ]
        with tf.Session() as sess:
            for test in tests:
                y, event_dims = test
                batch_size = len(y)

                y = np.array(y, dtype=np.float32)

                if event_dims == 1:
                    base_dist = tf.distributions.Normal(loc=0., scale=1.)
                else:
                    base_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.] * event_dims,
                                                                                scale_diag=[1.] * event_dims)
                params = tf.ones(shape=(batch_size, flow_class.get_param_size(event_dims)))
                flow = flow_class(params, event_dims)
                flow_dist = tf.contrib.distributions.TransformedDistribution(distribution=base_dist, bijector=flow)

                # reverse should transform (batch_size, event_dims) -> (batch_size, event_dims)
                self.assertEqual(y.shape, (batch_size, event_dims))
                inverse_y = flow.inverse(y).eval()
                self.assertEqual(inverse_y.shape, (batch_size, event_dims))

                # ildj is a reduction over event_dims
                # therefore transforms: (batch_size, event_dims) -> (batch_size, 1)
                self.assertEqual(y.shape, (batch_size, event_dims))
                ildj_y = flow.inverse_log_det_jacobian(y).eval()
                if event_dims == 1:
                    self.assertEqual(ildj_y.shape, (batch_size, 1))
                else:
                    self.assertEqual(ildj_y.shape, (batch_size, ))

                # probability: (batch_size, event_dims) -> (batch_size, 1)
                self.assertEqual(y.shape, (batch_size, event_dims))
                p = flow_dist.prob(y).eval()
                if event_dims == 1:
                    self.assertEqual(p.shape, (batch_size, 1))
                else:
                    self.assertEqual(p.shape, (batch_size, ))

                # the first an same element of every test is the same, this is a basic sanity test
                self.assertEqual(p[0], p[2])
                self.assertNotEqual(p[0], p[1])

    def test_affine_flow_correct_dimension(self):
        self._test_flow_correct_dims(AffineFlow)
        self._test_flow_correct_dims_NN('affine')

    def test_identity_flow_correct_dimension(self):
        self._test_flow_correct_dims(IdentityFlow)
        # we don't test NN dimensions for the Identity flow as it contains no trainable variables

    def test_planar_flow_correct_dimension(self):
        self._test_flow_correct_dims(InvertedPlanarFlow)
        self._test_flow_correct_dims_NN('planar')

    def test_radial_flow_correct_dimension(self):
        self._test_flow_correct_dims(InvertedRadialFlow)
        self._test_flow_correct_dims_NN('radial')


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

    def test_NF_affine_with_2d_gaussian(self):
        mu = 3
        std = 2
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_affine", 1, 1, flows_type=('affine',),
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

    def test_NF_chain_with_2d_gaussian2(self):
        mu = -5
        std = 2.5
        X, Y = self.get_samples(mu=mu, std=std)

        model = NormalizingFlowEstimator("nf_estimator_2d_chain_2", 1, 1, flows_type=('affine', 'planar', 'planar'),
                                         n_training_epochs=1000, random_seed=22)

        model.fit(X, Y)

        y = np.arange(mu - 3 * std, mu + 3 * std, 6 * std / 20)
        x = np.asarray([mu for i in range(y.shape[0])])
        p_est = model.pdf(x, y)
        p_true = norm.pdf(y, loc=mu, scale=std)
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
                    tf.distributions.Normal(loc=-.4, scale=0.4),
                    tf.distributions.Normal(loc=+.4, scale=0.4),
                ])
            x = tf.distributions.Normal(loc=0., scale=1.).sample([5000])
            y = bimix_gauss.sample([5000])
            x,y = sess.run([x, y])

            model = NormalizingFlowEstimator("nf_estimator_bimodal_planar", 1, 1, flows_type=('affine', 'planar', 'planar', 'planar'),
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
                    tf.distributions.Normal(loc=-.5, scale=0.4),
                    tf.distributions.Normal(loc=+.5, scale=0.4),
                ])
            x = tf.distributions.Normal(loc=0., scale=1.).sample([5000])
            y = bimix_gauss.sample([5000])
            x,y = sess.run([x, y])

            model = NormalizingFlowEstimator("nf_estimator_trimodal_chain", 1, 1, flows_type=('affine', 'radial', 'radial', 'radial'),
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
        X, Y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 3))

        with tf.Session() as sess:
            model = NormalizingFlowEstimator("nf_logprob", 3, 3, flows_type=('affine', 'planar'),
                                             n_training_epochs=10, random_seed=22)
            model.fit(X, Y)

            x, y = np.random.normal(size=(1000, 3)), np.random.normal(size=(1000, 3))
            prob = model.pdf(x,y)
            log_prob = model.log_pdf(x,y)
            self.assertLessEqual(np.mean(np.abs(prob - np.exp(log_prob))), 0.001)


class TestRegularization(unittest.TestCase):
    def get_samples(self, std=1.0, mean=2):
        np.random.seed(22)
        data = np.random.normal([mean, mean], std, size=(2000, 2))
        X = data[:, 0]
        Y = data[:, 1]
        return X, Y

    def test_data_normalization(self):
        X, Y = self.get_samples(std=2, mean=20)
        with tf.Session() as sess:
            model = NormalizingFlowEstimator("nf_data_normalization", 1, 1, flows_type=('affine', 'radial', 'radial'),
                                             x_noise_std=None, y_noise_std=None, data_normalization=True, n_training_epochs=100)
            model.fit(X, Y)

            # test if data statistics were properly assigned to tf graph
            x_mean, x_std = model.sess.run([model.mean_x_sym, model.std_x_sym])
            print(x_mean, x_std)
            mean_diff = float(np.abs(x_mean-20))
            std_diff = float(np.abs(x_std-2))
            self.assertLessEqual(mean_diff, 0.5)
            self.assertLessEqual(std_diff, 0.5)

    def test_bi_modal_radial_chain_w_gaussian_noise(self):
        with tf.Session() as sess:
            bimix_gauss = tf.contrib.distributions.Mixture(
                cat=tf.distributions.Categorical(probs=[0.5, 0.5]),
                components=[
                    tf.distributions.Normal(loc=-1., scale=0.5),
                    tf.distributions.Normal(loc=+1., scale=0.5),
                ])
            x = np.ones(5000)
            y = sess.run(bimix_gauss.sample([5000]))

            model = NormalizingFlowEstimator("nf_estimator_bimodal_radial_gaussian", 1, 1, flows_type=('radial', 'radial', 'radial'),
                                             data_normalization=True, x_noise_std=0.1, y_noise_std=0.1, n_training_epochs=1000, random_seed=22)
            model.fit(x, y)

            p_est = model.pdf(x, y)
            p_true = sess.run(bimix_gauss.prob(y))
            self.assertLessEqual(np.mean(np.abs(p_true - p_est)), 0.1)


class TestSerialization(unittest.TestCase):
    def get_samples(self, std=1.0):
        np.random.seed(22)
        data = np.random.normal([2, 2, 2, 2], std, size=(2000, 4))
        X = data[:, 0:2]
        Y = data[:, 2:4]
        return X, Y

    def test_pickle_unpickle_NF_estimator(self):
        X, Y = self.get_samples()
        with tf.Session() as sess:
            model = NormalizingFlowEstimator('nf_pickle', 2, 2, ('affine', 'radial', 'radial'),
                                             data_normalization=True, random_seed=22, n_training_epochs=10)
            model.fit(X, Y)
            pdf_before = model.pdf(X, Y)
            dump_string = pickle.dumps(model)
        tf.reset_default_graph()
        with tf.Session() as sess:
            model_loaded = pickle.loads(dump_string)
            pdf_after = model_loaded.pdf(X, Y)
        diff = np.sum(np.abs(pdf_after - pdf_before))
        self.assertAlmostEqual(diff, 0, places=2)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    testmodules = [
        'unittests_normalizing_flows.Test_NF_2d_gaussian',
        'unittests_normalizing_flows.TestLogProbability',
        'unittests_normalizing_flows.TestFlows',
        'unittests_normalizing_flows.TestMultiModal',
        'unittests_normalizing_flows.TestRegularization',
        'unittests_normalizing_flows.TestSerialization'
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
